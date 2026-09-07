#!/usr/bin/env python3
"""Batch downloader for ORPHANET XML products.

This script downloads ORPHANET XML files (for example, en_product1.xml)
into a local output folder and writes a manifest with checksums and metadata.

Examples:
  python scripts/download_orphanet.py
  python scripts/download_orphanet.py --products 1 2 6 --skip-existing
  python scripts/download_orphanet.py --base-url https://www.orphadata.com/data/xml
  python scripts/download_orphanet.py --urls https://www.orphadata.com/data/xml/en_product1.xml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


DEFAULT_BASE_URL = "https://www.orphadata.com/data/xml"
DEFAULT_PRODUCTS = [1, 2, 3, 4, 5, 6, 7, 9]
DEFAULT_TIMEOUT = 60
CHUNK_SIZE = 1024 * 1024


@dataclass
class DownloadResult:
    url: str
    output_file: str
    bytes_written: int
    sha256: str
    status: str
    elapsed_seconds: float
    error: str | None = None


def build_session(retries: int = 5, backoff: float = 1.0) -> requests.Session:
    retry = Retry(
        total=retries,
        connect=retries,
        read=retries,
        backoff_factor=backoff,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET", "HEAD"),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"User-Agent": "orphanet-downloader/1.0"})
    return session


def infer_filename(url: str, response: requests.Response) -> str:
    cd = response.headers.get("content-disposition", "")
    if "filename=" in cd:
        token = cd.split("filename=", 1)[1].strip().strip('"')
        if token:
            return token

    name = Path(urlparse(url).path).name
    return name or "download.xml"


def sha256_of_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def product_urls(base_url: str, products: Iterable[int]) -> List[str]:
    base = base_url.rstrip("/")
    return [f"{base}/en_product{p}.xml" for p in products]


def download_one(
    session: requests.Session,
    url: str,
    out_dir: Path,
    timeout: int,
    skip_existing: bool,
) -> DownloadResult:
    start = time.time()

    try:
        with session.get(url, stream=True, timeout=timeout) as resp:
            if resp.status_code >= 400:
                return DownloadResult(
                    url=url,
                    output_file="",
                    bytes_written=0,
                    sha256="",
                    status="failed",
                    elapsed_seconds=round(time.time() - start, 3),
                    error=f"HTTP {resp.status_code}",
                )

            filename = infer_filename(url, resp)
            target = out_dir / filename

            if skip_existing and target.exists() and target.stat().st_size > 0:
                return DownloadResult(
                    url=url,
                    output_file=str(target),
                    bytes_written=target.stat().st_size,
                    sha256=sha256_of_file(target),
                    status="skipped_existing",
                    elapsed_seconds=round(time.time() - start, 3),
                )

            bytes_written = 0
            with target.open("wb") as f:
                for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                    if not chunk:
                        continue
                    f.write(chunk)
                    bytes_written += len(chunk)

            checksum = sha256_of_file(target)
            return DownloadResult(
                url=url,
                output_file=str(target),
                bytes_written=bytes_written,
                sha256=checksum,
                status="ok",
                elapsed_seconds=round(time.time() - start, 3),
            )

    except requests.RequestException as exc:
        return DownloadResult(
            url=url,
            output_file="",
            bytes_written=0,
            sha256="",
            status="failed",
            elapsed_seconds=round(time.time() - start, 3),
            error=str(exc),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download ORPHANET XML products.")
    parser.add_argument(
        "--output-dir",
        default="data/raw/orphanet",
        help="Directory where downloaded files are stored.",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="Base URL for ORPHANET XML products.",
    )
    parser.add_argument(
        "--products",
        type=int,
        nargs="*",
        default=DEFAULT_PRODUCTS,
        help="Product IDs to download as en_product<ID>.xml.",
    )
    parser.add_argument(
        "--urls",
        nargs="*",
        default=[],
        help="Explicit URLs to download. If provided, these are used in addition to --products.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help="Per-request timeout in seconds.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip files already present in output directory.",
    )
    parser.add_argument(
        "--manifest",
        default="download_manifest.json",
        help="Manifest filename written inside output-dir.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    urls = product_urls(args.base_url, args.products)
    if args.urls:
        urls.extend(args.urls)

    # Preserve order while removing duplicates.
    seen = set()
    deduped_urls = []
    for u in urls:
        if u not in seen:
            deduped_urls.append(u)
            seen.add(u)

    if not deduped_urls:
        print("No URLs to download.")
        return 1

    session = build_session()
    results: List[DownloadResult] = []

    print(f"Downloading {len(deduped_urls)} ORPHANET file(s) into {out_dir}...")
    for i, url in enumerate(deduped_urls, start=1):
        print(f"[{i}/{len(deduped_urls)}] {url}")
        result = download_one(
            session=session,
            url=url,
            out_dir=out_dir,
            timeout=args.timeout,
            skip_existing=args.skip_existing,
        )
        results.append(result)

        if result.status == "ok":
            print(f"  OK: {result.output_file} ({result.bytes_written} bytes)")
        elif result.status == "skipped_existing":
            print(f"  SKIP: {result.output_file}")
        else:
            print(f"  FAIL: {result.error}")

    manifest_path = out_dir / args.manifest
    payload = {
        "base_url": args.base_url,
        "products": args.products,
        "generated_at_unix": int(time.time()),
        "results": [asdict(r) for r in results],
    }
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    ok_count = sum(1 for r in results if r.status in {"ok", "skipped_existing"})
    fail_count = sum(1 for r in results if r.status == "failed")
    print(f"Done. success={ok_count}, failed={fail_count}")
    print(f"Manifest: {manifest_path}")

    return 0 if fail_count == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
