#!/usr/bin/env python3
"""Step 1.4 support - download the reference cross-reference files needed for
entity alignment. Saved under data/raw/xref/.

  - UniChem ChEMBL<->DrugBank      : src1src2.txt.gz
  - NCBI Homo_sapiens.gene_info.gz : Entrez <-> Symbol / HGNC / Ensembl(ENSG)
  - STRING 9606 protein aliases    : ENSP <-> Entrez / symbol
  - MONDO ontology (obo)           : MONDO <-> DOID / Orphanet / UMLS / MeSH / OMIM / EFO
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import requests

OUT = Path("data/raw/xref")
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

FILES = {
    "unichem_chembl_drugbank.txt.gz":
        "https://ftp.ebi.ac.uk/pub/databases/chembl/UniChem/data/wholeSourceMapping/src_id1/src1src2.txt.gz",
    "Homo_sapiens.gene_info.gz":
        "https://ftp.ncbi.nlm.nih.gov/gene/DATA/GENE_INFO/Mammalia/Homo_sapiens.gene_info.gz",
    "9606.protein.aliases.v12.0.txt.gz":
        "https://stringdb-downloads.org/download/protein.aliases.v12.0/9606.protein.aliases.v12.0.txt.gz",
    "mondo.obo":
        "http://purl.obolibrary.org/obo/mondo.obo",
}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, url in FILES.items():
        dst = OUT / name
        if dst.exists() and dst.stat().st_size > 0:
            print(f"SKIP {name} ({dst.stat().st_size/1e6:.1f} MB)")
            continue
        t = time.time()
        try:
            with requests.get(url, headers=UA, stream=True, timeout=900) as r:
                r.raise_for_status()
                with open(dst, "wb") as fh:
                    for chunk in r.iter_content(chunk_size=8 << 20):
                        fh.write(chunk)
            print(f"OK   {name}  {dst.stat().st_size/1e6:.1f} MB in {time.time()-t:.0f}s")
        except Exception as e:  # noqa: BLE001
            print(f"FAIL {name}: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
