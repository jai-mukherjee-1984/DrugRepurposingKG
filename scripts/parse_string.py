#!/usr/bin/env python3
"""Step 1.3f - Parse & standardize STRING v12 human PPI.

Reads 9606.protein.links.detailed.v12.0.txt.gz (space-delimited) and writes:
  - string_ppi.csv : protein1_ensp | protein2_ensp | combined_score

Filters to combined_score >= SCORE_MIN (high-confidence). The "9606." taxon
prefix is stripped, leaving Ensembl protein (ENSP) ids. Ensembl-protein ->
Entrez mapping is deferred to the entity-alignment step (1.4).
"""

from __future__ import annotations

import gzip
import sys
import time
from pathlib import Path

import pandas as pd

RAW = Path("data/raw/string/9606.protein.links.detailed.v12.0.txt.gz")
OUT_DIR = Path("data/processed")
SCORE_MIN = 700
CHUNK = 2_000_000


def main() -> int:
    start = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not RAW.exists():
        print(f"STRING file missing: {RAW} - skipped")
        return 1

    kept = []
    total = 0
    reader = pd.read_csv(
        RAW, sep=" ", usecols=["protein1", "protein2", "combined_score"],
        dtype={"protein1": str, "protein2": str, "combined_score": "int32"},
        chunksize=CHUNK,
    )
    for chunk in reader:
        total += len(chunk)
        hi = chunk[chunk["combined_score"] >= SCORE_MIN].copy()
        hi["protein1"] = hi["protein1"].str.replace("9606.", "", regex=False)
        hi["protein2"] = hi["protein2"].str.replace("9606.", "", regex=False)
        kept.append(hi)

    df = pd.concat(kept, ignore_index=True)
    df = df.rename(columns={"protein1": "protein1_ensp", "protein2": "protein2_ensp"})
    out = OUT_DIR / "string_ppi.csv"
    df.to_csv(out, index=False)

    n_prot = pd.unique(df[["protein1_ensp", "protein2_ensp"]].values.ravel()).size
    print(f"Read {total:,} rows; kept {len(df):,} (score>={SCORE_MIN}); {n_prot:,} proteins")
    print(f"Done in {time.time() - start:.1f}s -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
