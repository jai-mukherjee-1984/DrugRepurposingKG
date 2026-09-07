#!/usr/bin/env python3
"""Step 1.3g - Parse & standardize OpenTargets gene-disease associations.

Reads the association_overall_direct parquet parts and writes:
  - opentargets_gene_disease.csv
    columns: ensembl_gene | disease_id | score | evidence_count

Filters to score >= SCORE_MIN. Excludes accidental duplicate download files.
"""

from __future__ import annotations

import glob
import sys
import time
from pathlib import Path

import pyarrow.dataset as ds
import pyarrow.compute as pc

RAW_GLOB = "data/raw/opentargets/*.parquet"
OUT_DIR = Path("data/processed")
SCORE_MIN = 0.3


def main() -> int:
    start = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Exclude accidental duplicates like "... (1).parquet".
    files = sorted(f for f in glob.glob(RAW_GLOB) if "(1)" not in f)
    if not files:
        print("No OpenTargets parquet files found - skipped")
        return 1
    print(f"Reading {len(files)} parquet part(s)")

    dataset = ds.dataset(files, format="parquet")
    cols = ["diseaseId", "targetId", "associationScore", "evidenceCount"]
    table = dataset.to_table(
        columns=cols,
        filter=(pc.field("associationScore") >= SCORE_MIN),
    )
    df = table.to_pandas()
    df = df.rename(columns={
        "targetId": "ensembl_gene",
        "diseaseId": "disease_id",
        "associationScore": "score",
        "evidenceCount": "evidence_count",
    })
    df = df.dropna(subset=["ensembl_gene", "disease_id"]).drop_duplicates(
        subset=["ensembl_gene", "disease_id"]
    )
    df = df[["ensembl_gene", "disease_id", "score", "evidence_count"]]
    out = OUT_DIR / "opentargets_gene_disease.csv"
    df.to_csv(out, index=False)

    print(f"Wrote {len(df):,} associations (score>={SCORE_MIN}) "
          f"({df['ensembl_gene'].nunique():,} genes, {df['disease_id'].nunique():,} diseases)")
    print(f"Done in {time.time() - start:.1f}s -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
