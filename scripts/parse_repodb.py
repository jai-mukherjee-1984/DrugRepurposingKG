#!/usr/bin/env python3
"""Step 1.3d - Parse & standardize REPODB.

Reads the raw REPODB CSV and writes standardized positive/negative edge sets:
  - repodb_approved.csv         (status == Approved)
  - repodb_investigational.csv  (status contains Phase/Clinical/Investigational)
  - repodb_withdrawn.csv        (status == Withdrawn/Terminated/Suspended)
  - repodb_all.csv              (union with normalized columns)

Canonical output columns:
  drug_name | drugbank_id | ind_name | umls_cui | nct | status | phase | detailed_status
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/raw/repodb")
OUT_DIR = Path("data/processed")


def load_repodb() -> pd.DataFrame:
    # Prefer the official file; fall back to repoDB.csv.
    candidates = [RAW_DIR / "repoDB_official.csv", RAW_DIR / "repoDB.csv"]
    src = next((p for p in candidates if p.exists()), None)
    if src is None:
        raise FileNotFoundError(f"No REPODB CSV found in {RAW_DIR}")

    df = pd.read_csv(src, dtype=str, keep_default_na=False, na_values=["NA", ""])
    # Normalize column names across the two file variants.
    rename = {
        "drug_id": "drugbank_id",
        "drugbank_id": "drugbank_id", # this line is useless 
        "DetailedStatus": "detailed_status",
    }
    df = df.rename(columns=rename)
    # Standard column set.
    df = df.rename(columns={"ind_id": "umls_cui", "NCT": "nct"})
    keep = ["drug_name", "drugbank_id", "ind_name", "umls_cui", "nct", "status", "phase", "detailed_status"]
    for col in keep:
        if col not in df.columns:
            df[col] = pd.NA
    df = df[keep]
    print(f"Loaded {len(df):,} rows from {src.name}")
    return df


def main() -> int:
    start = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_repodb()

    print("Status value counts:")
    print(df["status"].value_counts(dropna=False).to_string())

    status = df["status"].fillna("").str.lower()
    approved = df[status.eq("approved")]
    withdrawn = df[status.isin(["withdrawn", "terminated", "suspended"])]
    # Everything clinical / investigational that is not approved or withdrawn.
    investigational = df[~status.isin(["approved", "withdrawn", "terminated", "suspended"])]

    approved.to_csv(OUT_DIR / "repodb_approved.csv", index=False)
    investigational.to_csv(OUT_DIR / "repodb_investigational.csv", index=False)
    withdrawn.to_csv(OUT_DIR / "repodb_withdrawn.csv", index=False)
    df.to_csv(OUT_DIR / "repodb_all.csv", index=False)

    print(f"approved={len(approved):,}  investigational={len(investigational):,}  withdrawn={len(withdrawn):,}")
    print(f"Wrote 4 files to {OUT_DIR} in {time.time() - start:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
