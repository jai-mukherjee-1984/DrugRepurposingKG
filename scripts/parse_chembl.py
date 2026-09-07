#!/usr/bin/env python3
"""Step 1.3c - Parse & standardize ChEMBL 34.

Queries the ChEMBL SQLite database for compounds that reached at least Phase II
(max_phase >= 2) and writes:
  - chembl_compounds.csv    : chembl_id | inchikey | pref_name | max_phase | withdrawn_flag
  - chembl_indications.csv  : chembl_id | mesh_id | mesh_heading | efo_id | efo_term | max_phase_for_ind
  - chembl_atc.csv          : chembl_id | atc_code

max_phase < 4 with withdrawn_flag set (or a withdrawn/terminated indication) are
the repurposing-candidate pool; that derivation happens downstream.
"""

from __future__ import annotations

import sqlite3
import sys
import time
from pathlib import Path

import pandas as pd

DB = Path("data/raw/chembl/chembl_34/chembl_34_sqlite/chembl_34.db")
OUT_DIR = Path("data/processed")
MIN_PHASE = 2 # has to have passed safety testing (Phase I) to be considered a repurposing candidate

COMPOUNDS_SQL = f"""
SELECT md.chembl_id            AS chembl_id,
       cs.standard_inchi_key   AS inchikey,
       md.pref_name            AS pref_name,
       md.max_phase            AS max_phase,
       md.withdrawn_flag       AS withdrawn_flag
FROM   molecule_dictionary md
LEFT JOIN compound_structures cs ON cs.molregno = md.molregno
WHERE  md.max_phase >= {MIN_PHASE}
"""

INDICATIONS_SQL = f"""
SELECT md.chembl_id            AS chembl_id,
       di.mesh_id              AS mesh_id,
       di.mesh_heading         AS mesh_heading,
       di.efo_id               AS efo_id,
       di.efo_term             AS efo_term,
       di.max_phase_for_ind    AS max_phase_for_ind
FROM   drug_indication di
JOIN   molecule_dictionary md ON md.molregno = di.molregno
WHERE  md.max_phase >= {MIN_PHASE}
"""

ATC_SQL = f"""
SELECT md.chembl_id            AS chembl_id,
       mac.level5              AS atc_code
FROM   molecule_atc_classification mac
JOIN   molecule_dictionary md ON md.molregno = mac.molregno
WHERE  md.max_phase >= {MIN_PHASE}
"""


def main() -> int:
    start = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not DB.exists():
        print(f"ChEMBL DB missing: {DB} - skipped")
        return 1

    con = sqlite3.connect(f"file:{DB.as_posix()}?mode=ro", uri=True)
    try:
        compounds = pd.read_sql_query(COMPOUNDS_SQL, con)
        compounds.to_csv(OUT_DIR / "chembl_compounds.csv", index=False)
        n_ik = compounds["inchikey"].notna().sum()
        print(f"compounds (max_phase>={MIN_PHASE}): {len(compounds):,} "
              f"({n_ik:,} with InChIKey, {int(compounds['withdrawn_flag'].fillna(0).sum()):,} withdrawn)")

        indications = pd.read_sql_query(INDICATIONS_SQL, con)
        indications.to_csv(OUT_DIR / "chembl_indications.csv", index=False)
        print(f"indications: {len(indications):,} "
              f"({indications['chembl_id'].nunique():,} compounds)")

        atc = pd.read_sql_query(ATC_SQL, con)
        atc.to_csv(OUT_DIR / "chembl_atc.csv", index=False)
        print(f"atc: {len(atc):,} ({atc['chembl_id'].nunique():,} compounds)")
    finally:
        con.close()

    print(f"Done in {time.time() - start:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
