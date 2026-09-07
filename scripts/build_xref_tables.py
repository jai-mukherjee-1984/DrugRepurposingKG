#!/usr/bin/env python3
"""Step 1.4 (part 1) - Build canonical cross-reference tables.

Canonical IDs: drugs=InChIKey, genes=Entrez, diseases=MONDO.

Outputs (data/processed/):
  - drug_xref.csv        : inchikey | chembl_id | drugbank_id | lincs_pert_id
  - gene_xref.csv        : entrez_id | symbol | hgnc_id | ensembl_gene
  - gene_ensp_entrez.csv : ensembl_protein | entrez_id   (from STRING aliases)
  - disease_xref.csv     : mondo_id | doid | orphanet_id | umls_cui | mesh_id | omim_id | efo_id
"""

from __future__ import annotations

import gzip
import sqlite3
import sys
import time
from pathlib import Path

import pandas as pd

RAW = Path("data/raw")
X = RAW / "xref"
OUT = Path("data/processed")
CHEMBL_DB = RAW / "chembl/chembl_34/chembl_34_sqlite/chembl_34.db"


# --------------------------------------------------------------------------- #
# Drugs -> InChIKey
# --------------------------------------------------------------------------- #
def build_drug_xref() -> None:
    # full chembl_id -> inchikey
    con = sqlite3.connect(f"file:{CHEMBL_DB.as_posix()}?mode=ro", uri=True)
    try:
        chembl = pd.read_sql_query(
            "SELECT md.chembl_id AS chembl_id, cs.standard_inchi_key AS inchikey "
            "FROM molecule_dictionary md "
            "JOIN compound_structures cs ON cs.molregno = md.molregno "
            "WHERE cs.standard_inchi_key IS NOT NULL",
            con,
        )
    finally:
        con.close()
    chembl2ik = dict(zip(chembl["chembl_id"], chembl["inchikey"]))
    print(f"chembl_id->inchikey: {len(chembl2ik):,}")

    # UniChem chembl <-> drugbank
    uni = pd.read_csv(X / "unichem_chembl_drugbank.txt.gz", sep="\t",
                      names=["chembl_id", "drugbank_id"], header=0, dtype=str)
    uni["inchikey"] = uni["chembl_id"].map(chembl2ik)
    print(f"unichem rows: {len(uni):,}; with inchikey: {uni['inchikey'].notna().sum():,}")

    # LINCS inchikey <-> pert_id
    lincs = pd.read_parquet(OUT / "lincs_signatures.parquet", columns=["inchikey", "pert_id"])

    # KG-relevant ChEMBL compounds only (max_phase >= 2 subset), not the full
    # 2.4M-structure universe; the full chembl2ik dict above is still used to
    # resolve DrugBank -> InChIKey via UniChem.
    chembl_kg = pd.read_csv(OUT / "chembl_compounds.csv", dtype=str,
                            usecols=["chembl_id", "inchikey"]).dropna(subset=["inchikey"])

    # assemble union keyed by inchikey
    frames = []
    frames.append(chembl_kg.assign(drugbank_id=pd.NA, lincs_pert_id=pd.NA)[
        ["inchikey", "chembl_id", "drugbank_id", "lincs_pert_id"]])
    frames.append(uni.dropna(subset=["inchikey"]).assign(lincs_pert_id=pd.NA)[
        ["inchikey", "chembl_id", "drugbank_id", "lincs_pert_id"]])
    frames.append(lincs.rename(columns={"pert_id": "lincs_pert_id"}).assign(
        chembl_id=pd.NA, drugbank_id=pd.NA)[
        ["inchikey", "chembl_id", "drugbank_id", "lincs_pert_id"]])
    allrows = pd.concat(frames, ignore_index=True)

    def first_non_null(s):
        s = s.dropna()
        return s.iloc[0] if len(s) else pd.NA

    def join_unique(s):
        vals = sorted(set(s.dropna()))
        return ";".join(vals) if vals else pd.NA

    drug_xref = allrows.groupby("inchikey", as_index=False).agg(
        chembl_id=("chembl_id", first_non_null),
        drugbank_id=("drugbank_id", join_unique),
        lincs_pert_id=("lincs_pert_id", first_non_null),
    )
    # keep only inchikeys that participate in the KG (have chembl, drugbank, or lincs)
    drug_xref = drug_xref[
        drug_xref[["chembl_id", "drugbank_id", "lincs_pert_id"]].notna().any(axis=1)
    ]
    drug_xref.to_csv(OUT / "drug_xref.csv", index=False)
    print(f"drug_xref: {len(drug_xref):,} inchikeys "
          f"(chembl={drug_xref['chembl_id'].notna().sum():,}, "
          f"drugbank={drug_xref['drugbank_id'].notna().sum():,}, "
          f"lincs={drug_xref['lincs_pert_id'].notna().sum():,})")
    # still don't fully understand what this is doing

# --------------------------------------------------------------------------- #
# Genes -> Entrez
# --------------------------------------------------------------------------- #
def build_gene_xref() -> None:
    rows = []
    with gzip.open(X / "Homo_sapiens.gene_info.gz", "rt", encoding="utf-8") as f:
        f.readline()
        for line in f:
            p = line.rstrip("\n").split("\t")
            entrez, symbol, dbx = p[1], p[2], p[5]
            hgnc = ensg = None
            if dbx and dbx != "-":
                for tok in dbx.split("|"):
                    if tok.startswith("HGNC:"):
                        hgnc = tok.split(":")[-1]          # HGNC:HGNC:5 -> 5
                    elif tok.startswith("Ensembl:"):
                        ensg = tok.split(":", 1)[1]
            rows.append((entrez, symbol, hgnc, ensg))
    gene_xref = pd.DataFrame(rows, columns=["entrez_id", "symbol", "hgnc_id", "ensembl_gene"])
    gene_xref.to_csv(OUT / "gene_xref.csv", index=False)
    print(f"gene_xref: {len(gene_xref):,} genes "
          f"(hgnc={gene_xref['hgnc_id'].notna().sum():,}, "
          f"ensg={gene_xref['ensembl_gene'].notna().sum():,})")

    # ENSP -> Entrez from STRING aliases
    ensp = {}
    with gzip.open(X / "9606.protein.aliases.v12.0.txt.gz", "rt", encoding="utf-8") as f:
        f.readline()
        for line in f:
            sp = line.rstrip("\n").split("\t")
            if len(sp) == 3 and sp[2] == "Ensembl_HGNC_entrez_id":
                ensp[sp[0].replace("9606.", "")] = sp[1]
    pd.DataFrame(sorted(ensp.items()), columns=["ensembl_protein", "entrez_id"]).to_csv(
        OUT / "gene_ensp_entrez.csv", index=False)
    print(f"gene_ensp_entrez: {len(ensp):,} ENSP->entrez")


# --------------------------------------------------------------------------- #
# Diseases -> MONDO
# --------------------------------------------------------------------------- #
def build_disease_xref() -> None:
    # parse mondo.obo term stanzas
    recs = []
    cur_id = None
    xrefs: dict[str, list[str]] = {}
    obsolete = False

    def flush():
        if cur_id and not obsolete:
            recs.append((cur_id, dict(xrefs)))

    with open(X / "mondo.obo", "r", encoding="utf-8") as f:
        in_term = False
        for line in f:
            line = line.rstrip("\n")
            if line == "[Term]":
                flush()
                cur_id, xrefs, obsolete, in_term = None, {}, False, True
            elif not in_term:
                continue
            elif line.startswith("id: MONDO:"):
                cur_id = line[4:].strip()
            elif line.startswith("is_obsolete: true"):
                obsolete = True
            elif line.startswith("xref: "):
                vals = line[6:].split(" ", 1)
                if not vals:
                    continue
                val = vals[0].strip()
                if ":" in val:
                    pre = val.split(":", 1)[0]
                    xrefs.setdefault(pre, []).append(val)
        flush()

    def get(d, *prefixes):
        for pre in prefixes:
            if pre in d:
                return ";".join(d[pre])
        return pd.NA

    out = []
    for mondo, d in recs:
        out.append({
            "mondo_id": mondo,
            "doid": get(d, "DOID"),
            "orphanet_id": get(d, "Orphanet", "ORPHA"),
            "umls_cui": get(d, "UMLS"),
            "mesh_id": get(d, "MESH"),
            "omim_id": get(d, "OMIM"),
            "efo_id": get(d, "EFO"),
        })
    disease_xref = pd.DataFrame(out)
    disease_xref.to_csv(OUT / "disease_xref.csv", index=False)
    print(f"disease_xref: {len(disease_xref):,} MONDO terms "
          f"(doid={disease_xref['doid'].notna().sum():,}, "
          f"orphanet={disease_xref['orphanet_id'].notna().sum():,}, "
          f"umls={disease_xref['umls_cui'].notna().sum():,}, "
          f"mesh={disease_xref['mesh_id'].notna().sum():,})")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    t = time.time()
    print("--- drug xref ---");    build_drug_xref()
    print("--- gene xref ---");    build_gene_xref()
    print("--- disease xref ---"); build_disease_xref()
    print(f"Done in {time.time()-t:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
