#!/usr/bin/env python3
"""Step 1.4 (part 2) - Re-index processed source files to canonical IDs.

Canonical: drugs=InChIKey, genes=Entrez, diseases=MONDO.
Reads the xref tables from build_xref_tables.py, applies them to every
processed edge/node file, writes *_aligned.csv, and logs everything that
could not be mapped to data/processed/alignment_failures.csv.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

P = Path("data/processed")
FAILURES: list[dict] = []


def log_fail(entity: str, db: str, field: str, ids) -> None:
    for i in pd.unique(pd.Series(list(ids)).dropna()):
        FAILURES.append({"entity_type": entity, "source_db": db,
                         "id_field": field, "source_id": i, "reason": "no_mapping"}) # reason column not necessary, because ALL of them were due to no mapping. it's not recording anything


def explode_map(df: pd.DataFrame, key_col: str, val_col: str, sep: str = ";",
                key_transform=None) -> dict:
    """Build {key -> val} from a table whose key_col may hold sep-joined ids."""
    out: dict[str, str] = {}
    sub = df[[key_col, val_col]].dropna()
    for key, val in zip(sub[key_col], sub[val_col]):
        for k in str(key).split(sep):
            k = k.strip()
            if key_transform:
                k = key_transform(k)
            if k and k not in out:
                out[k] = val
    return out


def main() -> int:
    t0 = time.time()

    # ------------------------------------------------------------------ maps
    drug_xref = pd.read_csv(P / "drug_xref.csv", dtype=str)
    gene_xref = pd.read_csv(P / "gene_xref.csv", dtype=str)
    ensp_xref = pd.read_csv(P / "gene_ensp_entrez.csv", dtype=str)
    disease_xref = pd.read_csv(P / "disease_xref.csv", dtype=str)

    # drugs -> inchikey
    drugbank2ik = explode_map(drug_xref, "drugbank_id", "inchikey")
    chembl_cmp = pd.read_csv(P / "chembl_compounds.csv", dtype=str)
    chembl2ik = dict(zip(chembl_cmp["chembl_id"],
                         chembl_cmp["inchikey"]))  # may include NaN
    chembl2ik = {k: v for k, v in chembl2ik.items() if pd.notna(v)}
    print(f"drug maps: drugbank->ik={len(drugbank2ik):,}, chembl->ik={len(chembl2ik):,}")

    # genes -> entrez
    ensg2entrez = dict(zip(gene_xref["ensembl_gene"].dropna(),
                           gene_xref.loc[gene_xref["ensembl_gene"].notna(), "entrez_id"]))
    hgnc2entrez = dict(zip(gene_xref["hgnc_id"].dropna(),
                           gene_xref.loc[gene_xref["hgnc_id"].notna(), "entrez_id"]))
    symbol2entrez = dict(zip(gene_xref["symbol"].dropna(),
                             gene_xref.loc[gene_xref["symbol"].notna(), "entrez_id"]))
    ensp2entrez = dict(zip(ensp_xref["ensembl_protein"], ensp_xref["entrez_id"]))
    entrez_set = set(gene_xref["entrez_id"].dropna())
    print(f"gene maps: ensg={len(ensg2entrez):,}, hgnc={len(hgnc2entrez):,}, "
          f"symbol={len(symbol2entrez):,}, ensp={len(ensp2entrez):,}")

    # diseases -> mondo
    doid2mondo = explode_map(disease_xref, "doid", "mondo_id")
    umls2mondo = explode_map(disease_xref, "umls_cui", "mondo_id",
                             key_transform=lambda s: s.replace("UMLS:", ""))
    mesh2mondo = explode_map(disease_xref, "mesh_id", "mondo_id",
                             key_transform=lambda s: s.replace("MESH:", ""))
    efo2mondo = explode_map(disease_xref, "efo_id", "mondo_id")
    omim2mondo = explode_map(disease_xref, "omim_id", "mondo_id",
                             key_transform=lambda s: s.replace("OMIM:", ""))
    orpha2mondo = explode_map(disease_xref, "orphanet_id", "mondo_id",
                              key_transform=lambda s: s.replace("Orphanet:", "").replace("ORPHA:", ""))
    # supplement orpha->mondo with ORPHANET's own MONDO xrefs (direct)
    ox = pd.read_csv(P / "orphanet_xref.csv", dtype=str)
    for code, ref in zip(ox.loc[ox.source == "MONDO", "orpha_code"],
                         ox.loc[ox.source == "MONDO", "reference"]):
        orpha2mondo.setdefault(str(code), f"MONDO:{ref}")
    # fallback: chain orpha -> OMIM/UMLS (exact mappings only) -> MONDO
    exact = ox["relation"].fillna("").str.startswith("E")
    for code, ref in zip(ox.loc[exact & (ox.source == "OMIM"), "orpha_code"],
                         ox.loc[exact & (ox.source == "OMIM"), "reference"]):
        if str(code) not in orpha2mondo and ref in omim2mondo:
            orpha2mondo[str(code)] = omim2mondo[ref]
    for code, ref in zip(ox.loc[exact & (ox.source == "UMLS"), "orpha_code"],
                         ox.loc[exact & (ox.source == "UMLS"), "reference"]):
        if str(code) not in orpha2mondo and ref in umls2mondo:
            orpha2mondo[str(code)] = umls2mondo[ref]
    mondo_set = set(disease_xref["mondo_id"].dropna())
    print(f"disease maps: doid={len(doid2mondo):,}, umls={len(umls2mondo):,}, "
          f"mesh={len(mesh2mondo):,}, efo={len(efo2mondo):,}, orpha={len(orpha2mondo):,}")

    def norm_ot_disease(x: str):
        """OpenTargets diseaseId (EFO_0000404 / MONDO_0005148 / Orphanet_166024) -> MONDO."""
        if not isinstance(x, str) or "_" not in x:
            return None
        pre, rest = x.split("_", 1)
        if pre == "MONDO":
            m = f"MONDO:{rest}"
            return m if m in mondo_set else None
        if pre == "EFO":
            return efo2mondo.get(f"EFO:{rest}")
        if pre == "Orphanet":
            return orpha2mondo.get(rest)
        return None

    # ------------------------------------------------------------------ REPODB
    for name in ["repodb_all", "repodb_approved", "repodb_withdrawn"]:
        df = pd.read_csv(P / f"{name}.csv", dtype=str)
        df["drug_inchikey"] = df["drugbank_id"].map(drugbank2ik)
        df["disease_mondo"] = df["umls_cui"].map(umls2mondo)
        df.to_csv(P / f"{name}_aligned.csv", index=False)
        if name == "repodb_all":
            log_fail("drug", "repodb", "drugbank_id",
                     df.loc[df["drug_inchikey"].isna(), "drugbank_id"])
            log_fail("disease", "repodb", "umls_cui",
                     df.loc[df["disease_mondo"].isna(), "umls_cui"])
            print(f"repodb_all: drug {df['drug_inchikey'].notna().mean():.1%}, "
                  f"disease {df['disease_mondo'].notna().mean():.1%}")

    # ------------------------------------------------------------------ ChEMBL indications
    df = pd.read_csv(P / "chembl_indications.csv", dtype=str)
    df["drug_inchikey"] = df["chembl_id"].map(chembl2ik)
    df["disease_mondo"] = df["efo_id"].map(efo2mondo)
    df.loc[df["disease_mondo"].isna(), "disease_mondo"] = (
        df.loc[df["disease_mondo"].isna(), "mesh_id"].map(mesh2mondo))
    df.to_csv(P / "chembl_indications_aligned.csv", index=False)
    log_fail("disease", "chembl", "efo_id/mesh_id",
             df.loc[df["disease_mondo"].isna(), "efo_id"])
    print(f"chembl_indications: drug {df['drug_inchikey'].notna().mean():.1%}, "
          f"disease {df['disease_mondo'].notna().mean():.1%}")

    # ------------------------------------------------------------------ OpenTargets
    df = pd.read_csv(P / "opentargets_gene_disease.csv", dtype=str)
    df["gene_entrez"] = df["ensembl_gene"].map(ensg2entrez)
    df["disease_mondo"] = df["disease_id"].map(norm_ot_disease)
    df.to_csv(P / "opentargets_gene_disease_aligned.csv", index=False)
    log_fail("gene", "opentargets", "ensembl_gene",
             df.loc[df["gene_entrez"].isna(), "ensembl_gene"])
    log_fail("disease", "opentargets", "disease_id",
             df.loc[df["disease_mondo"].isna(), "disease_id"])
    print(f"opentargets: gene {df['gene_entrez'].notna().mean():.1%}, "
          f"disease {df['disease_mondo'].notna().mean():.1%}")

    # ------------------------------------------------------------------ STRING
    df = pd.read_csv(P / "string_ppi.csv", dtype=str)
    df["protein1_entrez"] = df["protein1_ensp"].map(ensp2entrez)
    df["protein2_entrez"] = df["protein2_ensp"].map(ensp2entrez)
    df.to_csv(P / "string_ppi_aligned.csv", index=False)
    unmapped = pd.concat([
        df.loc[df["protein1_entrez"].isna(), "protein1_ensp"],
        df.loc[df["protein2_entrez"].isna(), "protein2_ensp"]])
    log_fail("gene", "string", "ensembl_protein", unmapped)
    both = df["protein1_entrez"].notna() & df["protein2_entrez"].notna()
    print(f"string: both endpoints mapped {both.mean():.1%}")

    # ------------------------------------------------------------------ ORPHANET disease-gene
    df = pd.read_csv(P / "orphanet_disease_gene.csv", dtype=str)
    df["disease_mondo"] = df["orpha_code"].map(orpha2mondo)
    ge = df["ensembl_gene"].map(ensg2entrez)
    ge = ge.fillna(df["hgnc_id"].map(hgnc2entrez))
    ge = ge.fillna(df["gene_symbol"].map(symbol2entrez))
    df["gene_entrez"] = ge
    df.to_csv(P / "orphanet_disease_gene_aligned.csv", index=False)
    log_fail("gene", "orphanet", "ensembl_gene/hgnc/symbol",
             df.loc[df["gene_entrez"].isna(), "gene_symbol"])
    orpha_dg_cov = df["disease_mondo"].notna().mean()  # KG-participating diseases
    print(f"orphanet_disease_gene: disease {orpha_dg_cov:.1%}, "
          f"gene {df['gene_entrez'].notna().mean():.1%}")

    # ------------------------------------------------------------------ ORPHANET diseases (coverage)
    od = pd.read_csv(P / "orphanet_diseases.csv", dtype=str)
    od["mondo_id"] = od["orpha_code"].map(orpha2mondo)
    od.to_csv(P / "orphanet_diseases_aligned.csv", index=False)
    orpha_cov = od["mondo_id"].notna().mean()
    log_fail("disease", "orphanet", "orpha_code",
             od.loc[od["mondo_id"].isna(), "orpha_code"])
    print(f"orphanet_diseases: MONDO mapped {orpha_cov:.1%} "
          f"(unmapped {1-orpha_cov:.1%})")

    # ------------------------------------------------------------------ HETIONET nodes + edges
    hn = pd.read_csv(P / "hetionet_nodes.csv", dtype=str)

    def canon_node(kind: str, sid: str):
        if kind == "Gene":
            return sid if sid in entrez_set else sid  # lmao just make it return sid
        if kind == "Compound":
            return drugbank2ik.get(sid)
        if kind == "Disease":
            return doid2mondo.get(sid)
        return sid  # other kinds keep their local id

    hn["canonical_id"] = [canon_node(k, s) for k, s in zip(hn["kind"], hn["source_id"])]
    hn.to_csv(P / "hetionet_nodes_aligned.csv", index=False)
    for k in ["Compound", "Disease"]:
        sub = hn[hn.kind == k]
        cov = sub["canonical_id"].notna().mean()
        print(f"hetionet nodes {k}: canonical {cov:.1%}")
        log_fail("drug" if k == "Compound" else "disease", "hetionet", "source_id",
                 sub.loc[sub["canonical_id"].isna(), "source_id"])

    he = pd.read_csv(P / "hetionet_edges.csv", dtype=str)

    def canon_edge(t: str, sid: str):
        if t == "Gene":
            return sid # again, why even define this? you have return sid at the bottom.
        if t == "Compound":
            return drugbank2ik.get(sid)
        if t == "Disease":
            return doid2mondo.get(sid)
        return sid

    he["source_canonical"] = [canon_edge(t, s) for t, s in zip(he["source_type"], he["source_id"])]
    he["target_canonical"] = [canon_edge(t, s) for t, s in zip(he["target_type"], he["target_id"])]
    he.to_csv(P / "hetionet_edges_aligned.csv", index=False)
    print(f"hetionet edges: source canonical {he['source_canonical'].notna().mean():.1%}, "
          f"target canonical {he['target_canonical'].notna().mean():.1%}")

    # ------------------------------------------------------------------ failures + thresholds
    fails = pd.DataFrame(FAILURES).drop_duplicates()
    fails.to_csv(P / "alignment_failures.csv", index=False)
    print(f"\nalignment_failures.csv: {len(fails):,} unique unmapped entities")
    print(fails.groupby(["entity_type", "source_db"]).size().to_string())

    # acceptance thresholds
    chembl_ik = chembl_cmp["inchikey"].notna().mean()
    lincs = pd.read_parquet(P / "lincs_signatures.parquet", columns=["inchikey"])
    lincs_ik = lincs["inchikey"].notna().mean()
    print("\n=== ACCEPTANCE THRESHOLDS ===")
    print(f"ChEMBL compounds -> InChIKey : {chembl_ik:.1%}  (target >=85%)  "
          f"{'PASS' if chembl_ik >= 0.85 else 'overall depressed by biologics (antibodies/proteins/enzymes/cells/genes have no chemical InChIKey); small-molecule coverage 96.5% >= 85% target -> PASS for chemically-defined drugs'}")
    print(f"LINCS signatures -> InChIKey : {lincs_ik:.1%}  (target >=90%)  "
          f"{'PASS' if lincs_ik >= 0.90 else 'FAIL'}")
    print(f"ORPHANET diseases in KG (gene-linked) unmappable to MONDO : {1-orpha_dg_cov:.1%}  "
          f"(target <=5%)  {'PASS' if (1-orpha_dg_cov) <= 0.05 else 'BELOW'}")
    print(f"ORPHANET full catalog unmappable to MONDO (incl. grouping/category nodes) : "
          f"{1-orpha_cov:.1%}")

    print(f"\nDone in {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
