#!/usr/bin/env python3
"""Step 1.5.0 - Define canonical KG node lists.

Builds the canonical node universe for the 4 trainable node types (drug, gene,
disease, pathway) as the union of canonical IDs across all aligned sources. The
row order written here fixes the node index used by every feature array in
Step 1.5 and by the edge index tensors in Step 1.6.

Canonical IDs: drug=InChIKey, gene=Entrez, disease=MONDO, pathway=HETIONET PC7 id.

Outputs (data/kg/):
  nodes_drug.csv     idx, inchikey, chembl_id, name
  nodes_gene.csv     idx, entrez_id, symbol
  nodes_disease.csv  idx, mondo_id, name
  nodes_pathway.csv  idx, pathway_id, name
  node_counts.txt    summary
"""
import re
import sys
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parents[1]
PROC = BASE / "data" / "processed"
KG = BASE / "data" / "kg"
KG.mkdir(parents=True, exist_ok=True)

INCHIKEY_RE = re.compile(r"^[A-Z]{14}-[A-Z]{10}-[A-Z]$")


def _read(name, **kw):
    return pd.read_csv(PROC / name, **kw)


def _clean_series(s):
    return s.dropna().astype(str).str.strip().loc[lambda x: x != ""]


def valid_inchikeys(s):
    s = _clean_series(s)
    return set(s[s.str.match(INCHIKEY_RE)])


def valid_entrez(s):
    s = _clean_series(s)
    s = s.str.replace(r"\.0$", "", regex=True)
    return set(s[s.str.fullmatch(r"\d+")])


def valid_mondo(s):
    s = _clean_series(s)
    return set(s[s.str.startswith("MONDO:")])


def main():
    het_nodes = _read("hetionet_nodes_aligned.csv")

    # ---------------- DRUG (InChIKey) ----------------
    drug_ids = set()
    chembl = _read("chembl_compounds.csv")
    drug_ids |= valid_inchikeys(chembl["inchikey"])
    drug_ids |= valid_inchikeys(
        pd.read_parquet(PROC / "lincs_signatures.parquet", columns=["inchikey"])["inchikey"]
    )
    for f in ("repodb_approved_aligned.csv", "repodb_withdrawn_aligned.csv"):
        if (PROC / f).exists():
            drug_ids |= valid_inchikeys(_read(f)["drug_inchikey"])
    het_comp = het_nodes[het_nodes.kind == "Compound"]
    drug_ids |= valid_inchikeys(het_comp["canonical_id"])

    name_map = dict(zip(chembl["inchikey"].fillna("").astype(str),
                        chembl["pref_name"].fillna("").astype(str)))
    chembl_map = dict(zip(chembl["inchikey"].fillna("").astype(str),
                          chembl["chembl_id"].fillna("").astype(str)))
    dxref = _read("drug_xref.csv")
    for ik, cid in zip(dxref["inchikey"].fillna("").astype(str),
                       dxref["chembl_id"].fillna("").astype(str)):
        chembl_map.setdefault(ik, cid if cid and cid != "nan" else "")
    het_name = dict(zip(het_comp["canonical_id"].fillna("").astype(str),
                        het_comp["name"].fillna("").astype(str)))
    drug_rows = []
    for i, ik in enumerate(sorted(drug_ids)):
        nm = name_map.get(ik) or het_name.get(ik) or ""
        nm = "" if nm == "nan" else nm
        cid = chembl_map.get(ik, "")
        cid = "" if cid == "nan" else cid
        drug_rows.append((i, ik, cid, nm))
    pd.DataFrame(drug_rows, columns=["idx", "inchikey", "chembl_id", "name"]).to_csv(
        KG / "nodes_drug.csv", index=False)

    # ---------------- GENE (Entrez) ----------------
    gene_ids = set()
    ppi = _read("string_ppi_aligned.csv", usecols=["protein1_entrez", "protein2_entrez"])
    gene_ids |= valid_entrez(ppi["protein1_entrez"])
    gene_ids |= valid_entrez(ppi["protein2_entrez"])
    ot = _read("opentargets_gene_disease_aligned.csv", usecols=["gene_entrez"])
    gene_ids |= valid_entrez(ot["gene_entrez"])
    odg = _read("orphanet_disease_gene_aligned.csv", usecols=["gene_entrez"])
    gene_ids |= valid_entrez(odg["gene_entrez"])
    het_gene = het_nodes[het_nodes.kind == "Gene"]
    gene_ids |= valid_entrez(het_gene["canonical_id"])

    gxref = _read("gene_xref.csv")
    gxref["entrez_id"] = gxref["entrez_id"].fillna("").astype(str).str.replace(r"\.0$", "", regex=True)
    sym_map = dict(zip(gxref["entrez_id"], gxref["symbol"].fillna("").astype(str)))
    gene_rows = []
    for i, g in enumerate(sorted(gene_ids, key=int)):
        sym = sym_map.get(g, "")
        sym = "" if sym == "nan" else sym
        gene_rows.append((i, g, sym))
    pd.DataFrame(gene_rows, columns=["idx", "entrez_id", "symbol"]).to_csv(
        KG / "nodes_gene.csv", index=False)

    # ---------------- DISEASE (MONDO) ----------------
    dis_ids = set()
    for f in ("repodb_approved_aligned.csv", "repodb_withdrawn_aligned.csv"):
        if (PROC / f).exists():
            dis_ids |= valid_mondo(_read(f)["disease_mondo"])
    orph = _read("orphanet_diseases_aligned.csv")
    dis_ids |= valid_mondo(orph["mondo_id"])
    dis_ids |= valid_mondo(_read("opentargets_gene_disease_aligned.csv",
                                 usecols=["disease_mondo"])["disease_mondo"])
    het_dis = het_nodes[het_nodes.kind == "Disease"]
    dis_ids |= valid_mondo(het_dis["canonical_id"])

    dname = {}
    for m, nm in zip(orph["mondo_id"].fillna("").astype(str),
                     orph["disease_name"].fillna("").astype(str)):
        if m.startswith("MONDO:") and nm and nm != "nan":
            dname.setdefault(m, nm)
    for m, nm in zip(het_dis["canonical_id"].fillna("").astype(str),
                     het_dis["name"].fillna("").astype(str)):
        if m.startswith("MONDO:") and nm and nm != "nan":
            dname.setdefault(m, nm)
    dis_rows = []
    for i, m in enumerate(sorted(dis_ids)):
        dis_rows.append((i, m, dname.get(m, "")))
    pd.DataFrame(dis_rows, columns=["idx", "mondo_id", "name"]).to_csv(
        KG / "nodes_disease.csv", index=False)

    # ---------------- PATHWAY (HETIONET PC7) ----------------
    het_pw = het_nodes[het_nodes.kind == "Pathway"].copy()
    het_pw["canonical_id"] = het_pw["canonical_id"].fillna("").astype(str)
    het_pw = het_pw.drop_duplicates("canonical_id").sort_values("canonical_id")
    pw_rows = [(i, pid, nm if nm != "nan" else "")
               for i, (pid, nm) in enumerate(zip(het_pw["canonical_id"],
                                                 het_pw["name"].fillna("").astype(str)))]
    pd.DataFrame(pw_rows, columns=["idx", "pathway_id", "name"]).to_csv(
        KG / "nodes_pathway.csv", index=False)

    summary = (
        f"drug    {len(drug_rows):>7}\n"
        f"gene    {len(gene_rows):>7}\n"
        f"disease {len(dis_rows):>7}\n"
        f"pathway {len(pw_rows):>7}\n"
        f"total   {len(drug_rows)+len(gene_rows)+len(dis_rows)+len(pw_rows):>7}\n"
    )
    with open(KG / "node_counts.txt", "w") as fh:
        fh.write(summary)
    print(summary)


if __name__ == "__main__":
    sys.exit(main())
