#!/usr/bin/env python
"""Step 1.6 - Assemble the PyG HeteroData object.

Builds data/kg/heterodata.pt from:
  - node feature arrays in embeddings/*.npy  (row order = data/kg/nodes_*.csv)
  - 7 heterogeneous edge types built from data/processed/*_aligned.csv + LINCS/HPO

Node types (4): drug, gene, disease, pathway
Edge types (7):
  (drug,    targets,    gene)     <- HETIONET CbG
  (drug,    treats,     disease)  <- REPODB + ChEMBL indications + HETIONET CtD/CpD
  (drug,    similar_to, drug)     <- LINCS L1000 cosine kNN (symmetric)
  (gene,    interacts,  gene)     <- STRING PPI (symmetric)
  (gene,    associated, disease)  <- OpenTargets + HETIONET DaG + ORPHANET
  (gene,    member_of,  pathway)  <- HETIONET GpPW
  (disease, similar_to, disease)  <- ORPHANET HPO Jaccard (symmetric)

Edge attrs (float [E,1]) where meaningful: STRING conf, OpenTargets score, LINCS cos, HPO Jaccard.

Run:  ~/pyg-b10/bin/python scripts/build_heterodata.py
"""
import os, sys, time
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

t0 = time.time()
KG = "data/kg"
PROC = "data/processed"
EMB = "embeddings"
os.makedirs(KG, exist_ok=True)

# a timing function for different steps of the code
def log(*a):
    print(f"[{time.time()-t0:6.1f}s]", *a, flush=True)


# ----------------------------------------------------------------------------
# 1. Node index maps (keys normalized to str)
# ----------------------------------------------------------------------------
def norm_entrez(s):
    """Normalize an Entrez id (may be float/'123.0'/'123') to canonical '123'."""
    out = pd.to_numeric(s, errors="coerce")
    return out


nodes_drug = pd.read_csv(f"{KG}/nodes_drug.csv")
nodes_gene = pd.read_csv(f"{KG}/nodes_gene.csv")
nodes_dis = pd.read_csv(f"{KG}/nodes_disease.csv")
nodes_pw = pd.read_csv(f"{KG}/nodes_pathway.csv")

drug_map = {k: i for i, k in zip(nodes_drug.idx, nodes_drug.inchikey.astype(str))}
gene_ser = norm_entrez(nodes_gene.entrez_id)
gene_map = {int(k): i for i, k in zip(nodes_gene.idx, gene_ser) if pd.notna(k)}
dis_map = {k: i for i, k in zip(nodes_dis.idx, nodes_dis.mondo_id.astype(str))}
pw_map = {k: i for i, k in zip(nodes_pw.idx, nodes_pw.pathway_id.astype(str))}

N = {"drug": len(nodes_drug), "gene": len(nodes_gene),
     "disease": len(nodes_dis), "pathway": len(nodes_pw)}
log("node counts", N)


def map_drug(s):
    return s.astype(str).map(drug_map)


def map_gene(s):
    return norm_entrez(s).map(lambda x: gene_map.get(int(x)) if pd.notna(x) else np.nan)


def map_dis(s):
    return s.astype(str).map(dis_map)


def map_pw(s):
    return s.astype(str).map(pw_map)


def to_ei(src_idx, dst_idx):
    """Stack two index series into an int64 edge_index tensor, dropping NaN rows."""
    df = pd.DataFrame({"s": src_idx.values, "d": dst_idx.values}).dropna()
    df = df.astype(np.int64)
    return torch.tensor(np.vstack([df.s.values, df.d.values]), dtype=torch.long), len(df)


def dedup(ei, attr=None):
    """Remove duplicate (s,d) columns; keep first attr if provided."""
    key = ei[0].numpy().astype(np.int64) * (1 << 32) + ei[1].numpy().astype(np.int64)
    _, keep = np.unique(key, return_index=True)
    keep.sort()
    ei2 = ei[:, keep]
    if attr is not None:
        attr = attr[keep]
    return ei2, attr


def symmetrize(ei, attr=None):
    """Add reverse edges then dedup, so already-bidirectional sources (STRING, kNN,
    Jaccard) don't yield duplicate directed edges."""
    rev = ei.flip(0)
    ei2 = torch.cat([ei, rev], dim=1)
    if attr is not None:
        attr = torch.cat([attr, attr], dim=0)
    return dedup(ei2, attr)


data = HeteroData()

# ----------------------------------------------------------------------------
# 2. Node features
# ----------------------------------------------------------------------------
feat_files = {
    "drug": "drug_features.npy",
    "gene": "gene_esm2_3b.npy",
    "disease": "disease_sbert.npy",
    "pathway": "pathway_onehot.npy",
}
for nt, fn in feat_files.items():
    x = np.load(f"{EMB}/{fn}")
    assert x.shape[0] == N[nt], f"{nt} feat rows {x.shape[0]} != {N[nt]}"
    data[nt].x = torch.from_numpy(x.astype(np.float32))
    log(f"feat {nt:8s} {tuple(x.shape)}")

# ----------------------------------------------------------------------------
# 3. Load HETIONET once
# ----------------------------------------------------------------------------
he = pd.read_csv(f"{PROC}/hetionet_edges_aligned.csv", low_memory=False)


def he_meta(me):
    return he[he.metaedge == me]


# --- (drug, targets, gene)  <- CbG ---
cbg = he_meta("CbG")
ei, n = to_ei(map_drug(cbg.source_canonical), map_gene(cbg.target_canonical))
ei, _ = dedup(ei)
data["drug", "targets", "gene"].edge_index = ei
log(f"drug-targets-gene    {ei.shape[1]:>8d} edges (from {n} CbG)")

# --- (drug, treats, disease)  <- REPODB + ChEMBL + HETIONET CtD/CpD ---
repo = pd.read_csv(f"{PROC}/repodb_all_aligned.csv", low_memory=False)
ei_r, _ = to_ei(map_drug(repo.drug_inchikey), map_dis(repo.disease_mondo))
chem = pd.read_csv(f"{PROC}/chembl_indications_aligned.csv", low_memory=False)
ei_c, _ = to_ei(map_drug(chem.drug_inchikey), map_dis(chem.disease_mondo))
ctd = he_meta("CtD"); cpd = he_meta("CpD")
he_treat = pd.concat([ctd, cpd])
ei_h, _ = to_ei(map_drug(he_treat.source_canonical), map_dis(he_treat.target_canonical))
ei = torch.cat([ei_r, ei_c, ei_h], dim=1)
ei, _ = dedup(ei)
data["drug", "treats", "disease"].edge_index = ei
log(f"drug-treats-disease  {ei.shape[1]:>8d} edges "
    f"(repodb {ei_r.shape[1]}, chembl {ei_c.shape[1]}, hetio {ei_h.shape[1]})")

# --- (gene, interacts, gene)  <- STRING (symmetric) ---
st = pd.read_csv(f"{PROC}/string_ppi_aligned.csv", low_memory=False)
df = pd.DataFrame({
    # source, destination, weight
    "s": map_gene(st.protein1_entrez).values,
    "d": map_gene(st.protein2_entrez).values,
    "w": (st.combined_score.astype(np.float32) / 1000.0).values,
}).dropna()
df = df[df.s != df.d]
ei = torch.tensor(np.vstack([df.s.values.astype(np.int64), df.d.values.astype(np.int64)]), dtype=torch.long)
attr = torch.tensor(df.w.values, dtype=torch.float32).unsqueeze(1)
ei, attr = dedup(ei, attr)
ei, attr = symmetrize(ei, attr)
data["gene", "interacts", "gene"].edge_index = ei
data["gene", "interacts", "gene"].edge_attr = attr
log(f"gene-interacts-gene  {ei.shape[1]:>8d} edges (symmetric)")

# --- (gene, associated, disease)  <- OpenTargets + HETIONET DaG + ORPHANET ---
ot = pd.read_csv(f"{PROC}/opentargets_gene_disease_aligned.csv", low_memory=False)
df_ot = pd.DataFrame({
    "s": map_gene(ot.gene_entrez).values,
    "d": map_dis(ot.disease_mondo).values,
    "w": ot.score.astype(np.float32).values,
}).dropna()
dag = he_meta("DaG")
df_h = pd.DataFrame({
    "s": map_gene(dag.target_canonical).values,     # DaG: source=Disease, target=Gene
    "d": map_dis(dag.source_canonical).values,
    "w": np.ones(len(dag), dtype=np.float32),
}).dropna()
org = pd.read_csv(f"{PROC}/orphanet_disease_gene_aligned.csv", low_memory=False)
df_or = pd.DataFrame({
    "s": map_gene(org.gene_entrez).values,
    "d": map_dis(org.disease_mondo).values,
    "w": np.ones(len(org), dtype=np.float32),
}).dropna()
df = pd.concat([df_ot, df_h, df_or], ignore_index=True)
ei = torch.tensor(np.vstack([df.s.values.astype(np.int64), df.d.values.astype(np.int64)]), dtype=torch.long)
attr = torch.tensor(df.w.values, dtype=torch.float32).unsqueeze(1)
ei, attr = dedup(ei, attr)
data["gene", "associated", "disease"].edge_index = ei
data["gene", "associated", "disease"].edge_attr = attr
log(f"gene-associated-dis  {ei.shape[1]:>8d} edges "
    f"(ot {len(df_ot)}, dag {len(df_h)}, orph {len(df_or)})")

# --- (gene, member_of, pathway)  <- GpPW ---
gp = he_meta("GpPW")
ei, n = to_ei(map_gene(gp.source_canonical), map_pw(gp.target_canonical))
ei, _ = dedup(ei)
data["gene", "member_of", "pathway"].edge_index = ei
log(f"gene-member_of-pw    {ei.shape[1]:>8d} edges (from {n} GpPW)")

# --- (drug, similar_to, drug)  <- LINCS cosine kNN (symmetric) ---
# Pure top-k kNN with a mild positive-cosine floor (0.0): every profiled drug keeps its
# nearest transcriptomic neighbors so LINCS-profiled drugs are not stranded by a hard cutoff.
# The cosine is retained as edge_attr so the GAT can down-weight weak similarities.
K_DRUG, THR_DRUG = 10, 0.0
lincs = pd.read_parquet(f"{PROC}/lincs_signatures.parquet")
lm_cols = [c for c in lincs.columns if c.startswith("lm_")]
lincs = lincs.groupby("inchikey", as_index=False)[lm_cols].mean()   # 1 vec per inchikey
lincs["idx"] = lincs.inchikey.astype(str).map(drug_map)
lincs = lincs.dropna(subset=["idx"])
lincs["idx"] = lincs.idx.astype(np.int64)
sig = torch.tensor(lincs[lm_cols].values, dtype=torch.float32)
sig = torch.nn.functional.normalize(sig, dim=1)
dev = "cuda" if torch.cuda.is_available() else "cpu"
sig = sig.to(dev)
idx_vec = torch.tensor(lincs.idx.values, dtype=torch.long, device=dev)
rows_s, rows_d, rows_w = [], [], []
M = sig.shape[0]
CH = 2048
for i in range(0, M, CH):
    block = sig[i:i + CH] @ sig.T                      # [b, M] cosine
    b = block.shape[0]
    for r in range(b):
        block[r, i + r] = -2.0                          # drop self
    vals, nbr = torch.topk(block, K_DRUG, dim=1)
    keep = vals >= THR_DRUG
    src = idx_vec[i:i + CH].unsqueeze(1).expand(-1, K_DRUG)[keep]
    dst = idx_vec[nbr][keep]
    w = vals[keep]
    rows_s.append(src.cpu()); rows_d.append(dst.cpu()); rows_w.append(w.cpu())
src = torch.cat(rows_s); dst = torch.cat(rows_d); w = torch.cat(rows_w)
ei = torch.stack([src, dst]).long()
attr = w.unsqueeze(1).float()
ei, attr = dedup(ei, attr)
ei, attr = symmetrize(ei, attr)
data["drug", "similar_to", "drug"].edge_index = ei
data["drug", "similar_to", "drug"].edge_attr = attr
log(f"drug-similar-drug    {ei.shape[1]:>8d} edges (symmetric, {M} profiled drugs)")

# --- (disease, similar_to, disease) <- ORPHANET HPO Jaccard (top-k) + MONDO is_a ---
# HPO relaxed to a pure top-k kNN (floor 0.0) so HPO-bearing diseases are not stranded by a
# hard Jaccard cutoff; MONDO ontology parent-child (is_a) edges are folded into the same
# relation so ontology-connected diseases without HPO terms also join the graph. Both are
# "disease relatedness"; edge_attr = Jaccard for HPO edges, 1.0 for ontology edges.
K_DIS, THR_DIS = 10, 0.0
import scipy.sparse as sp
hpo = pd.read_csv(f"{PROC}/orphanet_hpo.csv", low_memory=False)
odis = pd.read_csv(f"{PROC}/orphanet_diseases_aligned.csv", low_memory=False)
orpha2mondo = {int(o): m for o, m in zip(odis.orpha_code, odis.mondo_id.astype(str)) if pd.notna(m)}
hpo = hpo.copy()
hpo["mondo"] = hpo.orpha_code.map(lambda o: orpha2mondo.get(int(o)) if pd.notna(o) else None)
hpo["didx"] = hpo.mondo.map(dis_map)
hpo = hpo.dropna(subset=["didx"])
hpo["didx"] = hpo.didx.astype(np.int64)
uniq_dis = np.sort(hpo.didx.unique())
d_pos = {d: i for i, d in enumerate(uniq_dis)}
uniq_hpo = {h: i for i, h in enumerate(hpo.hpo_id.unique())}
rows = hpo.didx.map(d_pos).values
cols = hpo.hpo_id.map(uniq_hpo).values
Mbin = sp.csr_matrix((np.ones(len(rows)), (rows, cols)),
                     shape=(len(uniq_dis), len(uniq_hpo)), dtype=np.float32)
sizes = np.asarray(Mbin.sum(1)).ravel()
inter = (Mbin @ Mbin.T).tocoo()
rows_s, rows_d, rows_w = [], [], []
from collections import defaultdict
buck = defaultdict(list)
# the jaccard similarity
for a, b, v in zip(inter.row, inter.col, inter.data):
    if a == b:
        continue
    union = sizes[a] + sizes[b] - v
    if union <= 0:
        continue
    j = v / union
    if j >= THR_DIS:
        buck[a].append((j, b))
for a, lst in buck.items():
    lst.sort(reverse=True)
    for j, b in lst[:K_DIS]:
        rows_s.append(uniq_dis[a]); rows_d.append(uniq_dis[b]); rows_w.append(j)
n_hpo_edges = len(rows_s)

# MONDO ontology is_a parent-child edges (both endpoints in disease node set)
obo_path = "data/raw/xref/mondo.obo"
n_mondo_edges = 0
if os.path.exists(obo_path):
    cur = None
    with open(obo_path, encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.rstrip("\n")
            if ln == "[Term]":
                cur = None
            elif ln.startswith("id: MONDO:"):
                cur = ln[4:].strip()
            elif ln.startswith("is_a: MONDO:") and cur is not None:
                par = ln.split("is_a:", 1)[1].split("!")[0].strip()
                ci, pi = dis_map.get(cur), dis_map.get(par)
                if ci is not None and pi is not None:
                    rows_s.append(ci); rows_d.append(pi); rows_w.append(1.0)
                    n_mondo_edges += 1
if rows_s:
    ei = torch.tensor(np.vstack([rows_s, rows_d]), dtype=torch.long)
    attr = torch.tensor(rows_w, dtype=torch.float32).unsqueeze(1)
    ei, attr = dedup(ei, attr)
    ei, attr = symmetrize(ei, attr)
else:
    ei = torch.zeros((2, 0), dtype=torch.long); attr = torch.zeros((0, 1))
data["disease", "similar_to", "disease"].edge_index = ei
data["disease", "similar_to", "disease"].edge_attr = attr
log(f"disease-similar-dis  {ei.shape[1]:>8d} edges (symmetric; "
    f"HPO {n_hpo_edges}, MONDO is_a {n_mondo_edges})")

# ----------------------------------------------------------------------------
# 4. Verify + save
# ----------------------------------------------------------------------------
total_nodes = sum(N.values())
total_edges = sum(data[et].edge_index.shape[1] for et in data.edge_types)
log("metadata:", data.metadata())
log(f"TOTAL nodes={total_nodes:,}  edges={total_edges:,}")

# NaN/self-loop sanity
for nt in data.node_types:
    assert not torch.isnan(data[nt].x).any(), f"NaN in {nt}.x"
# basically check no node is connecting to itself
for et in data.edge_types:
    ei = data[et].edge_index
    if et[0] == et[2]:
        sl = int((ei[0] == ei[1]).sum())
        assert sl == 0, f"self-loops in {et}: {sl}"

if torch.cuda.is_available():
    data = data.to("cuda")            # NB: PyG .to() is in-place (returns self)
    log("moved to cuda OK; peak MiB:",
        round(torch.cuda.max_memory_allocated() / 1024**2, 1))
    data = data.to("cpu")             # move back so the saved artifact is CPU/portable
    torch.cuda.empty_cache()

torch.save(data, f"{KG}/heterodata.pt")
log(f"saved {KG}/heterodata.pt")
print("\nEDGE SUMMARY")
for et in data.edge_types:
    ea = "edge_attr" if "edge_attr" in data[et] else "-"
    print(f"  {str(et):45s} E={data[et].edge_index.shape[1]:>8d}  {ea}")
print("DONE")
