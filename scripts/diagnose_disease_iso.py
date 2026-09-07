#!/usr/bin/env python
"""Diagnostic: why are ~4,305 disease nodes isolated, and which available edge
sources could connect them? Read-only; writes nothing."""
import pandas as pd, numpy as np, torch, os

KG, PROC, RAW = "data/kg", "data/processed", "data/raw/xref"

d = torch.load(f"{KG}/heterodata.pt", weights_only=False)
nd = pd.read_csv(f"{KG}/nodes_disease.csv")
ng = pd.read_csv(f"{KG}/nodes_gene.csv")
mondo2idx = {m: i for i, m in zip(nd.idx, nd.mondo_id.astype(str))}
dis_set = set(nd.mondo_id.astype(str))
gene_set = set(pd.to_numeric(ng.entrez_id, errors="coerce").dropna().astype(int))

# current disease degree
deg = torch.zeros(d["disease"].num_nodes, dtype=torch.long)
for et in d.edge_types:
    if et[0] == "disease":
        deg.scatter_add_(0, d[et].edge_index[0], torch.ones(d[et].edge_index.shape[1], dtype=torch.long))
    if et[2] == "disease":
        deg.scatter_add_(0, d[et].edge_index[1], torch.ones(d[et].edge_index.shape[1], dtype=torch.long))
iso_mask = (deg == 0).numpy()
iso_idx = set(np.where(iso_mask)[0].tolist())
iso_mondo = set(nd.mondo_id.astype(str)[iso_mask])
print(f"disease nodes: {len(nd)}  isolated: {len(iso_mondo)} ({100*len(iso_mondo)/len(nd):.1f}%)")
print(f"need to connect >= {len(iso_mondo) - int(0.20*len(nd))} isolated to reach <=20%\n")

he = pd.read_csv(f"{PROC}/hetionet_edges_aligned.csv", low_memory=False)


def dis_gene_connectable(meta):
    """HETIONET Disease->Gene metaedge: how many isolated diseases gain an edge
    (disease mapped + gene in gene_set)."""
    sub = he[he.metaedge == meta]
    dm = sub.source_canonical.astype(str)
    ge = pd.to_numeric(sub.target_canonical, errors="coerce")
    ok = dm.isin(iso_mondo) & ge.isin(gene_set)
    return set(dm[ok])


for meta in ["DuG", "DdG", "DaG"]:
    c = dis_gene_connectable(meta)
    print(f"HETIONET {meta}: could connect {len(c)} isolated diseases")

# DrD disease-resembles-disease (both endpoints in dis_set; at least one isolated)
drd = he[he.metaedge == "DrD"]
s = drd.source_canonical.astype(str); t = drd.target_canonical.astype(str)
drd_ok = (s.isin(dis_set) & t.isin(dis_set)) & (s.isin(iso_mondo) | t.isin(iso_mondo))
drd_conn = set(s[drd_ok & s.isin(iso_mondo)]) | set(t[drd_ok & t.isin(iso_mondo)])
print(f"HETIONET DrD: {len(drd)} edges, could connect {len(drd_conn)} isolated diseases")

# HPO: isolated diseases that have >=1 HPO term (via orpha->mondo)
hpo = pd.read_csv(f"{PROC}/orphanet_hpo.csv", low_memory=False)
odis = pd.read_csv(f"{PROC}/orphanet_diseases_aligned.csv", low_memory=False)
orpha2mondo = {int(o): m for o, m in zip(odis.orpha_code, odis.mondo_id.astype(str)) if pd.notna(m)}
hpo_mondo = set(hpo.orpha_code.map(lambda o: orpha2mondo.get(int(o)) if pd.notna(o) else None).dropna())
hpo_iso = iso_mondo & hpo_mondo
print(f"HPO: {len(hpo_iso)} isolated diseases HAVE HPO terms "
      f"(pure top-k could connect most of these)")

# MONDO ontology hierarchy (is_a) from mondo.obo
obo = os.path.join(RAW, "mondo.obo")
parent = {}
if os.path.exists(obo):
    cur = None
    for ln in open(obo, encoding="utf-8"):
        ln = ln.strip()
        if ln == "[Term]":
            cur = None
        elif ln.startswith("id: MONDO:"):
            cur = ln[4:]
        elif ln.startswith("is_a: MONDO:") and cur:
            par = ln.split("is_a:")[1].split("!")[0].strip()
            parent.setdefault(cur, []).append(par)
    # edges among node-set diseases where at least one endpoint isolated
    conn = set()
    n_edges = 0
    for c, pars in parent.items():
        if c not in dis_set:
            continue
        for p in pars:
            if p in dis_set:
                n_edges += 1
                if c in iso_mondo:
                    conn.add(c)
                if p in iso_mondo:
                    conn.add(p)
    print(f"MONDO is_a: {n_edges} parent-child edges within disease nodes; "
          f"could connect {len(conn)} isolated diseases")
else:
    print("mondo.obo not found at", obo)

# union of best additive fixes (DuG+DdG+DrD+HPO)
best = set()
for meta in ["DuG", "DdG"]:
    best |= dis_gene_connectable(meta)
best |= drd_conn
best |= hpo_iso
print(f"\nUNION DuG+DdG+DrD+HPO could connect ~{len(best)} isolated diseases "
      f"-> remaining isolated ~{len(iso_mondo)-len(best)} "
      f"({100*(len(iso_mondo)-len(best))/len(nd):.1f}%)")
