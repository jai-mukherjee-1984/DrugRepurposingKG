#!/usr/bin/env python3
"""Step 1.5 — Validate node feature arrays against node lists.

Checks every embedding file exists, has the expected row count (matching its
node list) and width, contains no NaN/Inf, and reports coverage stats.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
KG = BASE / "data" / "kg"
EMB = BASE / "embeddings"


def nrows(csv):
    return len(pd.read_csv(os.path.join(KG, csv)))


N_drug = nrows("nodes_drug.csv")
N_gene = nrows("nodes_gene.csv")
N_dis = nrows("nodes_disease.csv")
N_pw = nrows("nodes_pathway.csv")

EXPECT = {
    "drug_morgan.npy": (N_drug, 2048),
    "drug_lincs_pca.npy": (N_drug, 200),
    "drug_features.npy": (N_drug, 2248),
    "gene_esm2_3b.npy": (N_gene, 2560),
    "disease_sbert.npy": (N_dis, 384),
    "pathway_onehot.npy": (N_pw, 8),
}

print(f"node counts: drug={N_drug} gene={N_gene} disease={N_dis} pathway={N_pw}\n")
ok = True
for fn, (r, c) in EXPECT.items():
    fp = os.path.join(EMB, fn)
    if not os.path.exists(fp):
        print(f"MISSING  {fn}")
        ok = False
        continue
    X = np.load(fp)
    shape_ok = X.shape == (r, c)
    nan = bool(np.isnan(X).any()) if X.dtype.kind == "f" else False
    inf = bool(np.isinf(X).any()) if X.dtype.kind == "f" else False
    zero_rows = int((~X.any(axis=1)).sum())
    # genuinely just dtm, can just be 1 line
    flag = "OK " if (shape_ok and not nan and not inf) else "BAD"
    if flag == "BAD":
        ok = False
    print(f"{flag} {fn:22s} shape={str(X.shape):16s} "
          f"expect=({r},{c}) dtype={X.dtype} nan={nan} inf={inf} "
          f"zero_rows={zero_rows}")

# missing-flag files
for fn in ("drug_morgan_missing.txt", "drug_lincs_missing.txt", "gene_seq_missing.txt"):
    fp = os.path.join(EMB, fn)
    if not os.path.exists(fp):
        fp = os.path.join(BASE, "data", "processed", fn)
    n = sum(1 for _ in open(fp)) if os.path.exists(fp) else "MISSING"
    print(f"    flag {fn:26s} count={n}")

total = sum(os.path.getsize(os.path.join(EMB, f)) for f in os.listdir(EMB)
            if f.endswith(".npy"))
print(f"\nembeddings/*.npy total size: {total/1e9:.2f} GB")
print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
