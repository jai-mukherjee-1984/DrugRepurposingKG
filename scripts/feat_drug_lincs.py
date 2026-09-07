#!/usr/bin/env python3
"""Step 1.5 — Drug LINCS transcriptomic PCA features (200-dim).

Load the 978-dim LINCS L1000 landmark signatures, aggregate to one vector per
InChIKey (mean over pert_ids/cell contexts), fit PCA(200) on all profiled
compounds, then place each drug node's embedding in nodes_drug.csv order.
Drugs with no LINCS signature get the mean PCA vector (neutral placeholder) and
are flagged.

Outputs:
  embeddings/drug_lincs_pca.npy         float32 [N_drug, 200]
  embeddings/drug_lincs_missing.txt     inchikeys with no LINCS signature
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

BASE = Path(__file__).resolve().parents[1]
KG = BASE / "data" / "kg"
PROC = BASE / "data" / "processed"
EMB = BASE / "embeddings"
os.makedirs(EMB, exist_ok=True)
NCOMP = 200

drugs = pd.read_csv(os.path.join(KG, "nodes_drug.csv"))
inchikeys = drugs["inchikey"].astype(str).tolist()

sig = pd.read_parquet(os.path.join(PROC, "lincs_signatures.parquet"))
lm_cols = [c for c in sig.columns if c not in ("inchikey", "pert_id")]
print(f"LINCS rows={len(sig)}  signature dims={len(lm_cols)}")

# aggregate to one 978-dim vector per InChIKey
agg = sig.groupby("inchikey")[lm_cols].mean()
agg = agg.astype(np.float32)
print(f"unique profiled InChIKeys={len(agg)}")

pca = PCA(n_components=NCOMP, random_state=42)
P = pca.fit_transform(agg.values).astype(np.float32)   # [U, 200]
print(f"PCA explained variance (200 comps): {pca.explained_variance_ratio_.sum():.3f}")

ik2row = {ik: P[i] for i, ik in enumerate(agg.index.astype(str))}
mean_vec = P.mean(axis=0).astype(np.float32)

X = np.zeros((len(inchikeys), NCOMP), dtype=np.float32)
missing = []
for i, ik in enumerate(inchikeys):
    v = ik2row.get(ik)
    if v is None:
        X[i] = mean_vec
        missing.append(ik)
    else:
        X[i] = v

np.save(os.path.join(EMB, "drug_lincs_pca.npy"), X)
with open(os.path.join(EMB, "drug_lincs_missing.txt"), "w") as fh:
    fh.write("\n".join(missing))
cover = 100 * (len(inchikeys) - len(missing)) / len(inchikeys)
print(f"saved drug_lincs_pca.npy shape={X.shape}")
print(f"LINCS coverage: {len(inchikeys)-len(missing)}/{len(inchikeys)} ({cover:.1f}%)")
print(f"missing (mean-filled): {len(missing)}")
