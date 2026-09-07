#!/usr/bin/env python3
"""Step 1.5 — Disease node features: Sentence-BERT embeddings (384-dim).

Embeds each disease node's name with sentence-transformers all-MiniLM-L6-v2
(native 384-dim). Runs on CPU to avoid contending with the ESM-2 GPU job.
Diseases with no name fall back to their MONDO id string.

Output: embeddings/disease_sbert.npy  float32 [N_disease, 384]
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

BASE = Path(__file__).resolve().parents[1]
KG = BASE / "data" / "kg"
EMB = BASE / "embeddings"
os.makedirs(EMB, exist_ok=True)

dis = pd.read_csv(os.path.join(KG, "nodes_disease.csv"))
names = dis["name"].fillna("").astype(str).tolist()
mondo = dis["mondo_id"].astype(str).tolist()
texts = [n if n.strip() else m for n, m in zip(names, mondo)]
n_named = sum(1 for n in names if n.strip())
print(f"disease nodes={len(texts)}  with name={n_named} "
      f"({100*n_named/len(texts):.1f}%)")

_local = os.path.join(BASE, "models_cache", "all-MiniLM-L6-v2")
model_src = _local if os.path.isdir(_local) else "all-MiniLM-L6-v2"
print(f"loading SBERT from: {model_src}")
model = SentenceTransformer(model_src, device="cpu")
X = model.encode(texts, batch_size=256, show_progress_bar=True,
                 convert_to_numpy=True).astype(np.float32)

np.save(os.path.join(EMB, "disease_sbert.npy"), X)
print(f"saved disease_sbert.npy shape={X.shape} dtype={X.dtype} "
      f"nan={bool(np.isnan(X).any())}")
# so apparently this code encodes disease names/mondo IDs into a 384 dimension vector. idk WHAT the features are, but whatever