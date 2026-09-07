#!/usr/bin/env python3
"""Step 1.5 — Final drug node features = Morgan (2048) + LINCS PCA (200) = 2248."""
import os
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parents[1]
EMB = BASE / "embeddings"

morgan = np.load(os.path.join(EMB, "drug_morgan.npy")).astype(np.float32)
lincs = np.load(os.path.join(EMB, "drug_lincs_pca.npy")).astype(np.float32)
assert morgan.shape[0] == lincs.shape[0], (morgan.shape, lincs.shape)

X = np.concatenate([morgan, lincs], axis=1).astype(np.float32)
np.save(os.path.join(EMB, "drug_features.npy"), X)
print(f"drug_features.npy shape={X.shape} dtype={X.dtype} "
      f"nan={bool(np.isnan(X).any())}")
