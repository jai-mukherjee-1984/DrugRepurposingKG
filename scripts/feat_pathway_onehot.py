#!/usr/bin/env python3
"""Step 1.5 — Pathway node features: one-hot over 8 categories.

Categorizes each HETIONET pathway node's name into one of 8 biological process
categories by keyword matching, then one-hot encodes. Unmatched -> 'other'.

Category order (column index):
  0 metabolic  1 signaling  2 immune  3 apoptosis
  4 cell_cycle 5 dna_repair 6 transport 7 other

Output: embeddings/pathway_onehot.npy  float32 [N_pathway, 8]
"""
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
KG = BASE / "data" / "kg"
EMB = BASE / "embeddings"
os.makedirs(EMB, exist_ok=True)

CATS = ["metabolic", "signaling", "immune", "apoptosis",
        "cell_cycle", "dna_repair", "transport", "other"]
IDX = {c: i for i, c in enumerate(CATS)}

# ordered keyword rules -> first match wins
RULES = [
    ("apoptosis", r"apopto|cell death|caspase|necropto|pyropto"),
    ("dna_repair", r"dna repair|mismatch repair|nucleotide excision|base excision|"
                   r"double-strand break|homologous recomb|damage response"),
    ("cell_cycle", r"cell cycle|mitotic|mitosis|meiosis|cyclin|checkpoint|"
                   r"chromosome segregation|spindle"),
    ("immune", r"immun|inflamm|interferon|interleukin|cytokine|chemokine|"
               r"t cell|b cell|complement|toll-like|antigen|nf-kappa|nf-kb|"
               r"tnf|innate|adaptive|mhc|leukocyte|macrophage"),
    ("transport", r"transport|import|export|secretion|channel|symport|antiport|"
                  r"vesicle|trafficking|endocyto|exocyto|uptake"),
    ("metabolic", r"metabol|biosynth|catabol|glycolysis|gluconeogen|"
                  r"citric acid|tca cycle|oxidative phosphor|fatty acid|"
                  r"lipid|amino acid|nucleotide metabol|krebs|respiration|"
                  r"cholesterol|glucose|pentose|urea cycle|degradation of"),
    ("signaling", r"signal|pathway|receptor|kinase|cascade|gpcr|wnt|notch|"
                  r"mapk|pi3k|akt|mtor|hedgehog|jak|stat|erk|ras|tgf|"
                  r"phosphoryl|second messenger|calcium signal"),
]
COMPILED = [(cat, re.compile(pat)) for cat, pat in RULES]

pw = pd.read_csv(os.path.join(KG, "nodes_pathway.csv"))
names = pw["name"].fillna("").astype(str).str.lower().tolist()

X = np.zeros((len(names), 8), dtype=np.float32)
counts = {c: 0 for c in CATS}
for i, nm in enumerate(names):
    cat = "other"
    for c, rx in COMPILED:
        if rx.search(nm):
            cat = c
            break
    X[i, IDX[cat]] = 1.0
    counts[cat] += 1

np.save(os.path.join(EMB, "pathway_onehot.npy"), X)
print(f"saved pathway_onehot.npy shape={X.shape} dtype={X.dtype}")
for c in CATS:
    print(f"  {c:11s} {counts[c]:5d}")
