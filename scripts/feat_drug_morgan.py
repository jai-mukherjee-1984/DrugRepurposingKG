#!/usr/bin/env python3
"""Step 1.5 — Drug Morgan fingerprints (2048-bit, radius=2).

For each drug node (InChIKey order from data/kg/nodes_drug.csv), look up a
canonical SMILES in the ChEMBL SQLite DB (compound_structures.standard_inchi_key)
and compute a 2048-bit Morgan fingerprint (RDKit). Drugs with no SMILES or an
unparseable structure get an all-zero row and are logged.

Outputs:
  embeddings/drug_morgan.npy            uint8 [N_drug, 2048]
  embeddings/drug_morgan_missing.txt    inchikeys with no/failed structure
"""
import os
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit import DataStructs

RDLogger.DisableLog("rdApp.*")

BASE = Path(__file__).resolve().parents[1]
KG = BASE / "data" / "kg"
EMB = BASE / "embeddings"
DB = BASE / "data" / "raw" / "chembl" / "chembl_34" / "chembl_34_sqlite" / "chembl_34.db"
os.makedirs(EMB, exist_ok=True)
NBITS = 2048

drugs = pd.read_csv(os.path.join(KG, "nodes_drug.csv"))
inchikeys = drugs["inchikey"].astype(str).tolist()
want = set(inchikeys)
print(f"drug nodes: {len(inchikeys)}")

# Pull SMILES for the InChIKeys we need from ChEMBL.
con = sqlite3.connect(DB)
cur = con.cursor()
ik2smiles = {}
for ik, smi in cur.execute(
        "SELECT standard_inchi_key, canonical_smiles FROM compound_structures"):
    if ik in want and smi:
        ik2smiles[ik] = smi
con.close()
print(f"InChIKeys with a ChEMBL SMILES: {len(ik2smiles)} "
      f"({100*len(ik2smiles)/len(inchikeys):.1f}%)")

gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=NBITS)
X = np.zeros((len(inchikeys), NBITS), dtype=np.uint8)
missing = []
row = np.zeros((NBITS,), dtype=np.uint8)
for i, ik in enumerate(inchikeys):
    smi = ik2smiles.get(ik)
    if not smi:
        missing.append(ik)
        continue
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        missing.append(ik)
        continue
    fp = gen.GetFingerprint(mol)
    DataStructs.ConvertToNumpyArray(fp, row)
    X[i] = row

np.save(os.path.join(EMB, "drug_morgan.npy"), X)
with open(os.path.join(EMB, "drug_morgan_missing.txt"), "w") as fh:
    fh.write("\n".join(missing))
print(f"saved drug_morgan.npy shape={X.shape} dtype={X.dtype}")
print(f"missing/failed structures: {len(missing)} "
      f"({100*len(missing)/len(inchikeys):.1f}%)")
