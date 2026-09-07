#!/usr/bin/env python3
"""Step 1.5 — Fetch canonical protein sequences (FASTA) for gene nodes.

Maps each gene node's Entrez ID -> UniProt (Swiss-Prot first, then TrEMBL for
leftovers) via the UniProt ID-mapping REST API and writes one canonical
sequence per gene. Genes with no UniProt sequence are logged and get a zero
ESM-2 embedding downstream.

Outputs:
  data/processed/gene_sequences.fasta   headers ">entrez|accession"
  data/processed/gene_seq_missing.txt   entrez ids with no sequence
"""
import os
import time
import json
from pathlib import Path

import requests
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
KG = BASE / "data" / "kg"
PROC = BASE / "data" / "processed"
API = "https://rest.uniprot.org"
POLL = 3

genes = pd.read_csv(os.path.join(KG, "nodes_gene.csv"))
entrez = genes["entrez_id"].astype(str).str.replace(r"\.0$", "", regex=True).tolist()
print(f"gene nodes: {len(entrez)}")


def run_idmapping(ids, to_db):
    """Return dict entrez -> best UniProt entry (dict) for a target DB."""
    got = {}
    r = requests.post(f"{API}/idmapping/run",
                      data={"from": "GeneID", "to": to_db, "ids": ",".join(ids)})
    r.raise_for_status()
    job = r.json()["jobId"]
    # poll
    while True:
        s = requests.get(f"{API}/idmapping/status/{job}")
        s.raise_for_status()
        js = s.json()
        st = js.get("jobStatus")
        if st in ("RUNNING", "NEW"):
            time.sleep(POLL)
            continue
        break
    # stream all results as JSON (paged via link headers)
    url = f"{API}/idmapping/uniprotkb/results/stream/{job}?format=json"
    resp = requests.get(url)
    resp.raise_for_status()
    data = resp.json()
    for rec in data.get("results", []):
        frm = str(rec["from"])
        entry = rec["to"]
        if not isinstance(entry, dict):
            continue
        seq = entry.get("sequence", {}).get("value")
        if not seq:
            continue
        reviewed = entry.get("entryType", "").lower().find("reviewed") >= 0 or \
            entry.get("entryType", "").find("Swiss-Prot") >= 0
        prev = got.get(frm)
        cand = (reviewed, len(seq), entry.get("primaryAccession", ""), seq)
        if prev is None or cand[:2] > prev[:2]:
            got[frm] = cand
    return {k: (v[2], v[3]) for k, v in got.items()}   # entrez -> (acc, seq)

# take in batches of 50k to avoid too much memory
def batched(seq, n=50000):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


mapping = {}
# Pass 1: reviewed Swiss-Prot (one high-quality canonical seq per gene)
for chunk in batched(entrez):
    mapping.update(run_idmapping(chunk, "UniProtKB-Swiss-Prot"))
print(f"after Swiss-Prot: {len(mapping)} mapped")

# Pass 2: TrEMBL fallback for still-unmapped genes
left = [e for e in entrez if e not in mapping]
if left:
    for chunk in batched(left):
        mapping.update(run_idmapping(chunk, "UniProtKB"))
print(f"after TrEMBL fallback: {len(mapping)} mapped")

missing = []
with open(os.path.join(PROC, "gene_sequences.fasta"), "w") as fh:
    for e in entrez:
        if e in mapping:
            acc, seq = mapping[e]
            fh.write(f">{e}|{acc}\n{seq}\n")
        else:
            missing.append(e)
with open(os.path.join(PROC, "gene_seq_missing.txt"), "w") as fh:
    fh.write("\n".join(missing))

cover = 100 * (len(entrez) - len(missing)) / len(entrez)
print(f"wrote gene_sequences.fasta  mapped={len(entrez)-len(missing)}/{len(entrez)} "
      f"({cover:.1f}%)  missing={len(missing)}")
