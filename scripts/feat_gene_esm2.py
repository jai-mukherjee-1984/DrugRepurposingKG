#!/usr/bin/env python3
"""Step 1.5 — Gene node features: ESM-2 3B embeddings (2560-dim).

Embeds each gene node's canonical protein sequence with ESM-2 3B
(esm2_t36_3B_UR50D) and stores the BOS/CLS-token representation from the final
layer. Sequences are truncated to MAX_LEN residues; genes without a sequence
get a zero row (logged separately in Step 1.5 FASTA step).

Dynamic token-budget batching (sequences sorted by length) keeps GPU memory
bounded on the GB10. Runs in fp16 for speed/memory.

Output: embeddings/gene_esm2_3b.npy  float32 [N_gene, 2560]
"""
import os
import numpy as np
import pandas as pd
import torch
import esm

BASE = os.environ.get("PROJECT", os.path.expanduser("~/sts-kg"))
KG = os.path.join(BASE, "data", "kg")
PROC = os.path.join(BASE, "data", "processed")
EMB = os.path.join(BASE, "embeddings")
os.makedirs(EMB, exist_ok=True)

MAX_LEN = 1022          # ESM-2 context (BOS +  1022 + EOS)
MAX_TOKENS = 4000       # per-batch token budget (padded length * batch)
REPR_LAYER = 36
EMB_DIM = 2560
EMPTY_CACHE_EVERY = 25  # batches; unified memory fragments badly without this
CKPT_EVERY = 100        # batches between resumable checkpoints

genes = pd.read_csv(os.path.join(KG, "nodes_gene.csv"))
entrez = genes["entrez_id"].astype(str).str.replace(r"\.0$", "", regex=True).tolist()
idx_of = {e: i for i, e in enumerate(entrez)}
N = len(entrez)

# parse FASTA
seqs = {}
cur, buf = None, []
with open(os.path.join(PROC, "gene_sequences.fasta")) as fh:
    for line in fh:
        line = line.rstrip()
        if line.startswith(">"):
            if cur is not None:
                seqs[cur] = "".join(buf)
            cur = line[1:].split("|")[0]
            buf = []
        else:
            buf.append(line)
    if cur is not None:
        seqs[cur] = "".join(buf)
print(f"gene nodes={N}  with sequence={len(seqs)}")

# items to embed: (row_idx, seq[:MAX_LEN]); sort by length for efficient batching
items = [(idx_of[e], s[:MAX_LEN]) for e, s in seqs.items() if e in idx_of and s] # maps each parsed ID to its row index
items.sort(key=lambda t: len(t[1])) # sorts by length of ID
print(f"embedding {len(items)} sequences")

device = "cuda"
model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
model = model.eval().half().to(device)
bc = alphabet.get_batch_converter()

X = np.zeros((N, EMB_DIM), dtype=np.float32)
filled = np.zeros(N, dtype=bool)

# Resume support: a killed run must not cost the whole embedding pass.
PART_X = os.path.join(EMB, "gene_esm2_3b.partial.npy")
PART_F = os.path.join(EMB, "gene_esm2_3b.filled.npy")
if os.path.exists(PART_X) and os.path.exists(PART_F):
    X = np.load(PART_X)
    filled = np.load(PART_F)
    items = [(r, s) for r, s in items if not filled[r]]
    print(f"resuming: {int(filled.sum())} rows already embedded, {len(items)} remaining", flush=True)


def checkpoint():
    np.save(PART_X, X)
    np.save(PART_F, filled)


done = 0
nbatch = 0
i = 0
# in a nutshell, this takes everything in batches and embeds them with the model ("device")
while i < len(items):
    # build a batch under the token budget (based on longest seq in batch)
    j = i
    maxlen = 0
    while j < len(items):
        maxlen = max(maxlen, len(items[j][1]) + 2)
        if (j - i + 1) * maxlen > MAX_TOKENS and j > i:
            break
        j += 1
    batch = items[i:j]
    data = [(str(r), s) for r, s in batch]
    _, _, toks = bc(data)
    toks = toks.to(device)
    with torch.no_grad():
        out = model(toks, repr_layers=[REPR_LAYER])
    reps = out["representations"][REPR_LAYER][:, 0, :].float().cpu().numpy()  # BOS token
    for k, (r, _s) in enumerate(batch):
        X[r] = reps[k]
        filled[r] = True
    del out, reps, toks
    done += len(batch)
    nbatch += 1
    i = j
    if nbatch % EMPTY_CACHE_EVERY == 0:
        torch.cuda.empty_cache()
    if nbatch % CKPT_EVERY == 0:
        checkpoint()
    if done % 2000 < len(batch):
        print(f"  {done}/{len(items)} embedded", flush=True)

np.save(os.path.join(EMB, "gene_esm2_3b.npy"), X)
for p in (PART_X, PART_F):
    if os.path.exists(p):
        os.remove(p)
zero_rows = int((~X.any(axis=1)).sum())
print(f"saved gene_esm2_3b.npy shape={X.shape} dtype={X.dtype} "
      f"nan={bool(np.isnan(X).any())} zero_rows={zero_rows}")
