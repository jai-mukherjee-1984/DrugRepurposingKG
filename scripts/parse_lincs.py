#!/usr/bin/env python3
"""Step 1.3b - Parse & standardize LINCS L1000 (Level 5, GSE92742).

Reads the MODZ consensus signature matrix (473647 sigs x 12328 genes) plus the
gene/sig/pert metadata and writes:
  - lincs_signatures.parquet : inchikey | pert_id | lm_<geneid> x978

Steps:
  * keep only the 978 landmark genes (pr_is_lm == 1)
  * keep only compound signatures (pert_type == 'trt_cp') with a valid InChIKey
  * median-aggregate all signatures of a compound into one 978-dim vector

The matrix is read in contiguous row slabs to bound peak memory.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

RAW = Path("data/raw/lincs")
OUT_DIR = Path("data/processed")
GCTX = RAW / "GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx"
GENE_INFO = RAW / "GSE92742_Broad_LINCS_gene_info.txt.gz"
SIG_INFO = RAW / "GSE92742_Broad_LINCS_sig_info.txt.gz"
PERT_INFO = RAW / "GSE92742_Broad_LINCS_pert_info.txt.gz"
BLOCK = 50_000
MISSING = {"-666", "-666.0", "", "nan"}


def main() -> int:
    start = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- metadata ---
    gene_info = pd.read_csv(GENE_INFO, sep="\t", dtype=str)
    landmark_genes = set(gene_info.loc[gene_info["pr_is_lm"] == "1", "pr_gene_id"])
    print(f"landmark genes: {len(landmark_genes)}")

    sig_info = pd.read_csv(SIG_INFO, sep="\t", dtype=str, usecols=["sig_id", "pert_id", "pert_type"])
    sig_info = sig_info[sig_info["pert_type"] == "trt_cp"]
    sig_to_pert = dict(zip(sig_info["sig_id"], sig_info["pert_id"]))

    pert_info = pd.read_csv(PERT_INFO, sep="\t", dtype=str, usecols=["pert_id", "inchi_key"])
    pert_to_ik = {
        r.pert_id: r.inchi_key
        for r in pert_info.itertuples(index=False)
        if r.inchi_key and str(r.inchi_key) not in MISSING
    }
    print(f"trt_cp sigs: {len(sig_to_pert):,}; perts with InChIKey: {len(pert_to_ik):,}")

    with h5py.File(GCTX, "r") as f:
        row_ids = [x.decode() for x in f["0/META/ROW/id"][:]]        # gene ids (12328)
        col_ids = [x.decode() for x in f["0/META/COL/id"][:]]        # sig ids (473647)
        matrix = f["0/DATA/0/matrix"]                                 # (nsig, ngene)
        nsig, ngene = matrix.shape

        # landmark gene column indices (into the 12328 axis), preserving gene order
        gene_pos = [(i, g) for i, g in enumerate(row_ids) if g in landmark_genes]
        gene_idx = np.array([i for i, _ in gene_pos])
        gene_cols = [f"lm_{g}" for _, g in gene_pos]

        # which signatures to keep, and their compound InChIKey
        keep_pos, keep_ik, keep_pert = [], [], []
        for j, sid in enumerate(col_ids):
            pid = sig_to_pert.get(sid)
            if pid is None:
                continue
            ik = pert_to_ik.get(pid)
            if ik is None:
                continue
            keep_pos.append(j)
            keep_ik.append(ik)
            keep_pert.append(pid)
        keep_pos_arr = np.array(keep_pos)
        print(f"signatures kept (trt_cp + InChIKey): {len(keep_pos):,}")

        # read landmark slice in contiguous row slabs
        keep_set = set(keep_pos)
        collected = np.empty((len(keep_pos), len(gene_idx)), dtype=np.float32)
        pos_to_out = {p: k for k, p in enumerate(keep_pos)}
        filled = 0
        for s in range(0, nsig, BLOCK):
            e = min(s + BLOCK, nsig)
            block_keep = [p for p in range(s, e) if p in keep_set]
            if not block_keep:
                continue
            slab = matrix[s:e, :]                       # (e-s, 12328) contiguous read
            sub = slab[[p - s for p in block_keep]][:, gene_idx]
            for row_i, p in enumerate(block_keep):
                collected[pos_to_out[p]] = sub[row_i]
            filled += len(block_keep)
        print(f"filled {filled:,} signature rows")

    df = pd.DataFrame(collected, columns=gene_cols)
    df.insert(0, "pert_id", keep_pert)
    df.insert(0, "inchikey", keep_ik)

    # median-aggregate multiple signatures per compound
    agg = df.groupby("inchikey", as_index=False)[gene_cols].median()
    # keep one representative pert_id per inchikey
    pert_map = df.drop_duplicates("inchikey").set_index("inchikey")["pert_id"]
    agg.insert(1, "pert_id", agg["inchikey"].map(pert_map))

    out = OUT_DIR / "lincs_signatures.parquet"
    agg.to_parquet(out, index=False)
    print(f"compounds: {len(agg):,} x {len(gene_cols)} landmark dims")
    print(f"Done in {time.time() - start:.1f}s -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
