#!/usr/bin/env python3
"""Step 1.3a - Parse & standardize HETIONET v1.0.

Reads:
  - hetnet/tsv/hetionet-v1.0-nodes.tsv       (id, name, kind)
  - hetnet/tsv/hetionet-v1.0-edges.sif.gz    (source, metaedge, target)

Writes:
  - hetionet_nodes.csv   : node_id | kind | name | source_id
  - hetionet_edges.csv   : source_id | source_type | metaedge | target_id | target_type

Node ids are of the form "Kind::LocalID" (e.g. Compound::DB00001, Gene::9021,
Disease::DOID:1234). LocalID is extracted into source_id for later alignment.
"""

from __future__ import annotations

import gzip
import sys
import time
from pathlib import Path

import pandas as pd

RAW = Path("data/raw/hetionet/hetnet/tsv")
OUT_DIR = Path("data/processed")


def split_id(full: str):
    kind, _, local = full.partition("::")
    return kind, local


def parse_nodes(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str)
    parts = df["id"].apply(split_id)
    df["source_id"] = [p[1] for p in parts]
    df = df.rename(columns={"id": "node_id"})
    return df[["node_id", "kind", "name", "source_id"]]


def parse_edges(path: Path) -> pd.DataFrame:
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        header = next(fh)  # source, metaedge, target
        for line in fh:
            src, metaedge, tgt = line.rstrip("\n").split("\t")
            s_kind, s_id = split_id(src)
            t_kind, t_id = split_id(tgt)
            rows.append((s_id, s_kind, metaedge, t_id, t_kind))
    return pd.DataFrame(rows, columns=["source_id", "source_type", "metaedge", "target_id", "target_type"])


def main() -> int:
    start = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    nodes_path = RAW / "hetionet-v1.0-nodes.tsv"
    edges_path = RAW / "hetionet-v1.0-edges.sif.gz"

    nodes = parse_nodes(nodes_path)
    nodes.to_csv(OUT_DIR / "hetionet_nodes.csv", index=False)
    print(f"nodes: {len(nodes):,} across {nodes['kind'].nunique()} kinds")
    print(nodes["kind"].value_counts().to_string())

    edges = parse_edges(edges_path)
    edges.to_csv(OUT_DIR / "hetionet_edges.csv", index=False)
    print(f"edges: {len(edges):,} across {edges['metaedge'].nunique()} metaedges")

    print(f"Done in {time.time() - start:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
