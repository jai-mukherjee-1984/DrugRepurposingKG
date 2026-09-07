# DrugRepurposingKG

A multi-modal biomedical knowledge graph for drug repurposing and mechanistic interpretation.

This project builds a heterogeneous graph over drugs, genes, diseases, and pathways, integrates multiple biomedical sources, generates node features, and supports downstream graph learning and explainability workflows for repurposing candidate discovery.

## Overview

DrugRepurposingKG combines curated biomedical knowledge from sources such as:

- RepoDB / Repurposing Hub
- ORPHANET
- OpenTargets
- Hetionet
- STRING
- ChEMBL
- LINCS

The pipeline standardizes identifiers, aligns entities to canonical IDs, builds node feature matrices, assembles a heterogeneous PyTorch Geometric graph, and evaluates drug-disease or drug-target hypotheses through graph-based learning.

## Repository structure

```text
DrugRepurposingKG/
├── data/
│   ├── raw/
│   ├── processed/
│   └── kg/
├── embeddings/
├── models/
├── results/
├── scripts/
├── README.md
├── LICENSE
└── .gitignore
```

## Key capabilities

- Multi-entity biomedical KG construction
- Canonical ID alignment across drugs, genes, diseases, and pathways
- Feature generation for molecular, transcriptomic, protein, and disease signals
- Heterogeneous graph assembly using PyTorch Geometric
- Structural integrity checks and graph diagnostics
- Baseline and downstream evaluation workflows for repurposing tasks

## Data flow

1. Download source datasets
2. Parse and normalize source-specific records
3. Align entities to canonical identifiers
4. Build node lists and feature matrices
5. Assemble the heterogeneous graph
6. Run integrity audits and training/evaluation pipelines

## Getting started

### Prerequisites

- Python 3.10+
- PyTorch and PyTorch Geometric
- NumPy, pandas, scikit-learn
- Optional: networkx, rdkit, transformers, sentence-transformers

### Installation

```bash
git clone https://github.com/<your-org>/DrugRepurposingKG.git
cd DrugRepurposingKG
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# or .venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

If a requirements file is not present yet, install the key dependencies manually and pin versions as needed for your environment.

## Usage

The `scripts/` directory contains the project pipeline. Typical workflow:

```bash
python scripts/download_xref_refs.py
python scripts/parse_repodb.py
python scripts/align_reindex.py
python scripts/build_node_lists.py
python scripts/feat_drug_morgan.py
python scripts/build_heterodata.py
python scripts/kg_integrity.py
```

## Outputs

- Processed and aligned source tables under `data/processed/`
- Canonical node lists under `data/kg/`
- Generated embeddings under `embeddings/`
- Final graph objects under `data/kg/`
- Evaluation and diagnostic artifacts under `results/`

## Project status

This project is designed to support biomedical knowledge graph construction and graph-based drug repurposing research. The repository is intended for reproducible experimentation, benchmarking, and downstream interpretation work.

## License

This project is distributed under the license specified in the repository root. See `LICENSE` for details.

## Citation and provenance

Please cite this repository and related source datasets when using the graph or derived results in publication or benchmarking workflows.

## Contact

For inquiries or collaboration opportunities, contact the project maintainer or use the repository issue tracker.
