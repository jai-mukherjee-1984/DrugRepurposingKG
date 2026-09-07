#!/usr/bin/env python3
"""Step 1.3e - Parse & standardize ORPHANET XML products.

Parses (when present):
  - en_product6.xml  -> orphanet_disease_gene.csv  (disease -> gene associations)
  - en_product4.xml  -> orphanet_hpo.csv           (disease -> HPO phenotype terms)
  - en_product1.xml  -> orphanet_xref.csv          (disease -> external cross-references)
                        orphanet_diseases.csv      (master disease list)

Uses lxml.iterparse with element clearing for low memory use on large files.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
from lxml import etree

RAW_DIR = Path("data/raw/orphanet")
OUT_DIR = Path("data/processed")


def _text(el, path):
    found = el.find(path)
    return found.text.strip() if found is not None and found.text else None


def iter_disorders(xml_path: Path, tag: str = "Disorder"):
    """Yield each <Disorder> element, clearing processed elements to bound memory."""
    context = etree.iterparse(str(xml_path), events=("end",), tag=tag)
    for _, elem in context:
        yield elem
        elem.clear()
        # Also drop preceding siblings to keep the tree small.
        while elem.getprevious() is not None:
            del elem.getparent()[0]
    del context


def parse_genes(xml_path: Path) -> pd.DataFrame:
    rows = []
    for dis in iter_disorders(xml_path):
        orpha = _text(dis, "OrphaCode")
        name = _text(dis, "Name")
        dtype = _text(dis, "DisorderType/Name")
        for assoc in dis.findall("DisorderGeneAssociationList/DisorderGeneAssociation"):
            gene = assoc.find("Gene")
            if gene is None:
                continue
            symbol = _text(gene, "Symbol")
            xref = {}
            for ext in gene.findall("ExternalReferenceList/ExternalReference"):
                src = _text(ext, "Source")
                ref = _text(ext, "Reference")
                if src:
                    xref[src] = ref
            rows.append({
                "orpha_code": orpha,
                "disease_name": name,
                "disorder_type": dtype,
                "gene_symbol": symbol,
                "hgnc_id": xref.get("HGNC"),
                "ensembl_gene": xref.get("Ensembl"),
                "swissprot_id": xref.get("SwissProt"),
                "omim_gene": xref.get("OMIM"),
                "association_type": _text(assoc, "DisorderGeneAssociationType/Name"),
            })
    return pd.DataFrame(rows)


def parse_hpo(xml_path: Path) -> pd.DataFrame:
    rows = []
    # In product4 each disorder is nested under HPODisorderSetStatus/Disorder.
    context = etree.iterparse(str(xml_path), events=("end",), tag="HPODisorderSetStatus")
    for _, status in context:
        dis = status.find("Disorder")
        if dis is not None:
            orpha = _text(dis, "OrphaCode")
            name = _text(dis, "Name")
            for assoc in dis.findall("HPODisorderAssociationList/HPODisorderAssociation"):
                rows.append({
                    "orpha_code": orpha,
                    "disease_name": name,
                    "hpo_id": _text(assoc, "HPO/HPOId"),
                    "hpo_term": _text(assoc, "HPO/HPOTerm"),
                    "frequency": _text(assoc, "HPOFrequency/Name"),
                })
        status.clear()
        while status.getprevious() is not None:
            del status.getparent()[0]
    del context
    return pd.DataFrame(rows)


def parse_xrefs(xml_path: Path):
    xref_rows = []
    disease_rows = []
    for dis in iter_disorders(xml_path):
        orpha = _text(dis, "OrphaCode")
        name = _text(dis, "Name")
        disease_rows.append({
            "orpha_code": orpha,
            "disease_name": name,
            "disorder_type": _text(dis, "DisorderType/Name"),
            "disorder_group": _text(dis, "DisorderGroup/Name"),
        })
        for ext in dis.findall("ExternalReferenceList/ExternalReference"):
            xref_rows.append({
                "orpha_code": orpha,
                "disease_name": name,
                "source": _text(ext, "Source"),
                "reference": _text(ext, "Reference"),
                "relation": _text(ext, "DisorderMappingRelation/Name"),
            })
    return pd.DataFrame(xref_rows), pd.DataFrame(disease_rows)


def main() -> int:
    start = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    p6 = RAW_DIR / "en_product6.xml"
    p4 = RAW_DIR / "en_product4.xml"
    p1 = RAW_DIR / "en_product1.xml"

    if p6.exists():
        genes = parse_genes(p6)
        genes.to_csv(OUT_DIR / "orphanet_disease_gene.csv", index=False)
        print(f"product6: {len(genes):,} disease-gene associations "
              f"({genes['orpha_code'].nunique():,} diseases, {genes['gene_symbol'].nunique():,} genes)")
    else:
        print("product6 missing - skipped disease-gene parsing")

    if p4.exists():
        hpo = parse_hpo(p4)
        hpo.to_csv(OUT_DIR / "orphanet_hpo.csv", index=False)
        print(f"product4: {len(hpo):,} disease-HPO associations "
              f"({hpo['orpha_code'].nunique():,} diseases, {hpo['hpo_id'].nunique():,} HPO terms)")
    else:
        print("product4 missing - skipped HPO parsing")

    if p1.exists():
        xrefs, diseases = parse_xrefs(p1)
        xrefs.to_csv(OUT_DIR / "orphanet_xref.csv", index=False)
        diseases.to_csv(OUT_DIR / "orphanet_diseases.csv", index=False)
        srcs = ", ".join(sorted(xrefs["source"].dropna().unique())[:12])
        print(f"product1: {len(diseases):,} diseases, {len(xrefs):,} xrefs. Sources: {srcs}")
    else:
        print("product1 missing - skipped disorder/xref parsing")

    print(f"Done in {time.time() - start:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
