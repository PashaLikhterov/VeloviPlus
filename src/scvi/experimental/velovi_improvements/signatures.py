from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import scanpy as sc

from .constants import HUMAN_CYCLE_G2M_GENES, HUMAN_CYCLE_S_GENES


@dataclass
class CycleSignature:
    species: str
    s_genes: list[str]
    g2m_genes: list[str]
    coverage: float


def _normalize_gene_name(gene: str, mode: str) -> str:
    if mode == "upper":
        return gene.upper()
    if mode == "lower":
        return gene.lower()
    if mode == "title":
        if len(gene) == 0:
            return gene
        return gene[0].upper() + gene[1:].lower()
    return gene


def _match_gene_list(
    var_names: Sequence[str],
    genes: Iterable[str],
) -> tuple[list[str], float]:
    var_array = np.asarray(var_names, dtype=str)
    lookup = {name: idx for idx, name in enumerate(var_array)}
    candidates = []
    hits = 0
    seen = set()
    gene_list = list(genes)
    for gene in gene_list:
        for mode in ("identity", "upper", "lower", "title"):
            normalized = _normalize_gene_name(gene, mode)
            if normalized in lookup and normalized not in seen:
                candidates.append(normalized)
                seen.add(normalized)
                hits += 1
                break
    coverage = hits / max(1, len(gene_list))
    return candidates, coverage


def resolve_cycle_signatures(var_names: Sequence[str]) -> CycleSignature | None:
    """Auto-detect species-aware S/G2M signatures available in the dataset."""
    if len(var_names) == 0:
        return None
    upper_mask = sum(str(name).isupper() for name in var_names[: min(2000, len(var_names))])
    title_mask = sum(
        str(name)[0].isupper() and str(name)[1:].islower() for name in var_names[: min(2000, len(var_names))]
    )
    species = "human" if upper_mask >= title_mask else "mouse"
    s_genes, s_cov = _match_gene_list(var_names, HUMAN_CYCLE_S_GENES)
    g2m_genes, g2m_cov = _match_gene_list(var_names, HUMAN_CYCLE_G2M_GENES)
    if not s_genes or not g2m_genes:
        return None
    coverage = float(min(s_cov, g2m_cov))
    return CycleSignature(species=species, s_genes=s_genes, g2m_genes=g2m_genes, coverage=coverage)


def score_cycle_signatures(
    adata,
    signature: CycleSignature,
    score_key: str,
):
    """Score cells using matched cycle signatures and store the residual in obs."""
    sc.tl.score_genes_cell_cycle(
        adata,
        s_genes=signature.s_genes,
        g2m_genes=signature.g2m_genes,
        copy=False,
    )
    cycle_score = adata.obs["S_score"].astype(np.float32) - adata.obs["G2M_score"].astype(np.float32)
    adata.obs[score_key] = cycle_score
    return cycle_score


def load_cycle_signature_file(path: str | Path) -> CycleSignature:
    """Load a custom cycle signature JSON file."""
    path_obj = Path(path)
    data = json.loads(path_obj.read_text())
    species = data.get("species", "custom")
    s_genes = data.get("s_genes") or data.get("S_genes") or data.get("S") or []
    g2m_genes = data.get("g2m_genes") or data.get("G2M_genes") or data.get("G2M") or []
    if not s_genes or not g2m_genes:
        raise ValueError(f"Cycle signature file {path} must contain S/G2M gene lists.")
    return CycleSignature(species=species, s_genes=list(s_genes), g2m_genes=list(g2m_genes), coverage=0.0)
