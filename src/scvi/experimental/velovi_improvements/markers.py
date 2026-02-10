from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Sequence

import numpy as np


@dataclass
class MarkerSet:
    name: str
    direction: Literal["increase", "decrease"]
    genes: list[str]
    branches: list[int]
    weight: float


@dataclass
class MarkerConfigResult:
    sets: list[MarkerSet]
    matrix: np.ndarray  # shape (n_genes, n_sets)
    gene_counts: np.ndarray  # shape (n_sets,)
    branch_map: np.ndarray  # shape (n_sets, n_branches)
    direction: np.ndarray  # +1 increase, -1 decrease


def _normalize_gene(name: str) -> str:
    if not name:
        return name
    if name.isupper() or name.islower():
        return name
    return name.upper()


def load_marker_config(
    path: str | Path,
    var_names: Sequence[str],
    n_branches: int,
) -> MarkerConfigResult:
    """Load a marker configuration JSON file and map to expression indices."""
    path_obj = Path(path)
    data = json.loads(path_obj.read_text())
    raw_sets = data.get("marker_sets") or data.get("markers")
    if not raw_sets:
        raise ValueError(f"Marker configuration {path} must contain a 'marker_sets' list.")
    markers: list[MarkerSet] = []
    for item in raw_sets:
        name = item.get("name")
        genes = item.get("genes")
        direction = item.get("direction", "increase")
        branches = item.get("branches", [])
        weight = float(item.get("weight", 1.0))
        if not name or not genes:
            continue
        direction_l = "increase" if direction not in {"increase", "decrease"} else direction
        normalized_genes = [_normalize_gene(g) for g in genes]
        branch_ids = [int(b) for b in branches] if branches else list(range(max(1, n_branches)))
        markers.append(MarkerSet(name=name, direction=direction_l, genes=normalized_genes, branches=branch_ids, weight=weight))
    if not markers:
        raise ValueError(f"Marker configuration {path} did not yield any valid sets.")
    var_lookup = { _normalize_gene(str(g)): idx for idx, g in enumerate(var_names) }
    matrix = np.zeros((len(var_names), len(markers)), dtype=np.float32)
    counts = np.ones(len(markers), dtype=np.float32)
    direction = np.ones(len(markers), dtype=np.float32)
    branch_map = np.zeros((len(markers), max(1, n_branches)), dtype=np.float32)
    for col, marker_set in enumerate(markers):
        hit_genes = 0
        for gene in marker_set.genes:
            idx = var_lookup.get(gene) or var_lookup.get(gene.upper()) or var_lookup.get(gene.lower())
            if idx is None:
                continue
            matrix[idx, col] = 1.0
            hit_genes += 1
        counts[col] = max(1, hit_genes)
        if marker_set.direction == "decrease":
            direction[col] = -1.0
        for branch in marker_set.branches:
            if 0 <= branch < branch_map.shape[1]:
                branch_map[col, branch] = marker_set.weight
    # Normalize branch weights so inactive branches default to 1
    branch_map[branch_map <= 0] = 1.0
    return MarkerConfigResult(
        sets=markers,
        matrix=matrix,
        gene_counts=counts,
        branch_map=branch_map,
        direction=direction,
    )
