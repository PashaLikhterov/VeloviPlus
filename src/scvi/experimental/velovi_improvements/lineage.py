from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class LineageResult:
    clusters: list[int]
    score: float


def _cluster_metrics(cluster_labels: np.ndarray, values: Optional[np.ndarray]) -> np.ndarray:
    n_clusters = int(cluster_labels.max()) + 1
    if values is None:
        return np.zeros(n_clusters, dtype=np.float32)
    sums = np.bincount(cluster_labels, weights=values, minlength=n_clusters)
    counts = np.bincount(cluster_labels, minlength=n_clusters).astype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        means = np.divide(sums, counts, out=np.zeros_like(sums, dtype=np.float32), where=counts > 0)
    # Normalize to zero mean / unit variance for stability
    means = means.astype(np.float32)
    means = means - means.mean()
    std = means.std()
    if std > 1e-6:
        means = means / std
    return means


def detect_lineages(
    cluster_labels: np.ndarray,
    neighbor_indices: np.ndarray,
    cycle_scores: Optional[np.ndarray],
    latent_time: Optional[np.ndarray],
    max_lineages: int,
    min_edge_count: int = 25,
    min_metric_diff: float = 0.05,
) -> list[LineageResult]:
    if cluster_labels is None or cluster_labels.size == 0:
        return []
    n_clusters = int(cluster_labels.max()) + 1
    if n_clusters <= 1:
        return []
    metric = None
    if cycle_scores is not None:
        metric = _cluster_metrics(cluster_labels, cycle_scores)
    elif latent_time is not None:
        metric = _cluster_metrics(cluster_labels, latent_time)
    else:
        metric = np.zeros(n_clusters, dtype=np.float32)

    counts = np.zeros((n_clusters, n_clusters), dtype=np.int32)
    for idx, neighbors in enumerate(neighbor_indices):
        src = int(cluster_labels[idx])
        for nb in neighbors:
            dst = int(cluster_labels[nb])
            if src == dst:
                continue
            counts[src, dst] += 1

    edges: dict[int, list[tuple[int, float]]] = {}
    active_clusters: set[int] = set()
    for src in range(n_clusters):
        for dst in range(n_clusters):
            if src == dst:
                continue
            total = counts[src, dst] + counts[dst, src]
            if total < min_edge_count:
                continue
            diff = metric[src] - metric[dst]
            if abs(diff) < min_metric_diff:
                continue
            if diff > 0:
                origin, target = src, dst
                score = diff
            else:
                origin, target = dst, src
                score = -diff
            edges.setdefault(origin, []).append((target, score))
            active_clusters.add(origin)
            active_clusters.add(target)

    if not edges:
        return []

    for src, targets in edges.items():
        edges[src] = sorted(targets, key=lambda tpl: tpl[1], reverse=True)

    unused = set(active_clusters)
    lineages: list[LineageResult] = []
    while unused and len(lineages) < max_lineages:
        start = max(unused, key=lambda c: metric[c])
        path = [start]
        path_scores: list[float] = []
        current = start
        visited = set(path)
        while True:
            candidates = edges.get(current)
            if not candidates:
                break
            next_cluster = None
            next_score = None
            for dst, score in candidates:
                if dst in visited:
                    continue
                next_cluster = dst
                next_score = score
                break
            if next_cluster is None:
                break
            path.append(next_cluster)
            visited.add(next_cluster)
            if next_score is not None:
                path_scores.append(next_score)
            current = next_cluster
        lineage_score = float(np.mean(path_scores)) if path_scores else 0.0
        lineages.append(LineageResult(clusters=path, score=lineage_score))
        for node in path:
            unused.discard(node)

    remaining = sorted(unused, key=lambda c: metric[c], reverse=True)
    for cluster in remaining:
        if len(lineages) >= max_lineages:
            break
        lineages.append(LineageResult(clusters=[cluster], score=0.0))

    return lineages


def build_branch_prior_matrix(
    cluster_labels: np.ndarray,
    lineages: list[LineageResult],
    branch_count: int,
    epsilon: float = 1e-3,
) -> np.ndarray:
    n_cells = cluster_labels.shape[0]
    priors = np.full((n_cells, branch_count), fill_value=epsilon, dtype=np.float32)
    assigned_clusters: list[set[int]] = [
        set(lineage.clusters) for lineage in lineages[:branch_count]
    ]
    for branch_idx, clusters in enumerate(assigned_clusters):
        if not clusters:
            continue
        mask = np.isin(cluster_labels, list(clusters))
        priors[mask, branch_idx] = 1.0
    # Normalize rows
    row_sums = priors.sum(axis=1, keepdims=True)
    zero_mask = row_sums.squeeze(-1) <= 0
    if np.any(zero_mask):
        priors[zero_mask] = 1.0 / float(branch_count)
        row_sums = priors.sum(axis=1, keepdims=True)
    priors = priors / np.clip(row_sums, 1e-6, None)
    return priors
