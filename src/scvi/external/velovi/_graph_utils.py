from __future__ import annotations

from typing import Tuple

import numpy as np
from sklearn.neighbors import NearestNeighbors


def _build_knn_graph(
    features: np.ndarray,
    n_neighbors: int = 15,
    metric: str = "euclidean",
) -> Tuple[np.ndarray, np.ndarray]:
    """Internal helper to build kNN graph from provided features."""
    nn = NearestNeighbors(n_neighbors=n_neighbors + 1, metric=metric)
    nn.fit(features)
    distances, indices = nn.kneighbors(features)
    indices = indices[:, 1:]
    distances = distances[:, 1:]

    eps = 1e-12
    weights = np.exp(-distances)
    weights = weights / (weights.sum(axis=1, keepdims=True) + eps)
    return indices.astype(np.int64), weights.astype(np.float32)


def construct_expression_graph(
    spliced: np.ndarray,
    unspliced: np.ndarray,
    n_neighbors: int = 15,
    metric: str = "euclidean",
) -> Tuple[np.ndarray, np.ndarray]:
    """Build kNN graph from concatenated expression features."""
    features = np.concatenate([spliced, unspliced], axis=1)
    return _build_knn_graph(features, n_neighbors=n_neighbors, metric=metric)


def add_graph_to_adata(
    adata,
    spliced_layer: str,
    unspliced_layer: str,
    key_prefix: str = "velovi_gnn",
    n_neighbors: int = 15,
    metric: str = "euclidean",
) -> Tuple[np.ndarray, np.ndarray]:
    """Convenience wrapper to attach neighbor graph info to AnnData."""
    spliced = adata.layers[spliced_layer]
    unspliced = adata.layers[unspliced_layer]
    if hasattr(spliced, "toarray"):
        spliced = spliced.toarray()
    if hasattr(unspliced, "toarray"):
        unspliced = unspliced.toarray()
    indices, weights = construct_expression_graph(
        spliced=spliced,
        unspliced=unspliced,
        n_neighbors=n_neighbors,
        metric=metric,
    )
    adata.obsm[f"{key_prefix}_indices"] = indices
    adata.obsm[f"{key_prefix}_weights"] = weights
    return indices, weights


def construct_feature_graph(
    features: np.ndarray,
    n_neighbors: int = 15,
    metric: str = "euclidean",
) -> Tuple[np.ndarray, np.ndarray]:
    """Build kNN graph directly from provided feature matrix."""
    return _build_knn_graph(features, n_neighbors=n_neighbors, metric=metric)
