from __future__ import annotations

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def keep_type(adata, nodes, target, cluster_key):
    labels = adata.obs[cluster_key]
    values = labels.values if hasattr(labels, "values") else labels.to_numpy()
    return nodes[values[nodes] == target]


def cross_boundary_correctness(
    adata,
    cluster_key: str,
    velocity_key: str,
    cluster_edges,
    x_emb: str = "X_umap",
    velocity_basis: str = "umap",
) -> float:
    scores = {}
    if x_emb not in adata.obsm:
        raise KeyError(f"{x_emb} not found in adata.obsm")
    velocity_emb_key = f"{velocity_key}_{velocity_basis}"
    if velocity_emb_key not in adata.obsm:
        raise KeyError(f"{velocity_emb_key} not found in adata.obsm")
    v_emb = adata.obsm[velocity_emb_key]
    x_coords = adata.obsm[x_emb]
    neighbors = adata.uns["neighbors"]["indices"]
    for src, dst in cluster_edges:
        sel = adata.obs[cluster_key] == src
        nbs = neighbors[sel]
        boundary_nodes = map(lambda nodes: keep_type(adata, nodes, dst, cluster_key), nbs)
        x_points = x_coords[sel]
        x_velocities = v_emb[sel]
        type_scores = []
        for x_pos, x_vel, nodes in zip(x_points, x_velocities, boundary_nodes):
            if len(nodes) == 0:
                continue
            position_diff = x_coords[nodes] - x_pos
            dir_scores = cosine_similarity(position_diff, x_vel.reshape(1, -1)).flatten()
            type_scores.append(np.mean(dir_scores))
        scores[(src, dst)] = np.mean(type_scores) if type_scores else np.nan
    valid = [sc for sc in scores.values() if not np.isnan(sc)]
    return float(np.mean(valid)) if valid else float("nan")


def inner_cluster_coh(
    adata,
    cluster_key: str,
    velocity_key: str,
) -> float:
    clusters = np.unique(adata.obs[cluster_key])
    scores = []
    velocities = adata.layers[velocity_key]
    neighbors = adata.uns["neighbors"]["indices"]
    for cat in clusters:
        sel = adata.obs[cluster_key] == cat
        nbs = neighbors[sel]
        same_cat_nodes = map(lambda nodes: keep_type(adata, nodes, cat, cluster_key), nbs)
        cat_vels = velocities[sel]
        cat_scores = [
            cosine_similarity(cat_vels[[idx]], velocities[nodes]).mean()
            for idx, nodes in enumerate(same_cat_nodes)
            if len(nodes) > 0
        ]
        if cat_scores:
            scores.append(np.mean(cat_scores))
    return float(np.mean(scores)) if scores else float("nan")
