from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import scanpy as sc

try:  # pragma: no cover - optional dependency
    import scvelo as scv
except ModuleNotFoundError:  # pragma: no cover
    scv = None

from scipy import sparse

# TIVelo imports (local copy shipped with repository)
from scvi.experimental.tivelo.path.process import process_path
from scvi.experimental.tivelo.direction.correct import correct_path
from scvi.experimental.tivelo.velocity.DTI import get_child_dict, get_d_nn


@dataclass
class TIVeloGuidance:
    velocity_target: np.ndarray
    mask: np.ndarray
    group_key: str
    embedding_key: str
    child_dict: Dict[str, list[str]]
    directed_neighbors: np.ndarray
    same_cluster_mask: np.ndarray


def _ensure_neighbors(adata, n_neighbors: int = 30, n_pcs: int = 30) -> None:
    if "neighbors" not in adata.uns:
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs)
    if "distances" not in adata.obsp:
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs)
    if "indices" not in adata.uns["neighbors"]:
        distances = adata.obsp["distances"]
        if sparse.issparse(distances):
            distances = distances.toarray()
        sorted_indices = np.argsort(distances + np.eye(distances.shape[0]), axis=1)
        sorted_indices = np.flip(sorted_indices, axis=1)
        adata.uns["neighbors"]["indices"] = sorted_indices[:, :n_neighbors]


def compute_tivelo_guidance(
    adata,
    *,
    spliced_layer: str,
    unspliced_layer: str,
    group_key: Optional[str] = None,
    embedding_key: Optional[str] = None,
    resolution: float = 0.6,
    start_mode: str = "stochastic",
    rev_stat: str = "mean",
    threshold: float = 0.1,
    threshold_trans: float = 1.0,
    t_diff: float = 0.1,
    t_transition: float = 1.0,
    njobs: int = -1,
) -> TIVeloGuidance:
    """
    Compute directed cluster guidance following the TIVelo algorithm.

    Returns a per-cell velocity target that nudges each observation towards
    its downstream cluster mean, together with masks describing which cells
    received supervision.
    """

    if scv is None:
        raise ModuleNotFoundError(
            "scvelo is required to compute TIVelo guidance. "
            "Install it via `pip install scvelo`."
        )

    work_adata = adata.copy()

    if group_key is None or group_key not in work_adata.obs_keys():
        sc.tl.leiden(work_adata, resolution=resolution)
        group_key = "tivelo_leiden"

    _ensure_neighbors(work_adata, n_neighbors=30, n_pcs=30)

    if embedding_key is None or embedding_key not in work_adata.obsm_keys():
        scv.tl.umap(work_adata)
        embedding_key = "X_umap"

    path_dict, _ = process_path(
        work_adata,
        group_key,
        embedding_key,
        njobs=njobs,
        start_mode=start_mode,
    )
    path_dict, _ = correct_path(
        path_dict,
        work_adata,
        group_key,
        rev_stat=rev_stat,
        tree_gene=None,
        root_select="connectivities",
    )

    child_dict, level_dict, _ = get_child_dict(
        work_adata,
        group_key,
        path_dict,
        threshold=threshold,
        threshold_trans=threshold_trans,
        adjust=False,
    )
    start_node = next(iter(level_dict.keys()))
    d_nn, knn, child_c, same_c = get_d_nn(
        work_adata,
        group_key,
        child_dict,
        start_node,
        emb_key=embedding_key,
        root_select="connectivities",
    )

    spliced = work_adata.layers[spliced_layer]
    if sparse.issparse(spliced):
        spliced = spliced.toarray()
    spliced = np.asarray(spliced, dtype=np.float32)

    directional_sum = d_nn @ spliced
    neighbor_counts = d_nn.sum(axis=1, keepdims=True)
    neighbor_counts = np.asarray(neighbor_counts, dtype=np.float32)
    zero_mask = neighbor_counts <= 1e-6
    neighbor_counts[zero_mask] = 1.0

    directed_mean = directional_sum / neighbor_counts
    velocity_target = directed_mean - spliced
    supervision_mask = (~zero_mask.squeeze()).astype(np.float32)

    return TIVeloGuidance(
        velocity_target=velocity_target.astype(np.float32),
        mask=supervision_mask,
        group_key=group_key,
        embedding_key=embedding_key,
        child_dict=child_dict,
        directed_neighbors=d_nn.astype(np.float32),
        same_cluster_mask=same_c.astype(np.float32),
    )
