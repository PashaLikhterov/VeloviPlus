from __future__ import annotations

from typing import Dict, Literal, Optional, Tuple

import numpy as np

try:  # optional dependency for logging
    import wandb
except ModuleNotFoundError:  # pragma: no cover
    wandb = None  # type: ignore

import scvelo as scv
import scanpy as sc

from scvi.external.velovi._graph_utils import add_graph_to_adata

from .config import StreamEmbeddingResult, TrainingConfig
from .datasets import DatasetConfig


def start_wandb_run(config: TrainingConfig, dataset_name: str):
    if not config.use_wandb or wandb is None:
        return None

    wandb_config = {
        "dataset": dataset_name,
        "warmup_epochs": config.warmup_epochs,
        "total_epochs": config.total_epochs,
        "batch_size": config.batch_size,
        "baseline_encoder": config.baseline_encoder,
        "transformer_encoder_hidden_dim": config.transformer_encoder_hidden_dim,
        "transformer_encoder_layers": config.transformer_encoder_layers,
        "transformer_encoder_heads": config.transformer_encoder_heads,
        "transformer_encoder_dropout": config.transformer_encoder_dropout,
        "transformer_encoder_max_neighbors": config.transformer_encoder_max_neighbors,
        "transformer_encoder_neighbor_weight": config.transformer_encoder_neighbor_weight,
        "gnn_neighbor_source": config.gnn_neighbor_source,
        "enable_transformer": config.enable_transformer_refinement,
        "stream_embed_method": config.stream_embed_method,
        "stream_umap_neighbors": config.stream_umap_neighbors,
        "transformer_epochs": config.transformer_epochs,
        "transformer_hidden_dim": config.transformer_hidden_dim,
        "transformer_layers": config.transformer_layers,
        "transformer_heads": config.transformer_heads,
        "transformer_weight_smooth": config.transformer_weight_smooth,
        "transformer_weight_direction": config.transformer_weight_direction,
        "transformer_weight_alignment": config.transformer_weight_alignment,
        "transformer_weight_supervised": config.transformer_weight_supervised,
        "transformer_weight_celltype": config.transformer_weight_celltype,
        "transformer_weight_celltype_dir": config.transformer_weight_celltype_dir,
        "transformer_weight_celltype_mag": config.transformer_weight_celltype_mag,
        "transformer_celltype_penalty": config.transformer_celltype_penalty,
        "transformer_aux_cluster_loss_weight": config.transformer_aux_cluster_loss_weight,
        "transformer_neighbor_max_distance": config.transformer_neighbor_max_distance,
        "transformer_max_neighbors": config.transformer_max_neighbors,
        "transformer_residual_to_baseline": config.transformer_residual_to_baseline,
        "transformer_use_tivelo": config.transformer_use_tivelo,
        "transformer_tivelo_resolution": config.transformer_tivelo_resolution,
        "transformer_tivelo_threshold": config.transformer_tivelo_threshold,
        "transformer_tivelo_threshold_trans": config.transformer_tivelo_threshold_trans,
    }

    run = wandb.init(
        project=config.wandb_project,
        entity=config.wandb_entity,
        group=config.wandb_run_group,
        config=wandb_config,
        reinit=True,
    )
    return run


def log_training_history(wandb_run, history, prefix: str) -> None:
    if wandb_run is None or history is None or wandb is None:
        return
    for key, values in history.items():
        if not isinstance(values, (list, tuple)):
            continue
        for step, value in enumerate(values):
            try:
                wandb_run.log({f"{prefix}/{key}": value}, step=step)
            except Exception:  # pragma: no cover
                continue


def infer_cell_type_labels(adata, config: DatasetConfig) -> Tuple[Optional[np.ndarray], Optional[str]]:
    preferred_key = getattr(config, "celltype_key", None)
    if preferred_key and preferred_key in adata.obs:
        key = preferred_key
    else:
        candidate_keys = [
            "cell_type",
            "celltype",
            "cell_types",
            "celltype_major",
            "clusters",
            "cluster",
            "leiden",
            "louvain",
            "annotation",
            "annotations",
            "labels",
            "cell_state",
            "state",
        ]
        key = next((candidate for candidate in candidate_keys if candidate in adata.obs), None)

    if key is None:
        return None, None
    labels = adata.obs[key].astype(str).to_numpy()
    unique_labels = np.unique(labels)
    if unique_labels.size == 0 or unique_labels.size > 50:
        return None, None
    return labels, key


def compute_stream_embedding(
    adata,
    config: TrainingConfig,
    dataset_config: DatasetConfig,
) -> StreamEmbeddingResult:
    data_matrix = adata.layers[dataset_config.spliced_layer]
    if hasattr(data_matrix, "toarray"):
        data_matrix = data_matrix.toarray()
    data_matrix = np.asarray(data_matrix, dtype=np.float32)
    if data_matrix.shape[1] < 2:
        return StreamEmbeddingResult(None, None, None)
    centered = data_matrix - data_matrix.mean(axis=0, keepdims=True)
    if config.stream_embed_standardize:
        std = centered.std(axis=0, keepdims=True)
        std[std < 1e-6] = 1e-6
        centered = centered / std
    try:
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return StreamEmbeddingResult(None, None, None)

    method = getattr(config, "stream_embed_method", "pca")
    preferred_basis = getattr(dataset_config, "plot_basis", None)

    if method == "umap":
        pca_dim = max(2, min(config.stream_embed_pca_components, vt.shape[0]))
        components = vt[:pca_dim]
        scores = centered @ components.T
        basis_key = None
        if preferred_basis:
            basis_key = f"X_{preferred_basis}"
        if basis_key is None and "X_umap" in adata.obsm:
            basis_key = "X_umap"
        if basis_key is not None and basis_key in adata.obsm:
            embedding = np.asarray(adata.obsm[basis_key], dtype=np.float32)
            if embedding.shape[0] == scores.shape[0]:
                try:
                    projection = np.linalg.lstsq(scores, embedding, rcond=None)[0]
                except np.linalg.LinAlgError:
                    projection = None
                return StreamEmbeddingResult(
                    embedding.astype(np.float32),
                    components.astype(np.float32),
                    projection.astype(np.float32) if projection is not None else None,
                )
        try:
            import umap

            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=config.stream_umap_neighbors,
                min_dist=config.stream_umap_min_dist,
                spread=config.stream_umap_spread,
                random_state=0,
            )
            embedding = reducer.fit_transform(scores)
            projection = np.linalg.lstsq(scores, embedding, rcond=None)[0]
            return StreamEmbeddingResult(
                embedding.astype(np.float32),
                components.astype(np.float32),
                projection.astype(np.float32),
            )
        except ModuleNotFoundError:
            method = "pca"

    n_components = min(2, vt.shape[0])
    if n_components < 2:
        return StreamEmbeddingResult(None, None, None)
    components = vt[:n_components]
    embedding = centered @ components.T
    return StreamEmbeddingResult(
        embedding.astype(np.float32),
        components.astype(np.float32),
        None,
    )


def compute_latent_stream_embedding(
    latent: Optional[np.ndarray],
    config: TrainingConfig,
) -> StreamEmbeddingResult:
    if latent is None:
        return StreamEmbeddingResult(None, None, None)
    data_matrix = np.asarray(latent, dtype=np.float32)
    if data_matrix.ndim != 2 or data_matrix.shape[1] < 2:
        return StreamEmbeddingResult(None, None, None)
    centered = data_matrix - data_matrix.mean(axis=0, keepdims=True)
    if config.stream_embed_standardize:
        std = centered.std(axis=0, keepdims=True)
        std[std < 1e-6] = 1e-6
        centered = centered / std
    try:
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return StreamEmbeddingResult(None, None, None)

    method = getattr(config, "stream_embed_method", "pca")
    if method == "umap":
        pca_dim = max(2, min(config.stream_embed_pca_components, vt.shape[0]))
        components = vt[:pca_dim]
        scores = centered @ components.T
        try:
            import umap

            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=config.stream_umap_neighbors,
                min_dist=config.stream_umap_min_dist,
                spread=config.stream_umap_spread,
                random_state=0,
            )
            embedding = reducer.fit_transform(scores)
            projection = np.linalg.lstsq(scores, embedding, rcond=None)[0]
            return StreamEmbeddingResult(
                embedding.astype(np.float32),
                components.astype(np.float32),
                projection.astype(np.float32),
            )
        except ModuleNotFoundError:
            method = "pca"

    n_components = min(2, vt.shape[0])
    if n_components < 2:
        return StreamEmbeddingResult(None, None, None)
    components = vt[:n_components]
    embedding = centered @ components.T
    return StreamEmbeddingResult(
        embedding.astype(np.float32),
        components.astype(np.float32),
        None,
    )


def prepare_alignment_vectors(
    adata,
    neighbor_indices: np.ndarray,
    config: DatasetConfig,
    data_matrix: Optional[np.ndarray] = None,
) -> np.ndarray:
    if data_matrix is None:
        spliced = adata.layers[config.spliced_layer]
        if hasattr(spliced, "toarray"):
            spliced = spliced.toarray()
        matrix = spliced
    else:
        matrix = data_matrix
    matrix = np.asarray(matrix, dtype=np.float32)
    primary_neighbor = neighbor_indices[:, 0]
    alignment = matrix[primary_neighbor] - matrix
    return alignment.astype(np.float32, copy=False)


def add_stream_graph(adata, dataset_config: DatasetConfig, key_prefix: str = "velovi_gnn"):
    indices, weights = add_graph_to_adata(
        adata,
        spliced_layer=dataset_config.spliced_layer,
        unspliced_layer=dataset_config.unspliced_layer,
        key_prefix=key_prefix,
        n_neighbors=15,
    )
    return indices, weights
