#!/usr/bin/env python
"""Generate velocity streamline plots for every model defined in a benchmark config."""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData, read_h5ad
from scipy import sparse

try:
    import scvelo as scv  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover
    scv = None

from scvi.experimental.benchmark.config import (
    BenchmarkConfig,
    DatasetConfig,
    load_benchmark_config,
)
from scvi.experimental.benchmark.data import (
    ensure_dataset_available,
    prepare_velocity_layers,
)
from scvi.experimental.benchmark.models import (
    build_model,
    _extract_celltype_statistics,
    _extract_neighbor_indices,
    _extract_neighbor_weights,
)
from scvi.experimental.benchmark.preprocess import (
    apply_preprocessing,
    ensure_neighbor_graph,
    subset_adata,
)
from scvi.experimental.benchmark.utils import set_random_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render velocity streamline plots for models defined in a benchmark config.",
    )
    parser.add_argument("--config", required=True, type=Path, help="Benchmark YAML configuration.")
    parser.add_argument(
        "--datasets",
        nargs="*",
        help="Optional subset of dataset names to process (defaults to all in config).",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        help="Optional subset of model names to process (defaults to all in config).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory to store streamline figures. Defaults to <config.output_dir>/streamlines.",
    )
    parser.add_argument(
        "--max-cells",
        type=int,
        default=None,
        help="Optional cap on number of cells per dataset (random subsample) to accelerate plotting.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for subsampling and model initialisation (overrides config seed).",
    )
    return parser.parse_args()


def ensure_plot_backend():
    if scv is None:
        raise ModuleNotFoundError(
            "scvelo is required to generate velocity streamplots. Install it before running."
        )
    scv.settings.set_figure_params(dpi=220, frameon=False)
    scv.settings.verbosity = 0


def load_dataset(cfg: DatasetConfig) -> AnnData:
    path = ensure_dataset_available(cfg)
    adata = read_h5ad(path)
    return adata


def compute_embedding(adata: AnnData) -> np.ndarray:
    for key in ("X_umap", "X_stream", "X_tsne", "X_pca"):
        if key in adata.obsm:
            embedding = np.asarray(adata.obsm[key], dtype=np.float32)
            if embedding.shape[0] == adata.n_obs and embedding.shape[1] >= 2:
                return embedding[:, :2]
    raise ValueError(
        "Could not locate a 2D embedding (UMAP/stream/tsne/PCA). Ensure preprocessing computed one."
    )


def neighbor_weights_from_adata(adata: AnnData, indices: np.ndarray) -> np.ndarray:
    neighbors_uns = adata.uns.get("neighbors", {})
    connectivities_key = neighbors_uns.get("connectivities_key", "connectivities")
    connectivities = adata.obsp.get(connectivities_key)
    if connectivities is None:
        return np.ones_like(indices, dtype=np.float32)
    weights = _extract_neighbor_weights(connectivities, indices)
    if weights is None:
        return np.ones_like(indices, dtype=np.float32)
    return weights.astype(np.float32)


def build_stream_graph(
    embedding: np.ndarray,
    neighbor_indices: np.ndarray,
    neighbor_weights: np.ndarray,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix, Dict[str, object]]:
    n_cells = embedding.shape[0]
    k_neighbors = neighbor_indices.shape[1]
    row_idx = np.repeat(np.arange(n_cells), k_neighbors)
    col_idx = neighbor_indices.reshape(-1)
    weights = neighbor_weights.reshape(-1)

    connectivities = sparse.csr_matrix((weights, (row_idx, col_idx)), shape=(n_cells, n_cells))
    connectivities = connectivities.maximum(connectivities.T)
    connectivities.setdiag(1.0)

    diffs = embedding[col_idx] - embedding[row_idx]
    dists = np.linalg.norm(diffs, axis=1)
    distances = sparse.csr_matrix((dists, (row_idx, col_idx)), shape=(n_cells, n_cells))
    distances = distances.maximum(distances.T)
    distances.setdiag(0.0)

    neighbors_uns = {
        "connectivities_key": "connectivities",
        "distances_key": "distances",
        "params": {
            "n_neighbors": int(k_neighbors),
            "method": "user_defined",
            "metric": "euclidean",
        },
    }
    return connectivities, distances, neighbors_uns


def plot_streamlines(
    dataset_cfg: DatasetConfig,
    adata: AnnData,
    embedding: np.ndarray,
    velocities: Dict[str, np.ndarray],
    neighbor_indices: np.ndarray,
    neighbor_weights: np.ndarray,
    output_dir: Path,
) -> Dict[str, Path]:
    ensure_plot_backend()

    if embedding.shape[1] < 2:
        warnings.warn(
            f"Embedding for dataset {dataset_cfg.name} is not 2D; skipping streamline plots.",
            RuntimeWarning,
        )
        return {}

    x_span = embedding[:, 0].ptp()
    y_span = embedding[:, 1].ptp()
    if np.isclose(x_span, 0.0) or np.isclose(y_span, 0.0):
        warnings.warn(
            f"Embedding for dataset {dataset_cfg.name} collapses to a line; skipping streamline plots.",
            RuntimeWarning,
        )
        return {}

    connectivities, distances, neighbors_uns = build_stream_graph(
        embedding, neighbor_indices, neighbor_weights
    )

    cell_types = None
    if dataset_cfg.celltype_key and dataset_cfg.celltype_key in adata.obs:
        cell_types = adata.obs[dataset_cfg.celltype_key]
    unique_labels = None
    if cell_types is not None:
        cell_types = pd.Categorical(cell_types)
        unique_labels = cell_types.categories

    output_dir.mkdir(parents=True, exist_ok=True)
    saved: Dict[str, Path] = {}

    for model_name, velocity_matrix in velocities.items():
        velocity_matrix = np.asarray(velocity_matrix, dtype=np.float32)
        if velocity_matrix.ndim != 2 or velocity_matrix.shape[0] != adata.n_obs:
            warnings.warn(
                f"Velocity matrix for model `{model_name}` has unexpected shape, skipping plot.",
                RuntimeWarning,
            )
            continue

        adata_stream = adata.copy()
        adata_stream.obsm["X_stream"] = embedding.astype(np.float32)
        adata_stream.obsp["connectivities"] = connectivities
        adata_stream.obsp["distances"] = distances
        adata_stream.uns["neighbors"] = neighbors_uns
        adata_stream.layers["velocity"] = velocity_matrix
        if dataset_cfg.spliced_layer not in adata_stream.layers:
            warnings.warn(
                f"Spliced layer `{dataset_cfg.spliced_layer}` missing for dataset `{dataset_cfg.name}`.",
                RuntimeWarning,
            )
            continue
        if "Ms" not in adata_stream.layers:
            adata_stream.layers["Ms"] = adata_stream.layers[dataset_cfg.spliced_layer]
        if cell_types is not None:
            adata_stream.obs["celltype"] = cell_types

        scv.tl.velocity_graph(
            adata_stream,
            vkey="velocity",
            xkey=dataset_cfg.spliced_layer,
            n_jobs=1,
            mode_neighbors="distances",
            approx=False,
            show_progress_bar=False,
        )
        scv.tl.velocity_embedding(
            adata_stream,
            basis="stream",
            vkey="velocity",
            retain_scale=True,
            use_negative_cosines=False,
        )

        fig_path = output_dir / f"{dataset_cfg.name}_{model_name}_streamlines.png"
        fig, ax = plt.subplots(figsize=(5, 4))
        scv.pl.velocity_embedding_stream(
            adata_stream,
            basis="stream",
            vkey="velocity",
            color="celltype" if cell_types is not None else None,
            legend_loc="right margin",
            ax=ax,
            show=False,
        )
        fig.tight_layout()
        fig.savefig(fig_path, dpi=240)
        plt.close(fig)
        saved[model_name] = fig_path

    return saved


def generate_streamlines_for_dataset(
    config: BenchmarkConfig,
    dataset_cfg: DatasetConfig,
    model_names: Optional[Iterable[str]],
    output_root: Path,
    max_cells: Optional[int],
    seed: Optional[int],
):
    print(f"\n=== Dataset: {dataset_cfg.name} ===")
    raw_adata = load_dataset(dataset_cfg)
    resolved_cfg = prepare_velocity_layers(raw_adata, dataset_cfg)
    if max_cells is not None and max_cells < raw_adata.n_obs:
        raw_adata = subset_adata(raw_adata, max_cells, random_state=seed)
        resolved_cfg = prepare_velocity_layers(raw_adata, resolved_cfg)

    adata = raw_adata.copy()
    adata.X = adata.layers[resolved_cfg.spliced_layer]
    apply_preprocessing(adata, resolved_cfg.preprocess)
    ensure_neighbor_graph(adata, resolved_cfg.preprocess.n_neighbors)

    embedding = compute_embedding(adata)
    neighbor_indices = _extract_neighbor_indices(adata)
    neighbor_weights = neighbor_weights_from_adata(adata, neighbor_indices)

    selected_models = [
        m for m in config.models if (not model_names or m.name in model_names)
    ]
    velocities: Dict[str, np.ndarray] = {}
    metadata_records = {}

    for model_cfg in selected_models:
        print(f"  -> Running model `{model_cfg.name}`")
        model = build_model(model_cfg, resolved_cfg)
        set_random_seed(seed if seed is not None else config.reproducibility_seed)
        adata_model = adata.copy()
        fit_info = model.fit(adata_model)
        velocity_info = model.get_velocity(adata_model)
        velocities[model_cfg.name] = velocity_info["values"]

        metadata_records[model_cfg.name] = {
            "type": model_cfg.type,
            "fit_args": model_cfg.fit_args,
            "init_args": model_cfg.init_args,
            "velocity_args": model_cfg.velocity_args,
            "spliced_layer": resolved_cfg.spliced_layer,
            "unspliced_layer": resolved_cfg.unspliced_layer,
        }
        if model_cfg.type == "velovi_transformer":
            metadata_records[model_cfg.name]["transformer_args"] = model_cfg.transformer_args
        if fit_info.get("fit_time") is not None:
            metadata_records[model_cfg.name]["fit_time"] = fit_info["fit_time"]
        if velocity_info.get("velocity_time") is not None:
            metadata_records[model_cfg.name]["velocity_time"] = velocity_info["velocity_time"]

        model.cleanup()

    if not velocities:
        warnings.warn(f"No velocities computed for dataset {dataset_cfg.name}.", RuntimeWarning)
        return

    dataset_dir = output_root / dataset_cfg.name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = dataset_dir / "model_parameters.json"
    metadata_path.write_text(json.dumps(metadata_records, indent=2, default=str))

    saved_paths = plot_streamlines(
        resolved_cfg,
        adata,
        embedding,
        velocities,
        neighbor_indices,
        neighbor_weights,
        dataset_dir,
    )
    for model_name, fig_path in saved_paths.items():
        print(f"    Saved streamline plot for {model_name} -> {fig_path}")


def main():
    args = parse_args()
    config = load_benchmark_config(args.config)

    seed = args.seed if args.seed is not None else config.reproducibility_seed

    output_dir = args.output_dir or (config.expand_output_dir() / "streamlines")
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_names = args.datasets or [d.name for d in config.datasets]
    missing = set(dataset_names) - {d.name for d in config.datasets}
    if missing:
        raise ValueError(f"Datasets not found in config: {sorted(missing)}")

    for dataset_cfg in config.datasets:
        if dataset_cfg.name not in dataset_names:
            continue
        generate_streamlines_for_dataset(
            config=config,
            dataset_cfg=dataset_cfg,
            model_names=args.models,
            output_root=output_dir,
            max_cells=args.max_cells,
            seed=seed,
        )


if __name__ == "__main__":
    main()
