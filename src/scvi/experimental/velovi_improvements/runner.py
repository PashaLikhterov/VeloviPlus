from __future__ import annotations

import argparse
import contextlib
import os
import resource
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple
import warnings

import numpy as np
import pandas as pd
from anndata import AnnData
import matplotlib.pyplot as plt
from scipy import sparse
import scvelo as scv
from scvi import settings
from scvi.external.velovi import (
    LatentEmbeddingGraphBuilder,
    VELOVI,
    VELOVITransformerEncoder,
    VELOVIWithGNN,
    smooth_velocities_with_graph,
)
from scvi.external.velovi._graph_utils import construct_feature_graph

from .datasets import (
    VELOVI_DATASETS,
    DatasetConfig,
    DEFAULT_PREPROCESS,
    load_dataset,
    resolve_dataset_name,
)
from .utils import add_velovi_outputs_to_adata
from .metrics import result_to_dict, summarize_velocity_metrics
from .transformer_velocity import TransformerConfig, refine_velocities_with_transformer
from scvi.experimental.tivelo.main import tivelo as run_tivelo
from scvi.experimental.tivelo.path.process import process_path
from scvi.experimental.tivelo.direction.correct import correct_path
from scvi.experimental.tivelo.utils import metrics as tivelo_metrics
from . import benchmark
from .analysis import generate_variant_streamplots, generate_graph_diagnostics
from .config import StreamEmbeddingResult, TrainingConfig
from .training import (
    add_stream_graph,
    compute_latent_stream_embedding,
    compute_stream_embedding,
    infer_cell_type_labels,
    log_training_history,
    prepare_alignment_vectors,
    start_wandb_run,
)

warnings.filterwarnings("ignore")


@dataclass
class MetricContext:
    name: str
    neighbor_indices: np.ndarray
    alignment_vectors: Optional[np.ndarray]


def _build_boundary_secondary_graph(
    cluster_labels: Optional[np.ndarray],
    edges_int: Optional[List[Tuple[int, int]]],
    primary_indices: np.ndarray,
    primary_weights: np.ndarray,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Build a boundary-aware neighbor set from a primary kNN graph.

    For each anchor, restrict neighbors to those whose cluster label is in the
    allowed target set based on directed edges. If no valid neighbors exist,
    fall back to the original primary neighbors.
    """
    if cluster_labels is None or edges_int is None or not len(edges_int):
        return None
    n, k = primary_indices.shape
    labels = np.asarray(cluster_labels).astype(int)
    # Build map: src_label -> set(dst_labels)
    edge_map: Dict[int, set] = {}
    for s, t in edges_int:
        edge_map.setdefault(int(s), set()).add(int(t))

    boundary_indices = np.full_like(primary_indices, fill_value=0)
    boundary_weights = np.zeros_like(primary_weights, dtype=np.float32)

    for i in range(n):
        src = int(labels[i])
        allowed = edge_map.get(src)
        nbrs = primary_indices[i]
        w = primary_weights[i]
        if allowed:
            nbr_labels = labels[nbrs]
            mask = np.isin(nbr_labels, list(allowed))
            if np.any(mask):
                sel_idx = nbrs[mask]
                sel_w = w[mask]
                # Renormalize to sum 1
                s = float(sel_w.sum())
                if s > 1e-8:
                    sel_w = sel_w / s
                # If fewer than k, pad with anchor (zero weight)
                if sel_idx.shape[0] < k:
                    pad = np.full(k - sel_idx.shape[0], i, dtype=nbrs.dtype)
                    pad_w = np.zeros(k - sel_w.shape[0], dtype=sel_w.dtype)
                    sel_idx = np.concatenate([sel_idx, pad], axis=0)
                    sel_w = np.concatenate([sel_w, pad_w], axis=0)
                boundary_indices[i] = sel_idx[:k]
                boundary_weights[i] = sel_w[:k].astype(np.float32)
                continue
        # Fallback: use primary neighbors/weights (already normalized)
        boundary_indices[i] = nbrs
        # Ensure weights are normalized
        s = float(w.sum())
        boundary_weights[i] = (w / s if s > 1e-8 else w).astype(np.float32)

    return boundary_indices.astype(np.int64), boundary_weights.astype(np.float32)


def _build_future_neighbor_graph(
    latent_time: np.ndarray,
    neighbor_indices: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Construct a future-neighbor graph based on latent times for continuity loss."""
    lat = np.asarray(latent_time).reshape(-1)
    n, k = neighbor_indices.shape
    future_idx = np.empty_like(neighbor_indices)
    future_w = np.zeros((n, k), dtype=np.float32)

    for i in range(n):
        candidates = neighbor_indices[i]
        deltas = lat[candidates] - lat[i]
        positive_mask = deltas > 1e-6
        if np.any(positive_mask):
            pos_candidates = candidates[positive_mask]
            pos_deltas = deltas[positive_mask]
            order = np.argsort(pos_deltas)
            ordered_candidates = pos_candidates[order]
            ordered_deltas = pos_deltas[order]
        else:
            order = np.argsort(-deltas)
            ordered_candidates = candidates[order]
            ordered_deltas = np.maximum(deltas[order], 0.0)
        selected = ordered_candidates[:k]
        selected_deltas = ordered_deltas[:k]
        if selected.shape[0] < k:
            pad = np.full(k - selected.shape[0], candidates[0], dtype=candidates.dtype)
            pad_delta = np.zeros(k - selected_deltas.shape[0], dtype=selected_deltas.dtype)
            selected = np.concatenate([selected, pad], axis=0)
            selected_deltas = np.concatenate([selected_deltas, pad_delta], axis=0)
        weights = np.maximum(selected_deltas, 0.0).astype(np.float32)
        weight_sum = float(weights.sum())
        if weight_sum <= 1e-8:
            weights = np.full(k, 1.0 / max(1, k), dtype=np.float32)
        else:
            weights = weights / weight_sum
        future_idx[i] = selected.astype(np.int64)
        future_w[i] = weights
    return future_idx.astype(np.int64), future_w.astype(np.float32)


def _choose_plot_basis(
    adata,
    dataset_config: DatasetConfig,
) -> str:
    candidates: List[str] = []
    if dataset_config.embedding_key:
        candidates.append(dataset_config.embedding_key)
    if dataset_config.plot_basis:
        basis = dataset_config.plot_basis
        candidates.append(basis if basis.startswith("X_") else f"X_{basis.lower()}")
    candidates.extend(["X_umap", "X_tsne"])
    for key in candidates:
        if key and key in adata.obsm:
            if key.lower().startswith("x_"):
                return key[2:]
            return key
    return "umap"


def _plot_latent_time_scatter(
    adata: AnnData,
    dataset_name: str,
    method_name: str,
    basis: str,
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    title = f"{dataset_name} - {method_name} Latent Time"
    fig = scv.pl.scatter(
        adata,
        basis=basis,
        color="latent_time",
        color_map="viridis",
        title=title,
        show=False,
        return_fig=True,
    )
    if fig is None:  # pragma: no cover
        fig = plt.gcf()
    path = output_dir / f"{dataset_name}_{method_name}_latent_time.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_training_summary(
    history: dict,
    title: str,
    output_path: Path,
) -> Optional[Path]:
    """Render a compact training summary figure from scvi-style history dict.

    Plots available series among: ELBO (train/val), reconstruction, KL.
    """
    try:
        if history is None or not isinstance(history, dict):
            return None

        # Collect available series
        series_defs = [
            ("elbo_train", "ELBO (train)"),
            ("elbo_validation", "ELBO (val)"),
            ("train_loss_epoch", "Train Loss (epoch)"),
            ("reconstruction_loss_train", "Recon (train)"),
            ("reconstruction_loss_validation", "Recon (val)"),
            ("kl_local_train", "KL (train)"),
            ("kl_local_validation", "KL (val)"),
            ("velocity_laplacian", "Vel Laplacian"),
            ("velocity_angular", "Vel Angular"),
            ("gnn_continuity", "Continuity"),
        ]

        plotted = 0
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        for key, label in series_defs:
            if key in history and history[key] is not None:
                y = history[key]
                try:
                    # pandas Series or list
                    if hasattr(y, "values"):
                        y = y.values
                    y = list(y)
                except Exception:
                    continue
                if not y:
                    continue
                ax.plot(range(1, len(y) + 1), y, label=label, linewidth=1.5)
                plotted += 1
        if plotted == 0:
            plt.close(fig)
            return None
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Value")
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=8, ncol=2)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(output_path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        return output_path
    except Exception:  # pragma: no cover
        return None


def _should_use_gpu(config: TrainingConfig) -> bool:
    if not config.use_gpu:
        return False
    try:
        import torch

        return torch.cuda.is_available()
    except ModuleNotFoundError:
        return False


def _derive_tivelo_cluster_edges(
    adata,
    cluster_key: Optional[str],
    emb_key: Optional[str] = "X_umap",
) -> Optional[List[Tuple[str, str]]]:
    if cluster_key is None or cluster_key not in adata.obs:
        return None
    try:
        path_dict, _ = process_path(
            adata,
            group_key=cluster_key,
            emb_key=emb_key if emb_key in adata.obsm else None,
            njobs=32,
            start_mode="stochastic",
        )
        path_dict, _ = correct_path(path_dict, adata, cluster_key, tree_gene=None)
        edges = [(parent, child) for parent, children in path_dict.items() for child in children]
        return edges if edges else None
    except Exception as exc:  # pragma: no cover
        warnings.warn(f"Failed to derive TiVelo cluster edges: {exc}", RuntimeWarning)
        return None


def _visualize_results(results_df: pd.DataFrame, output_dir: Path):
    import matplotlib.pyplot as plt

    variants = [
        ("baseline", "Baseline"),
        ("latent", "Step1_PostHocLatent"),
        ("gnn", "Step2_GNNEncoder"),
        ("gnn_latent", "Step3_GNNPlusLatent"),
    ]

    metric_keys = [
        "gene_likelihood_mean",
        "velocity_norm_mean",
        "velocity_local_smoothness",
        "velocity_cosine_alignment",
    ]

    for _, row in results_df.iterrows():
        dataset = row["dataset"]
        dest_dir = Path(output_dir) / f"velovi_{dataset}"
        dest_dir.mkdir(parents=True, exist_ok=True)

        available_variants = [
            (prefix, label)
            for prefix, label in variants
            if all(f"{prefix}_{metric}" in row for metric in metric_keys)
        ]
        if len(available_variants) <= 1:
            continue

        for metric in metric_keys:
            values = []
            labels = []
            for prefix, label in available_variants:
                values.append(row[f"{prefix}_{metric}"])
                labels.append(label)

            plt.figure(figsize=(6, 4))
            plt.bar(
                labels,
                values,
                color=["#4c72b0", "#55a868", "#c44e52", "#8172b2"][: len(labels)],
            )
            plt.ylabel(metric.replace("_", " ").title())
            plt.title(f"{dataset}: {metric.replace('_', ' ').title()}")
            plt.xticks(rotation=25, ha="right")
            plt.tight_layout()
            metric_slug = metric.replace("_", "")
            filename = dest_dir / f"{dataset}_{metric_slug}_comparison.png"
            plt.savefig(filename, dpi=200)
            plt.close()


class VELOVIImprovementRunner:
    """Orchestrate baseline and enhanced VELOVI experiments across datasets."""

    def __init__(
        self,
        data_dir: Path,
        output_dir: Path,
        config: TrainingConfig | None = None,
        dataset_configs: Dict[str, DatasetConfig] | None = None,
    ):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.config = config or TrainingConfig()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        settings.dl_num_workers = self.config.num_workers
        settings.dl_pin_memory = True
        settings.dl_persistent_workers = self.config.num_workers > 0
        self.dataset_configs: Dict[str, DatasetConfig] = dataset_configs or VELOVI_DATASETS
        self.checkpoint_dir = Path(self.config.checkpoint_dir or (self.output_dir / "checkpoints"))
        if self.config.use_checkpoints:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._wandb = None
        if self.config.use_wandb:
            try:
                import wandb

                if self.config.wandb_api_key and "WANDB_API_KEY" not in os.environ:
                    os.environ["WANDB_API_KEY"] = self.config.wandb_api_key
                wandb.login()
                self._wandb = wandb
            except ModuleNotFoundError:
                warnings.warn(
                    "Weights & Biases is not installed; disabling wandb logging.",
                    RuntimeWarning,
                )
                self._wandb = None

    def run(self, dataset_names: List[str] | None = None) -> pd.DataFrame:
        if dataset_names is None:
            dataset_names = list(self.dataset_configs.keys())
        else:
            resolved: List[str] = []
            for name in dataset_names:
                canonical = resolve_dataset_name(name)
                if canonical not in self.dataset_configs:
                    available = ", ".join(sorted(self.dataset_configs.keys()))
                    raise KeyError(f"Unknown dataset '{name}'. Available keys: {available}")
                resolved.append(canonical)
            dataset_names = resolved

        records: List[Dict[str, float]] = []
        for name in dataset_names:
            records.append(self._run_single(name))

        results = pd.DataFrame(records)
        results.to_csv(self.output_dir / "velovi_improvements_summary.csv", index=False)

        if self.config.produce_plots:
            try:
                _visualize_results(results, self.output_dir)
            except ModuleNotFoundError as exc:
                raise ModuleNotFoundError(
                    "Result plotting requires matplotlib. Install it via `pip install matplotlib`."
                ) from exc
        return results

    def _checkpoint_stem(self, dataset_name: str, dataset_config: DatasetConfig) -> str:
        if dataset_config.preprocess_enabled:
            preprocess = dataset_config.preprocess or DEFAULT_PREPROCESS
            preprocess_sig = (
                f"top{preprocess.n_top_genes}_mc{preprocess.min_counts}_"
                f"mcs{preprocess.min_shared_counts}_pcs{preprocess.n_pcs}_nn{preprocess.n_neighbors}"
            )
        else:
            preprocess_sig = "nopre"
        encoder_tag = self.config.baseline_encoder
        model_sig = (
            f"nh{self.config.n_hidden}_nl{self.config.n_layers}_nz{self.config.n_latent}_"
            f"bs{self.config.batch_size}_ep{self.config.total_epochs}_enc{encoder_tag}"
        )
        return f"{dataset_name}_{preprocess_sig}_{model_sig}"

    def _project_velocities_to_embedding(
        self, velocities: Optional[np.ndarray], stream_embedding: StreamEmbeddingResult
    ) -> Optional[np.ndarray]:
        if velocities is None or stream_embedding.components is None:
            return None
        velocities = np.asarray(velocities, dtype=np.float32)
        projected = velocities @ stream_embedding.components.T
        if stream_embedding.projection is not None:
            projected = projected @ stream_embedding.projection
        return projected

    def _run_single(self, dataset_name: str) -> Dict[str, float]:
        dataset_config = self.dataset_configs[dataset_name]
        base_dataset_name = dataset_config.name
        preprocess_cfg = dataset_config.preprocess or DEFAULT_PREPROCESS
        preprocess_signature = (
            preprocess_cfg.signature() if dataset_config.preprocess_enabled else "nopre"
        )
        preprocess_label = (
            preprocess_cfg.display_name() if dataset_config.preprocess_enabled else "preprocess_off"
        )
        preprocess_active = dataset_config.preprocess_enabled and not self.config.skip_preprocess
        print(
            f"[VELOVI] Loading dataset {dataset_name} "
            f"(preprocess={'on' if preprocess_active else 'off'})"
        )
        metric_cluster_edges: Optional[List[Tuple[str, str]]] = (
            list(dataset_config.cluster_edges) if dataset_config.cluster_edges is not None else None
        )
        training_cluster_edges: Optional[List[Tuple[str, str]]] = metric_cluster_edges
        if not dataset_config.preprocess_enabled and not self.config.skip_preprocess:
            warnings.warn(
                f"Dataset {dataset_name} is marked as preprocessed; skipping additional preprocessing.",
                RuntimeWarning,
            )
        adata = load_dataset(
            self.data_dir,
            dataset_config,
            apply_preprocess=preprocess_active,
        )
        if self.config.skip_preprocess and dataset_config.preprocess_enabled:
            warnings.warn(
                "Skipping preprocessing per --skip-preprocess flag; ensure the dataset already "
                "contains normalized layers, PCA/neighbor graphs, and embeddings as needed.",
                RuntimeWarning,
            )
        runtime_records: List[Dict[str, float]] = []
        record: Dict[str, float] = {
            "dataset": base_dataset_name,
            "dataset_variant": dataset_name,
            "preprocess": preprocess_label,
            "preprocess_signature": preprocess_signature,
        }
        use_gpu = _should_use_gpu(self.config)
        accelerator = "gpu" if use_gpu else "cpu"
        devices = "auto"

        def record_runtime(method: str):
            class _RuntimeContext:
                def __enter__(self_nonlocal):
                    self_nonlocal.start_time = time.perf_counter()
                    self_nonlocal.start_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    return None

                def __exit__(self_nonlocal, exc_type, exc, tb):
                    end_time = time.perf_counter()
                    end_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    runtime_records.append(
                        {
                            "dataset": dataset_name,
                            "method": method,
                            "runtime_seconds": end_time - self_nonlocal.start_time,
                            "memory_MB": max(0.0, (end_mem - self_nonlocal.start_mem) / 1024.0),
                        }
                    )

            return _RuntimeContext()

        adata_reference = adata.copy()

        stream_embedding = compute_stream_embedding(adata_reference, self.config, dataset_config)
        cell_types, celltype_key = infer_cell_type_labels(adata_reference, dataset_config)
        plot_color_key = celltype_key or dataset_config.plot_color_key
        if plot_color_key and plot_color_key not in adata_reference.obs:
            fallback_keys = [
                dataset_config.plot_color_key,
                "clusters",
                "cell_type",
                "celltype",
                "leiden",
                "louvain",
            ]
            for cand in fallback_keys:
                if cand and cand in adata_reference.obs:
                    plot_color_key = cand
                    break
            else:
                plot_color_key = None

        record["metric_edge_source"] = "config" if metric_cluster_edges else "none"
        record["metric_edge_count"] = len(metric_cluster_edges) if metric_cluster_edges else 0
        training_edge_source = "config" if training_cluster_edges else "none"
        training_edge_count = len(training_cluster_edges) if training_cluster_edges else 0
        record["training_edge_source"] = training_edge_source
        record["training_edge_count"] = training_edge_count

        cluster_label_int: Optional[np.ndarray] = None
        cluster_label_mapping: Optional[Dict[str, int]] = None
        if plot_color_key and plot_color_key in adata_reference.obs:
            cluster_cat = pd.Categorical(adata_reference.obs[plot_color_key].astype(str))
            if np.all(cluster_cat.codes >= 0):
                cluster_label_int = cluster_cat.codes.astype(np.int64)
                cluster_label_mapping = {label: idx for idx, label in enumerate(cluster_cat.categories)}

        def _convert_edges_to_int(
            edges: Optional[List[Tuple[str, str]]],
        ) -> Tuple[Optional[List[Tuple[int, int]]], Optional[Dict[int, Dict[int, float]]]]:
            if edges is None or cluster_label_mapping is None:
                return None, None
            converted = [
                (cluster_label_mapping[src], cluster_label_mapping[dst])
                for src, dst in edges
                if src in cluster_label_mapping and dst in cluster_label_mapping
            ]
            return (converted if converted else None, None)

        stream_velocity_variants: Dict[str, np.ndarray] = {}
        method_specific_anndatas: Dict[str, AnnData] = {}
        figure_paths: Dict[str, Tuple[Path, bool]] = {}
        variant_metric_rows: List[Tuple[str, Dict[str, float]]] = []
        wandb_scalar_logs: Dict[str, float] = {}
        wandb_run = start_wandb_run(self.config, dataset_name)

        indices, weights = add_stream_graph(adata_reference, dataset_config, key_prefix="velovi_gnn")
        adata.obsm["velovi_gnn_indices"] = indices
        adata.obsm["velovi_gnn_weights"] = weights
        alignment_vectors = prepare_alignment_vectors(adata_reference, indices, dataset_config)

        use_transformer_encoder = self.config.baseline_encoder == "transformer"
        baseline_cls = VELOVITransformerEncoder if use_transformer_encoder else VELOVI
        if use_transformer_encoder:
            baseline_cls.setup_anndata(
                adata,
                spliced_layer=dataset_config.spliced_layer,
                unspliced_layer=dataset_config.unspliced_layer,
                neighbor_index_key="velovi_gnn_indices",
                neighbor_weight_key="velovi_gnn_weights",
            )
        else:
            VELOVI.setup_anndata(
                adata,
                spliced_layer=dataset_config.spliced_layer,
                unspliced_layer=dataset_config.unspliced_layer,
            )
        baseline_model = None
        baseline_ckpt = None
        baseline_loaded_from_ckpt = False
        if self.config.use_checkpoints:
            stem = self._checkpoint_stem(dataset_name, dataset_config)
            baseline_ckpt = self.checkpoint_dir / f"{stem}_baseline"
            if baseline_ckpt.exists():
                if self.config.load_pretrained:
                    baseline_model = baseline_cls.load(baseline_ckpt, adata=adata)
                    baseline_loaded_from_ckpt = True
                else:
                    warnings.warn(
                        f"Found baseline checkpoint at {baseline_ckpt} but load_pretrained is disabled. "
                        "Retraining baseline model from scratch.",
                        RuntimeWarning,
                    )
        if baseline_model is None:
            print(f"[VELOVI] Training baseline VELOVI for {dataset_name}")
            baseline_kwargs = dict(
                n_hidden=self.config.n_hidden,
                n_latent=self.config.n_latent,
                n_layers=self.config.n_layers,
                dropout_rate=self.config.dropout_rate,
            )
            if use_transformer_encoder:
                baseline_kwargs.update(
                    transformer_hidden_dim=self.config.transformer_encoder_hidden_dim,
                    transformer_layers=self.config.transformer_encoder_layers,
                    transformer_heads=self.config.transformer_encoder_heads,
                    transformer_dropout=self.config.transformer_encoder_dropout,
                    transformer_max_neighbors=self.config.transformer_encoder_max_neighbors,
                    transformer_neighbor_reg_weight=self.config.transformer_encoder_neighbor_weight,
                )
            baseline_model = baseline_cls(
                adata,
                **baseline_kwargs,
            )
            with record_runtime("baseline"):
                baseline_model.train(
                    max_epochs=self.config.warmup_epochs,
                    batch_size=self.config.batch_size,
                    early_stopping=False,
                    accelerator=accelerator,
                    devices=devices,
                )
                remaining_epochs = max(0, self.config.total_epochs - self.config.warmup_epochs)
                if remaining_epochs > 0:
                    baseline_model.train(
                        max_epochs=remaining_epochs,
                        batch_size=self.config.batch_size,
                        early_stopping=False,
                        accelerator=accelerator,
                        devices=devices,
                    )
            if self.config.use_checkpoints and baseline_ckpt is not None:
                baseline_model.save(baseline_ckpt, overwrite=True)
            history = getattr(baseline_model, "history_", None)
            if history is None:
                history = getattr(baseline_model, "history", None)
            log_training_history(wandb_run, history, "baseline/train")
            # Baseline training summary figure
            try:
                summary_path = _plot_training_summary(
                    history,
                    title=f"{dataset_name} – Baseline Training Summary",
                    output_path=Path(self.output_dir)
                    / f"velovi_{dataset_name}"
                    / "figures"
                    / f"{dataset_name}_baseline_train_summary.png",
                )
                if summary_path is not None:
                    figure_paths["diagnostic/train_baseline_summary"] = (summary_path, False)
            except Exception:  # pragma: no cover
                pass
        if baseline_loaded_from_ckpt and wandb_run is not None:
            wandb_run.log({"baseline/checkpoint_loaded": 1})

        baseline_eval_adata: Optional[AnnData] = None
        try:
            baseline_eval_adata = add_velovi_outputs_to_adata(adata, baseline_model)
            scv.tl.velocity_graph(
                baseline_eval_adata,
                vkey="velocity",
                n_jobs=max(1, self.config.scvelo_dynamics_n_jobs),
                mode_neighbors="distances",
                approx=False,
                show_progress_bar=False,
            )
            method_specific_anndatas["baseline"] = baseline_eval_adata
        except Exception as exc:
            warnings.warn(f"Failed to build scVelo-compatible baseline AnnData: {exc}", RuntimeWarning)

        latent_time_values: Optional[np.ndarray] = None
        if self.config.gnn_continuity_weight > 0.0:
            try:
                latent_time_result = baseline_model.get_latent_time()
                if hasattr(latent_time_result, "values"):
                    latent_time_values = np.asarray(latent_time_result.values, dtype=np.float32).reshape(-1)
                else:
                    latent_time_values = np.asarray(latent_time_result, dtype=np.float32).reshape(-1)
            except Exception as exc:
                latent_time_values = None
                warnings.warn(
                    f"Failed to compute latent time for GNN continuity loss: {exc}",
                    RuntimeWarning,
                )

        latent = None
        latent_graph = None
        latent_needed = (
            self.config.enable_latent_smoothing
            or self.config.gnn_neighbor_source == "latent"
            or self.config.enable_transformer_refinement
        )
        if latent_needed:
            latent = baseline_model.get_latent_representation()
        if latent is not None and (self.config.enable_latent_smoothing or self.config.enable_transformer_refinement):
            latent_graph = LatentEmbeddingGraphBuilder(self.config.latent_graph).build(latent)
        latent_metric_indices: Optional[np.ndarray] = None
        latent_metric_weights: Optional[np.ndarray] = None
        latent_alignment_vectors: Optional[np.ndarray] = None
        latent_stream_embedding = compute_latent_stream_embedding(latent, self.config)
        if latent is not None:
            try:
                latent_metric_indices, latent_metric_weights = construct_feature_graph(
                    latent,
                    n_neighbors=max(1, self.config.latent_metric_n_neighbors),
                    metric="euclidean",
                )
                # For cosine alignment in latent context, compare gene-space velocities
                # to gene-space expression deltas, but use latent neighbor indices.
                # This keeps vector spaces consistent while changing neighborhood definition.
                latent_alignment_vectors = prepare_alignment_vectors(
                    adata_reference,
                    latent_metric_indices,
                    dataset_config,
                    data_matrix=None,
                )
            except Exception as exc:
                latent_metric_indices = None
                latent_metric_weights = None
                latent_alignment_vectors = None
                warnings.warn(
                    f"Failed to construct latent metric graph: {exc}",
                    RuntimeWarning,
                )
        metric_contexts: List[MetricContext] = [
            MetricContext(
                name="expression",
                neighbor_indices=indices,
                alignment_vectors=alignment_vectors,
            )
        ]
        if latent_metric_indices is not None and latent_alignment_vectors is not None:
            metric_contexts.append(
                MetricContext(
                    name="latent",
                    neighbor_indices=latent_metric_indices,
                    alignment_vectors=latent_alignment_vectors,
                )
            )
        baseline_latent_velocity = None
        raw_baseline_velocity = baseline_model.get_velocity(return_numpy=True)
        baseline_velocity = raw_baseline_velocity.copy()
        baseline_raw_metrics = summarize_velocity_metrics(
            velocities=raw_baseline_velocity,
            neighbor_indices=indices,
            gene_likelihood_mean=float(
                baseline_model.get_gene_likelihood(return_mean=True, return_numpy=True).mean()
            ),
            alignment_vectors=alignment_vectors,
        )
        wandb_scalar_logs.update({f"baseline_raw/{k}": v for k, v in result_to_dict(baseline_raw_metrics).items()})
        gene_likelihood_mean = float(baseline_raw_metrics.gene_likelihood_mean)

        def compute_metrics_for_variant(
            label: str,
            velocities: np.ndarray,
            likelihood_override: Optional[float] = None,
        ) -> Dict[str, Dict[str, float]]:
            metrics: Dict[str, Dict[str, float]] = {}
            gene_ll = gene_likelihood_mean if likelihood_override is None else likelihood_override
            for context in metric_contexts:
                ctx_label = label if context.name == "expression" else f"{label}_{context.name}"
                try:
                    result = summarize_velocity_metrics(
                        velocities=velocities,
                        neighbor_indices=context.neighbor_indices,
                        gene_likelihood_mean=gene_ll,
                        alignment_vectors=context.alignment_vectors,
                    )
                    metrics[ctx_label] = result_to_dict(result)
                except Exception as exc:
                    warnings.warn(
                        f"Failed to compute metrics for {ctx_label}: {exc}",
                        RuntimeWarning,
                    )
            return metrics

        def register_metrics(metrics_by_label: Dict[str, Dict[str, float]]) -> None:
            for ctx_label, metrics in metrics_by_label.items():
                record.update({f"{ctx_label}_{k}": v for k, v in metrics.items()})
                wandb_scalar_logs.update({f"{ctx_label}/{k}": v for k, v in metrics.items()})
                variant_metric_rows.append((ctx_label, metrics))

        def register_guided_variant(label: str, velocities: Optional[np.ndarray]) -> Optional[np.ndarray]:
            if apply_guidance is None or velocities is None:
                return None
            guided = apply_guidance(velocities)
            guided_label = f"{label}_tivelo"
            guided_metrics = compute_metrics_for_variant(guided_label, guided)
            register_metrics(guided_metrics)
            stream_velocity_variants[guided_label] = guided.astype(np.float32)
            comparison_pairs[guided_label] = (velocities, guided)
            return guided

        tivelo_supervised = None
        tivelo_weights = None
        guidance_confidence = None
        comparison_pairs: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        apply_guidance = None
        if self.config.transformer_use_tivelo and raw_baseline_velocity.shape[0] == adata.n_obs:
            print(f"[VELOVI] Running TIVelo guidance for {dataset_name}")
            try:
                tivelo_output_dir = Path(self.output_dir) / f"velovi_{dataset_name}" / "tivelo_raw"
                tivelo_output_dir.mkdir(parents=True, exist_ok=True)

                adata_tivelo = adata.copy()
                cluster_key_for_tivelo = (
                    dataset_config.group_key or celltype_key or dataset_config.plot_color_key
                )
                if cluster_key_for_tivelo is None and "clusters" in adata_tivelo.obs:
                    cluster_key_for_tivelo = "clusters"
                embedding_key_for_tivelo = dataset_config.embedding_key
                if embedding_key_for_tivelo not in adata_tivelo.obsm:
                    embedding_key_for_tivelo = "X_umap" if "X_umap" in adata_tivelo.obsm else None

                tivelo_adata = run_tivelo(
                    adata_tivelo,
                    group_key=cluster_key_for_tivelo,
                    emb_key=embedding_key_for_tivelo,
                    res=self.config.transformer_tivelo_resolution,
                    data_name=dataset_name,
                    save_folder=str(tivelo_output_dir),
                    njobs=32,
                    start_mode="stochastic",
                    rev_stat="mean",
                    tree_gene=self.config.tivelo_tree_gene,
                    t1=self.config.transformer_tivelo_threshold,
                    t2=self.config.transformer_tivelo_threshold_trans,
                    show_fig=self.config.tivelo_show_fig,
                    filter_genes=self.config.tivelo_filter_genes,
                    constrain=self.config.tivelo_constrain,
                    loss_fun=self.config.tivelo_loss_fun,
                    only_s=self.config.tivelo_only_spliced,
                    alpha_1=self.config.tivelo_alpha1,
                    alpha_2=self.config.tivelo_alpha2,
                    batch_size=self.config.tivelo_batch_size,
                    n_epochs=self.config.tivelo_epochs,
                    adjust_DTI=self.config.tivelo_adjust_dti,
                    show_DTI=self.config.tivelo_show_dti,
                    velocity_key="velocity",
                    measure_performance=False,
                    device="cuda" if use_gpu else "cpu",
                )

                guidance_velocity = np.asarray(tivelo_adata.layers.get("velocity"), dtype=np.float32)
                if guidance_velocity.shape != baseline_velocity.shape or np.allclose(guidance_velocity, 0.0):
                    warnings.warn(
                        "TIVelo guidance produced invalid velocity field; skipping guidance.",
                        RuntimeWarning,
                    )
                else:
                    tivelo_supervised = guidance_velocity
                    tivelo_weights = np.ones(guidance_velocity.shape[0], dtype=np.float32)
                    guidance_confidence = tivelo_weights[:, None]
                    prior_strength = np.clip(self.config.tivelo_prior_strength, 0.0, 1.0)

                    adata.layers["tivelo_velocity"] = guidance_velocity.astype(np.float32, copy=False)

                    cluster_key_metrics = cluster_key_for_tivelo
                    if cluster_key_metrics is None:
                        cluster_key_metrics = "clusters"
                        tivelo_adata.obs[cluster_key_metrics] = (
                            cell_types if cell_types is not None else np.arange(tivelo_adata.n_obs)
                        )

                    tivelo_metrics_summary: Dict[str, float] = {}
                    child_dict = tivelo_adata.uns.get("child_dict")
                    tivelo_cluster_edges: Optional[List[Tuple[str, str]]] = None
                    if child_dict:
                        tivelo_cluster_edges = [
                            (parent, child) for parent, children in child_dict.items() for child in children
                        ]
                        training_cluster_edges = tivelo_cluster_edges
                        new_edge_int, _ = _convert_edges_to_int(training_cluster_edges)
                        if new_edge_int is not None:
                            transformer_cluster_edges_int = new_edge_int

                    try:
                        if cluster_key_metrics in tivelo_adata.obs:
                            if tivelo_cluster_edges:
                                _, cbdir = tivelo_metrics.cross_boundary_correctness(
                                    tivelo_adata,
                                    cluster_key=cluster_key_metrics,
                                    velocity_key="velocity",
                                    cluster_edges=tivelo_cluster_edges,
                                )
                                _, cbdir2 = tivelo_metrics.cross_boundary_correctness2(
                                    tivelo_adata,
                                    cluster_key=cluster_key_metrics,
                                    velocity_key="velocity",
                                    cluster_edges=tivelo_cluster_edges,
                                )
                                _, transprob = tivelo_metrics.cross_boundary_scvelo_probs(
                                    tivelo_adata,
                                    cluster_key=cluster_key_metrics,
                                    cluster_edges=tivelo_cluster_edges,
                                    trans_g_key="velocity_graph",
                                )
                            else:
                                cbdir = cbdir2 = transprob = float("nan")
                            _, icvcoh = tivelo_metrics.inner_cluster_coh(
                                tivelo_adata, cluster_key=cluster_key_metrics, velocity_key="velocity"
                            )
                            _, icvcoh2 = tivelo_metrics.inner_cluster_coh2(
                                tivelo_adata,
                                cluster_key=cluster_key_metrics,
                                velocity_key="velocity",
                                x_emb="X_umap",
                            )
                            velocoh = tivelo_metrics.velo_coh(
                                tivelo_adata, velocity_key="velocity", trans_g_key="velocity_graph"
                            )
                            tivelo_metrics_summary = {
                                "CBDir": float(cbdir) if not np.isnan(cbdir) else 0.0,
                                "CBDir2": float(cbdir2) if not np.isnan(cbdir2) else 0.0,
                                "TransProb": float(transprob) if not np.isnan(transprob) else 0.0,
                                "ICVCoh": float(icvcoh),
                                "ICVCoh2": float(icvcoh2),
                                "VeloCoh": float(velocoh),
                            }
                    except Exception as exc:  # pragma: no cover
                        warnings.warn(f"Failed to compute TIVelo metrics: {exc}", RuntimeWarning)

                    def apply_guidance(velocities: np.ndarray, strength: Optional[float] = None) -> np.ndarray:
                        coeff = prior_strength if strength is None else np.clip(strength, 0.0, 1.0)
                        alpha = np.clip(guidance_confidence * coeff, 0.0, 1.0)
                        return (1.0 - alpha) * velocities + alpha * tivelo_supervised

                    if tivelo_metrics_summary:
                        record.update({f"tivelo_{k.lower()}": v for k, v in tivelo_metrics_summary.items()})
                        wandb_scalar_logs.update({f"tivelo_guidance/{k}": v for k, v in tivelo_metrics_summary.items()})
                    wandb_scalar_logs["tivelo_guidance/active_fraction"] = float(np.mean(tivelo_weights))
            except Exception as exc:  # pragma: no cover
                warnings.warn(f"TIVelo guidance failed: {exc}", RuntimeWarning)

        if tivelo_supervised is None:
            if wandb_run is not None:
                wandb_scalar_logs["tivelo_guidance/active_fraction"] = 0.0

        if self.config.enable_latent_smoothing and latent_graph is not None:
            with record_runtime("baseline_latent"):
                baseline_latent_velocity = smooth_velocities_with_graph(baseline_velocity, latent_graph)

        cell_type_ids_array: Optional[np.ndarray] = None
        cell_type_velocity_means: Optional[np.ndarray] = None
        if cell_types is not None and baseline_velocity.shape[0] == len(cell_types):
            unique_labels, inverse_indices = np.unique(cell_types, return_inverse=True)
            if unique_labels.size > 0:
                cell_type_ids_array = inverse_indices.astype(np.int64)
                n_types = unique_labels.size
                velocity_dim = baseline_velocity.shape[1]
                type_means = np.zeros((n_types, velocity_dim), dtype=np.float32)
                for type_id in range(n_types):
                    mask = cell_type_ids_array == type_id
                    if np.any(mask):
                        type_means[type_id] = baseline_velocity[mask].mean(axis=0)
                cell_type_velocity_means = type_means
        elif cell_types is not None and baseline_velocity.shape[0] != len(cell_types):
            warnings.warn(
                "Cell type labels are present but do not match the number of cells; skipping cell-type refinement.",
                RuntimeWarning,
            )

        baseline_metrics_by_context = compute_metrics_for_variant("baseline", baseline_velocity)
        baseline_metrics_raw = baseline_metrics_by_context.get("baseline", {})
        if tivelo_supervised is not None:
            improvement = {
                key: baseline_metrics_raw.get(key, 0.0) - result_to_dict(baseline_raw_metrics).get(key, 0.0)
                for key in baseline_metrics_raw.keys()
            }
            wandb_scalar_logs.update({f"tivelo_improvement/{k}": v for k, v in improvement.items()})
            record.update({f"baseline_raw_{k}": v for k, v in result_to_dict(baseline_raw_metrics).items()})
        stream_velocity_variants["baseline"] = baseline_velocity.astype(np.float32)
        register_metrics(baseline_metrics_by_context)
        register_guided_variant("baseline", baseline_velocity)

        if self.config.use_tivelo_cluster_edges:
            emb_basis = dataset_config.plot_basis
            emb_key = f"X_{emb_basis}" if emb_basis else "X_umap"
            paga_input = adata_reference.copy()
            paga_input.layers["velocity"] = baseline_velocity.astype(np.float32, copy=False)
            derived_edges = _derive_tivelo_cluster_edges(
                paga_input,
                cluster_key=plot_color_key,
                emb_key=emb_key if emb_key in paga_input.obsm else None,
            )
            if derived_edges:
                training_cluster_edges = derived_edges
                training_edge_source = "tivelo_paga"
                training_edge_count = len(derived_edges)
                record["training_edge_source"] = training_edge_source
                record["training_edge_count"] = training_edge_count

        transformer_cluster_edges_int, _ = _convert_edges_to_int(training_cluster_edges)

        dynamic_velocity = None
        adata_dynamic_result = None
        if self.config.enable_scvelo_dynamic:
            print(f"[VELOVI] Running scVelo dynamical baseline for {dataset_name}")
            previous_n_jobs = None
            try:
                adata_dynamic = adata.copy()
                n_jobs = max(1, self.config.scvelo_dynamics_n_jobs)
                previous_n_jobs = getattr(scv.settings, "N_jobs", None)
                scv.settings.N_jobs = n_jobs
                with record_runtime("scvelo_dynamic"):
                    scv.tl.recover_dynamics(adata_dynamic, n_jobs=n_jobs)
                    scv.tl.velocity(adata_dynamic, mode="dynamical", n_jobs=32)
                    scv.tl.velocity_graph(adata_dynamic, vkey="velocity", n_jobs=32)
                    dynamic_velocity = np.asarray(adata_dynamic.layers["velocity"], dtype=np.float32)
            except ModuleNotFoundError:
                warnings.warn(
                    "scvelo is required to compute the dynamical baseline; skipping this variant.",
                    RuntimeWarning,
                )
            except Exception as exc:
                warnings.warn(
                    f"Failed to compute scvelo dynamical velocities: {exc}",
                    RuntimeWarning,
                )
            finally:
                if "scv" in locals() and previous_n_jobs is not None:
                    scv.settings.N_jobs = previous_n_jobs
            if dynamic_velocity is not None:
                if np.allclose(dynamic_velocity, 0.0) or np.isnan(dynamic_velocity).all():
                    warnings.warn(
                        "scvelo dynamical returned empty velocities; skipping metrics for scvelo_dynamic.",
                        RuntimeWarning,
                    )
                    dynamic_velocity = None
            if dynamic_velocity is not None and dynamic_velocity.shape == baseline_velocity.shape:
                scvelo_metrics = compute_metrics_for_variant(
                    "scvelo_dynamic",
                    dynamic_velocity,
                    likelihood_override=float("nan"),
                )
                register_metrics(scvelo_metrics)
                stream_velocity_variants["scvelo_dynamic"] = dynamic_velocity.astype(np.float32)
                adata_dynamic_result = adata_dynamic
                register_guided_variant("scvelo_dynamic", dynamic_velocity)

        transformer_velocity = None
        transformer_velocity_latent = None
        if self.config.enable_transformer_refinement:
            print(f"[VELOVI] Training transformer refiner for {dataset_name}")
            if latent is None:
                latent = baseline_model.get_latent_representation()
            transformer_config = TransformerConfig(
                n_layers=self.config.transformer_layers,
                n_heads=self.config.transformer_heads,
                hidden_dim=self.config.transformer_hidden_dim,
                dropout=self.config.transformer_dropout,
                batch_size=self.config.transformer_batch_size,
                epochs=self.config.transformer_epochs,
                learning_rate=self.config.transformer_learning_rate,
                weight_alignment=self.config.transformer_weight_alignment,
                weight_supervised=self.config.transformer_weight_supervised,
                weight_smooth=self.config.transformer_weight_smooth,
                weight_smooth_same=self.config.transformer_weight_smooth_same,
                weight_boundary_align=self.config.transformer_weight_boundary_align,
                weight_boundary_contrast=self.config.transformer_weight_boundary_contrast,
                weight_direction=self.config.transformer_weight_direction,
                weight_celltype=self.config.transformer_weight_celltype,
                weight_celltype_dir=self.config.transformer_weight_celltype_dir,
                weight_celltype_mag=self.config.transformer_weight_celltype_mag,
                max_neighbors=self.config.transformer_max_neighbors,
                celltype_penalty=self.config.transformer_celltype_penalty,
                aux_cluster_loss_weight=self.config.transformer_aux_cluster_loss_weight,
                neighbor_max_distance=self.config.transformer_neighbor_max_distance,
                residual_to_baseline=self.config.transformer_residual_to_baseline,
                device="cuda" if use_gpu else "cpu",
            )
            transformer_state_path = self.output_dir / f"velovi_{dataset_name}_transformer.pt"
            with record_runtime("baseline_transformer"):
                transformer_velocity = refine_velocities_with_transformer(
                    latent=latent,
                    embedding=stream_embedding.embedding if stream_embedding.embedding is not None else latent,
                    baseline_velocity=baseline_velocity,
                    neighbor_indices=indices,
                    velocity_components=stream_embedding.components,
                    projection=stream_embedding.projection,
                    config=transformer_config,
                    wandb_run=wandb_run,
                    wandb_prefix="transformer/train",
                    cell_type_ids=cell_type_ids_array,
                    type_means=cell_type_velocity_means,
                    cluster_labels=cluster_label_int,
                    cluster_edge_list=transformer_cluster_edges_int,
                    alignment_vectors=alignment_vectors,
                    supervised_target=tivelo_supervised,
                    supervised_weight=tivelo_weights,
                    save_path=str(transformer_state_path),
                )
            transformer_metrics = compute_metrics_for_variant("transformer", transformer_velocity)
            register_metrics(transformer_metrics)
            stream_velocity_variants["transformer"] = transformer_velocity.astype(np.float32)
            register_guided_variant("transformer", transformer_velocity)
            if latent_graph is not None:
                with record_runtime("baseline_transformer_latent"):
                    transformer_velocity_latent = smooth_velocities_with_graph(transformer_velocity, latent_graph)
                transformer_latent_metrics = compute_metrics_for_variant(
                    "transformer_latent",
                    transformer_velocity_latent,
                )
                register_metrics(transformer_latent_metrics)
                stream_velocity_variants["transformer_latent"] = transformer_velocity_latent.astype(np.float32)
                register_guided_variant("transformer_latent", transformer_velocity_latent)

        if self.config.enable_latent_smoothing and baseline_latent_velocity is not None:
            print(f"[VELOVI] Applying latent smoothing for {dataset_name}")
            latent_metrics = compute_metrics_for_variant("latent", baseline_latent_velocity)
            register_metrics(latent_metrics)
            stream_velocity_variants["latent"] = baseline_latent_velocity.astype(np.float32)
            register_guided_variant("latent", baseline_latent_velocity)

        adata_gnn = adata_reference.copy()
        latent_indices = None
        latent_weights = None
        if self.config.gnn_neighbor_source in ("latent", "both") and latent is not None:
            latent_indices, latent_weights = construct_feature_graph(
                latent,
                n_neighbors=15,
                metric="euclidean",
            )
            if self.config.gnn_neighbor_source == "latent":
                # Use latent graph as the primary neighbor source
                adata_gnn.obsm["velovi_gnn_indices"] = latent_indices
                adata_gnn.obsm["velovi_gnn_weights"] = latent_weights
            else:
                # both: keep expression graph as primary, add latent graph as secondary
                adata_gnn.obsm["velovi_gnn_indices_latent"] = latent_indices
                adata_gnn.obsm["velovi_gnn_weights_latent"] = latent_weights
        # Optionally construct a boundary-aware secondary neighbor graph to guide GNN aggregation
        boundary_secondary = None
        if transformer_cluster_edges_int is not None and cluster_label_int is not None:
            try:
                boundary_secondary = _build_boundary_secondary_graph(
                    cluster_labels=cluster_label_int,
                    edges_int=transformer_cluster_edges_int,
                    primary_indices=indices,
                    primary_weights=weights,
                )
            except Exception as exc:  # pragma: no cover
                warnings.warn(f"Failed to construct boundary secondary graph: {exc}", RuntimeWarning)

        # If a boundary-aware graph is available, prefer it as the secondary neighbor source
        if self.config.gnn_neighbor_source == "both" and boundary_secondary is not None:
            b_idx, b_w = boundary_secondary
            adata_gnn.obsm["velovi_gnn_indices_latent"] = b_idx
            adata_gnn.obsm["velovi_gnn_weights_latent"] = b_w
            print(
                f"[VELOVI] Using boundary-aware secondary neighbor graph for GNN (replacing latent secondary)"
            )

        future_indices = None
        future_weights = None
        if self.config.gnn_continuity_weight > 0.0 and latent_time_values is not None:
            try:
                future_indices, future_weights = _build_future_neighbor_graph(latent_time_values, indices)
                adata_gnn.obsm["velovi_future_indices"] = future_indices
                adata_gnn.obsm["velovi_future_weights"] = future_weights
            except Exception as exc:  # pragma: no cover
                future_indices = None
                future_weights = None
                warnings.warn(f"Failed to construct future neighbor graph: {exc}", RuntimeWarning)

        VELOVIWithGNN.setup_anndata(
            adata_gnn,
            spliced_layer=dataset_config.spliced_layer,
            unspliced_layer=dataset_config.unspliced_layer,
            neighbor_index_key="velovi_gnn_indices",
            neighbor_weight_key="velovi_gnn_weights",
            neighbor_index_key_latent=(
                "velovi_gnn_indices_latent" if self.config.gnn_neighbor_source == "both" else None
            ),
            neighbor_weight_key_latent=(
                "velovi_gnn_weights_latent" if self.config.gnn_neighbor_source == "both" else None
            ),
            future_index_key="velovi_future_indices" if future_indices is not None else None,
            future_weight_key="velovi_future_weights" if future_weights is not None else None,
        )
        gnn_velocity = None
        gnn_gene_likelihood = None
        gnn_loaded_from_ckpt = False
        gnn_eval_adata: Optional[AnnData] = None
        if self.config.enable_gnn:
            print(f"[VELOVI] Training VELOVI+GNN for {dataset_name}")
            stem = self._checkpoint_stem(dataset_name, dataset_config)
            gnn_sig = (
                f"gh{self.config.gnn_hidden_dim}_gd{self.config.gnn_dropout_rate}_"
                f"att{int(self.config.gnn_use_attention)}_gate{int(self.config.gnn_use_gate)}_"
                f"res{int(self.config.gnn_use_residual)}_diff{int(self.config.gnn_use_differences)}"
            )
            if self.config.use_checkpoints:
                gnn_ckpt = self.checkpoint_dir / f"{stem}_{gnn_sig}_gnn"
            else:
                gnn_ckpt = None
            gnn_model = None
            if gnn_ckpt is not None and gnn_ckpt.exists():
                if self.config.load_pretrained:
                    gnn_model = VELOVIWithGNN.load(gnn_ckpt, adata=adata_gnn)
                    gnn_loaded_from_ckpt = True
                else:
                    warnings.warn(
                        f"Found GNN checkpoint at {gnn_ckpt} but load_pretrained is disabled. "
                        "Retraining GNN model from scratch.",
                        RuntimeWarning,
                    )
            if gnn_model is None:
                gnn_model = VELOVIWithGNN(
                    adata_gnn,
                    n_hidden=self.config.n_hidden,
                    n_latent=self.config.n_latent,
                    n_layers=self.config.n_layers,
                    dropout_rate=self.config.dropout_rate,
                    gnn_hidden_dim=self.config.gnn_hidden_dim,
                    gnn_dropout_rate=self.config.gnn_dropout_rate,
                    gnn_use_attention=self.config.gnn_use_attention,
                    gnn_use_gate=self.config.gnn_use_gate,
                    gnn_use_residual=self.config.gnn_use_residual,
                    gnn_use_differences=self.config.gnn_use_differences,
                    velocity_laplacian_weight=self.config.velocity_laplacian_weight,
                    velocity_angle_weight=self.config.velocity_angle_weight,
                    velocity_angle_eps=self.config.velocity_angle_eps,
                    gnn_continuity_weight=self.config.gnn_continuity_weight,
                    gnn_continuity_horizon=self.config.gnn_continuity_horizon,
                )
                with record_runtime("baseline_gnn"):
                    gnn_train_epochs = (
                        self.config.gnn_epochs if self.config.gnn_epochs is not None else self.config.total_epochs
                    )
                    gnn_batch_size = (
                        self.config.gnn_batch_size
                        if self.config.gnn_batch_size is not None
                        else self.config.batch_size
                    )
                    gnn_model.train(
                        max_epochs=gnn_train_epochs,
                        batch_size=gnn_batch_size,
                        early_stopping=True,
                        accelerator=accelerator,
                        devices=devices,
                    )
                if self.config.use_checkpoints and gnn_ckpt is not None:
                    gnn_model.save(gnn_ckpt, overwrite=True)
                # Always drop a reusable checkpoint under output_dir
                gnn_persist_path = self.output_dir / f"velovi_{dataset_name}_gnn"
                gnn_model.save(gnn_persist_path, overwrite=True)
                history = getattr(gnn_model, "history_", None)
                if history is None:
                    history = getattr(gnn_model, "history", None)
                log_training_history(wandb_run, history, "gnn/train")
                # GNN training summary figure
                try:
                    gnn_summary_path = _plot_training_summary(
                        history,
                        title=f"{dataset_name} – GNN Training Summary",
                        output_path=Path(self.output_dir)
                        / f"velovi_{dataset_name}"
                        / "figures"
                        / f"{dataset_name}_gnn_train_summary.png",
                    )
                    if gnn_summary_path is not None:
                        figure_paths["diagnostic/train_gnn_summary"] = (gnn_summary_path, False)
                except Exception:  # pragma: no cover
                    pass
            if gnn_loaded_from_ckpt and wandb_run is not None:
                wandb_run.log({"gnn/checkpoint_loaded": 1})

            if gnn_model is not None:
                try:
                    gnn_eval_adata = add_velovi_outputs_to_adata(adata_gnn, gnn_model)
                    scv.tl.velocity_graph(
                        gnn_eval_adata,
                        vkey="velocity",
                        n_jobs=max(1, self.config.scvelo_dynamics_n_jobs),
                        mode_neighbors="distances",
                        approx=False,
                        show_progress_bar=False,
                    )
                    method_specific_anndatas["baseline_gnn"] = gnn_eval_adata
                except Exception as exc:
                    warnings.warn(f"Failed to build scVelo-compatible GNN AnnData: {exc}", RuntimeWarning)

            gnn_velocity = gnn_model.get_velocity(return_numpy=True)
            gnn_gene_likelihood = float(
                gnn_model.get_gene_likelihood(return_mean=True, return_numpy=True).mean()
            )
            gnn_metrics = compute_metrics_for_variant(
                "gnn",
                gnn_velocity,
                likelihood_override=gnn_gene_likelihood,
            )
            register_metrics(gnn_metrics)
            stream_velocity_variants["gnn"] = gnn_velocity.astype(np.float32)
            register_guided_variant("gnn", gnn_velocity)

        if self.config.enable_gnn and self.config.enable_gnn_latent_smoothing and gnn_velocity is not None:
            print(f"[VELOVI] Applying latent smoothing to GNN outputs for {dataset_name}")
            gnn_latent = gnn_model.get_latent_representation()
            gnn_latent_graph = LatentEmbeddingGraphBuilder(self.config.latent_graph).build(gnn_latent)
            with record_runtime("baseline_gnn_latent"):
                gnn_smoothed_velocity = smooth_velocities_with_graph(gnn_velocity, gnn_latent_graph)
            gnn_latent_metrics = compute_metrics_for_variant(
                "gnn_latent",
                gnn_smoothed_velocity,
                likelihood_override=gnn_gene_likelihood,
            )
            register_metrics(gnn_latent_metrics)
            stream_velocity_variants["gnn_latent"] = gnn_smoothed_velocity.astype(np.float32)
            register_guided_variant("gnn_latent", gnn_smoothed_velocity)

        benchmark_results: Optional[Dict[str, benchmark.MethodBenchmarkResult]] = None
        benchmark_figures: Dict[str, Path] = {}
        benchmark_tables: Dict[str, Path] = {}
        advanced_velocities: Dict[str, np.ndarray] = {}
        advanced_embeddings: Dict[str, np.ndarray] = {}
        if dynamic_velocity is not None:
            advanced_velocities["scvelo_dynamic"] = dynamic_velocity.astype(np.float32)
        advanced_velocities["baseline"] = baseline_velocity.astype(np.float32)
        if baseline_latent_velocity is not None:
            advanced_velocities["baseline_latent"] = baseline_latent_velocity.astype(np.float32)
        if gnn_velocity is not None:
            advanced_velocities["baseline_gnn"] = gnn_velocity.astype(np.float32)
        if "gnn_latent" in stream_velocity_variants:
            advanced_velocities["baseline_gnn_latent"] = stream_velocity_variants["gnn_latent"]
        if transformer_velocity is not None:
            advanced_velocities["baseline_transformer"] = transformer_velocity.astype(np.float32)
        if transformer_velocity_latent is not None:
            advanced_velocities["baseline_transformer_latent"] = transformer_velocity_latent.astype(np.float32)
        if tivelo_supervised is not None:
            advanced_velocities["tivelo"] = tivelo_supervised.astype(np.float32)
        if apply_guidance is not None:
            guided_key_map = {
                "baseline": "baseline",
                "baseline_latent": "latent",
                "baseline_gnn": "gnn",
                "baseline_gnn_latent": "gnn_latent",
                "baseline_transformer": "transformer",
                "baseline_transformer_latent": "transformer_latent",
                "scvelo_dynamic": "scvelo_dynamic",
            }
            for adv_key, metric_label in guided_key_map.items():
                stream_key = f"{metric_label}_tivelo"
                if adv_key in advanced_velocities and stream_key in stream_velocity_variants:
                    advanced_velocities[f"{adv_key}_tivelo"] = stream_velocity_variants[stream_key]

        benchmark_dir = Path(self.output_dir) / f"velovi_{dataset_name}"
        gene_names = adata_reference.var_names.to_numpy()
        subset_col = None
        if "group" in adata_reference.obs:
            subset_col = "group"
        elif "subset_type" in adata_reference.obs:
            subset_col = "subset_type"
        celltype_col = None
        if "cell.labels2" in adata_reference.obs:
            celltype_col = "cell.labels2"
        elif celltype_key and celltype_key in adata_reference.obs:
            celltype_col = celltype_key
        subset_values = (
            adata_reference.obs[subset_col].astype(str).to_numpy()
            if subset_col is not None
            else np.full(adata_reference.n_obs, "all", dtype=object)
        )
        celltype_values = (
            adata_reference.obs[celltype_col].astype(str).to_numpy()
            if celltype_col is not None
            else np.full(adata_reference.n_obs, "all", dtype=object)
        )
        group_df = pd.DataFrame(
            {
                "subset_type": subset_values,
                "celltype": celltype_values,
            }
        )
        group_indices = list(group_df.groupby(["subset_type", "celltype"], sort=False).indices.items())
        tables_dir = benchmark_dir / "tables"
        tables_dir.mkdir(parents=True, exist_ok=True)
        for method_name, velocities in advanced_velocities.items():
            if (
                velocities.ndim != 2
                or velocities.shape[0] != adata_reference.n_obs
                or velocities.shape[1] != len(gene_names)
            ):
                continue
            method_key = method_name.replace("/", "_")
            output_path = tables_dir / f"{dataset_name}_{method_key}_gene_velocity.csv"
            header_written = False
            for (subset_value, celltype_value), indices in group_indices:
                if len(indices) == 0:
                    continue
                idx = np.asarray(indices, dtype=int)
                mean_abs = np.nanmean(np.abs(velocities[idx]), axis=0)
                df = pd.DataFrame(
                    {
                        "subset_type": subset_value,
                        "celltype": celltype_value,
                        "gene_name": gene_names,
                        "velocity": mean_abs,
                    }
                )
                df.to_csv(output_path, mode="a", index=False, header=not header_written)
                header_written = True
            if header_written:
                benchmark_tables[f"{method_key}_gene_velocity"] = output_path

        if stream_embedding.embedding is not None:
            embedding_coords = stream_embedding.embedding
        else:
            embedding_coords = adata_reference.obsm.get("X_umap", None)
        if embedding_coords is not None:
            for name, vel in advanced_velocities.items():
                projected = self._project_velocities_to_embedding(vel, stream_embedding)
                if projected is not None:
                    advanced_embeddings[name] = projected.astype(np.float32)

        benchmark_clusters = (
            cell_types if cell_types is not None else np.full(adata_reference.n_obs, "all", dtype="<U8")
        )
        method_anndatas = dict(method_specific_anndatas)
        if adata_dynamic_result is not None:
            method_anndatas["scvelo_dynamic"] = adata_dynamic_result

        benchmark_results = benchmark.benchmark_methods(
            adata=adata_reference,
            velocities=advanced_velocities,
            neighbor_indices=indices,
            clusters=np.asarray(benchmark_clusters),
            output_dir=benchmark_dir,
            method_embedding=advanced_embeddings if advanced_embeddings else None,
            embedding_coordinates=embedding_coords,
            fucci_key=dataset_config.fucci_key,
            cluster_edges=metric_cluster_edges,
            cluster_key=plot_color_key,
            method_anndatas=method_anndatas if method_anndatas else None,
            cell_cycle_rad_key=dataset_config.cell_cycle_rad_key,
        )

        stream_panel_path = benchmark.plot_stream_grid(
            dataset_id=dataset_name,
            adata=adata_reference,
            velocities=advanced_velocities,
            cluster_key=plot_color_key,
            output_path=benchmark_dir / "figures" / f"{dataset_name}_stream_panel.png",
        )
        dynamic_tivelo_path = None
        if (
            apply_guidance is not None
            and "scvelo_dynamic" in stream_velocity_variants
            and "scvelo_dynamic_tivelo" in stream_velocity_variants
        ):
            dynamic_pairs = {
                "scvelo_dynamic": (
                    stream_velocity_variants["scvelo_dynamic"],
                    stream_velocity_variants["scvelo_dynamic_tivelo"],
                )
            }
            dynamic_tivelo_path = benchmark.plot_comparison_panel(
                dataset_id=f"{dataset_name}_dynamic_vs_tivelo",
                adata=adata_reference,
                pairs=dynamic_pairs,
                cluster_key=plot_color_key,
                output_path=benchmark_dir / "figures" / f"{dataset_name}_dynamic_vs_tivelo.png",
            )

        if comparison_pairs:
            tivelo_comparison_path = benchmark.plot_comparison_panel(
                dataset_id=dataset_name,
                adata=adata_reference,
                pairs=comparison_pairs,
                cluster_key=plot_color_key,
                output_path=benchmark_dir / "figures" / f"{dataset_name}_tivelo_comparison.png",
            )
        else:
            tivelo_comparison_path = None
        performance_path = benchmark.plot_performance_bars(
            dataset_id=dataset_name,
            results=benchmark_results,
            output_path=benchmark_dir / "figures" / f"{dataset_name}_performance.pdf",
        )
        fucci_path = benchmark.plot_fucci_violin(
            dataset_id=dataset_name,
            results=benchmark_results,
            output_path=benchmark_dir / "figures" / f"{dataset_name}_fucci.pdf",
        )
        gene_methods = {
            key: advanced_velocities[key]
            for key in ("baseline", "scvelo_dynamic", "baseline_transformer", "tivelo")
            if key in advanced_velocities
        }
        gene_paths = benchmark.plot_gene_level_panels(
            dataset_id=dataset_name,
            adata=adata_reference,
            genes=adata_reference.var_names[:3],
            methods=gene_methods,
            cluster_key=plot_color_key,
            output_dir=benchmark_dir / "figures",
        )
        summary_df, edge_df = benchmark.build_tables(dataset_name, benchmark_results)
        if not summary_df.empty and "baseline" in summary_df["method"].values:
            baseline_row = summary_df.loc[summary_df["method"] == "baseline"].iloc[0]
            for col in ("cbdir", "cbdir2", "icvcoh", "velocoh"):
                improvement_col = f"{col}_delta"
                summary_df[improvement_col] = summary_df[col] - baseline_row[col]
        summary_path = benchmark_dir / "tables" / f"{dataset_name}_advanced_summary.csv"
        edge_path = benchmark_dir / "tables" / f"{dataset_name}_advanced_edges.csv"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(summary_path, index=False)
        edge_df.to_csv(edge_path, index=False)
        benchmark_figures["stream_panel"] = stream_panel_path
        benchmark_figures["performance"] = performance_path
        if fucci_path is not None:
            benchmark_figures["fucci"] = fucci_path
        for idx, path in enumerate(gene_paths):
            benchmark_figures[f"gene_panel_{idx}"] = path
        if dynamic_tivelo_path is not None:
            benchmark_figures["dynamic_vs_tivelo"] = dynamic_tivelo_path
        if tivelo_comparison_path is not None:
            benchmark_figures["tivelo_comparison"] = tivelo_comparison_path
        benchmark_tables["summary"] = summary_path
        benchmark_tables["edges"] = edge_path
        runtime_df = pd.DataFrame(runtime_records)
        runtime_path = benchmark_dir / "tables" / f"{dataset_name}_runtime.csv"
        runtime_df.to_csv(runtime_path, index=False)
        benchmark_tables["runtime"] = runtime_path
        training_runtime_df = None
        training_method_labels = {
            "baseline": "veloVI base",
            "baseline_transformer": "veloVI transformer refinement",
        }
        training_agg: Dict[str, Dict[str, float]] = {}
        for entry in runtime_records:
            method = entry.get("method")
            if method not in training_method_labels:
                continue
            bucket = training_agg.setdefault(
                method,
                {"runtime_seconds": 0.0, "memory_MB": 0.0},
            )
            bucket["runtime_seconds"] += float(entry.get("runtime_seconds", 0.0))
            bucket["memory_MB"] = max(bucket["memory_MB"], float(entry.get("memory_MB", 0.0)))
        if training_agg:
            training_rows = []
            for method, stats in training_agg.items():
                training_rows.append(
                    {
                        "dataset": dataset_name,
                        "stage": training_method_labels[method],
                        "runtime_seconds": stats["runtime_seconds"],
                        "memory_MB": stats["memory_MB"],
                    }
                )
            training_runtime_df = pd.DataFrame(training_rows)
            training_runtime_path = benchmark_dir / "tables" / f"{dataset_name}_training_runtime.csv"
            training_runtime_df.to_csv(training_runtime_path, index=False)
            benchmark_tables["training_runtime"] = training_runtime_path
            baseline_time = training_agg.get("baseline", {}).get("runtime_seconds")
            transformer_time = training_agg.get("baseline_transformer", {}).get("runtime_seconds")
            if baseline_time is not None:
                wandb_scalar_logs["runtime/baseline_seconds"] = baseline_time
            if transformer_time is not None:
                wandb_scalar_logs["runtime/transformer_seconds"] = transformer_time
            if baseline_time and transformer_time:
                wandb_scalar_logs["runtime/transformer_over_baseline"] = transformer_time / baseline_time
        baseline_benchmark = benchmark_results.get("baseline")
        for method_name, res in benchmark_results.items():
            wandb_scalar_logs[f"benchmark/{method_name}/CBDir"] = res.cbdir
            wandb_scalar_logs[f"benchmark/{method_name}/CBDir2"] = res.cbdir2
            wandb_scalar_logs[f"benchmark/{method_name}/TransCosine"] = res.trans_cosine
            wandb_scalar_logs[f"benchmark/{method_name}/TransProb"] = res.trans_probability
            wandb_scalar_logs[f"benchmark/{method_name}/ICVCoh"] = res.icvcoh
            wandb_scalar_logs[f"benchmark/{method_name}/ICVCoh2"] = res.icvcoh2
            wandb_scalar_logs[f"benchmark/{method_name}/VeloCoh"] = res.velocoh
            if res.cell_cycle_velocity_accuracy is not None:
                wandb_scalar_logs[
                    f"benchmark/{method_name}/cell_cycle_accuracy"
                ] = res.cell_cycle_velocity_accuracy
            if baseline_benchmark is not None and method_name != "baseline":
                wandb_scalar_logs[f"improvement_vs_baseline/{method_name}/CBDir"] = res.cbdir - baseline_benchmark.cbdir
                wandb_scalar_logs[f"improvement_vs_baseline/{method_name}/ICVCoh"] = res.icvcoh - baseline_benchmark.icvcoh
                wandb_scalar_logs[f"improvement_vs_baseline/{method_name}/VeloCoh"] = res.velocoh - baseline_benchmark.velocoh

        if tivelo_supervised is not None:
            stream_velocity_variants["tivelo"] = tivelo_supervised.astype(np.float32)

        if self.config.produce_plots and stream_velocity_variants:
            print(f"[VELOVI] Generating streamline plots for {dataset_name}")
            figure_paths.update(
                generate_variant_streamplots(
                    dataset_name=dataset_name,
                    adata=adata_reference,
                    dataset_config=dataset_config,
                    stream_embedding=stream_embedding,
                    variant_velocities=stream_velocity_variants,
                    neighbor_indices=indices,
                    neighbor_weights=weights,
                    cell_types=cell_types,
                    color_key=plot_color_key,
                    output_dir=self.output_dir,
                    save_locally=self.config.save_figures_locally,
                )
            )
            if (
                latent_stream_embedding.embedding is not None
                and latent_metric_indices is not None
                and latent_metric_weights is not None
            ):
                print(f"[VELOVI] Generating latent streamline plots for {dataset_name}")
                figure_paths.update(
                    generate_variant_streamplots(
                        dataset_name=dataset_name,
                        adata=adata_reference,
                        dataset_config=dataset_config,
                        stream_embedding=latent_stream_embedding,
                        variant_velocities=stream_velocity_variants,
                        neighbor_indices=latent_metric_indices,
                        neighbor_weights=latent_metric_weights,
                        cell_types=cell_types,
                        color_key=plot_color_key,
                        output_dir=self.output_dir,
                        context_label="latent",
                        save_locally=self.config.save_figures_locally,
                    )
                )
            print(f"[VELOVI] Generating PAGA/FDL diagnostics for {dataset_name}")
            figure_paths.update(
                generate_graph_diagnostics(
                    dataset_name=dataset_name,
                    adata=adata_reference,
                    dataset_config=dataset_config,
                    output_dir=self.output_dir,
                    color_key=plot_color_key,
                    save_locally=self.config.save_figures_locally,
                )
            )

            latent_fig_dir = Path(self.output_dir) / f"velovi_{dataset_name}" / "figures"
            for variant_name in stream_velocity_variants.keys():
                source_adata = method_specific_anndatas.get(variant_name)
                if source_adata is None:
                    if "gnn" in variant_name and "baseline_gnn" in method_specific_anndatas:
                        source_adata = method_specific_anndatas.get("baseline_gnn")
                    else:
                        source_adata = method_specific_anndatas.get("baseline")
                if source_adata is None or "latent_time" not in source_adata.obs:
                    continue
                try:
                    basis = _choose_plot_basis(source_adata, dataset_config)
                    latent_path = _plot_latent_time_scatter(
                        adata=source_adata,
                        dataset_name=dataset_name,
                        method_name=variant_name,
                        basis=basis,
                        output_dir=latent_fig_dir,
                    )
                    figure_paths[f"latent_time/{variant_name}"] = (latent_path, False)
                except Exception as exc:  # pragma: no cover
                    warnings.warn(f"Failed to create latent time scatter for {variant_name}: {exc}", RuntimeWarning)
        if wandb_run is not None:
            if variant_metric_rows and self._wandb is not None:
                metric_keys = sorted({key for _, metrics in variant_metric_rows for key in metrics})
                columns = ["dataset", "variant"] + metric_keys
                data = []
                for variant_name, metrics in variant_metric_rows:
                    row = [dataset_name, variant_name] + [metrics.get(key) for key in metric_keys]
                    data.append(row)
                if data:
                    metrics_table = self._wandb.Table(columns=columns, data=data)
                    wandb_run.log({f"{dataset_name}/metrics": metrics_table})
            if wandb_scalar_logs and self._wandb is not None:
                wandb_run.log(wandb_scalar_logs)
            if benchmark_results is not None and self._wandb is not None:
                summary_table = self._wandb.Table(dataframe=summary_df)
                edges_table = self._wandb.Table(dataframe=edge_df)
                runtime_table = self._wandb.Table(dataframe=runtime_df)
                wandb_run.log({f"{dataset_name}/benchmark_summary": summary_table})
                wandb_run.log({f"{dataset_name}/benchmark_edges": edges_table})
                wandb_run.log({f"{dataset_name}/runtime": runtime_table})
            if training_runtime_df is not None and self._wandb is not None:
                training_table = self._wandb.Table(dataframe=training_runtime_df)
                wandb_run.log({f"{dataset_name}/training_runtime": training_table})
            if figure_paths:
                for variant_key, fig_info in figure_paths.items():
                    fig_path, ephemeral = fig_info
                    try:
                        path_obj = Path(fig_path)
                        exists = path_obj.exists()
                        print(
                            f"[VELOVI][DIAGNOSTIC] Logging figure {variant_key} from {fig_path} (exists={exists})"
                        )
                        if not exists:
                            warnings.warn(
                                f"Figure path missing before wandb upload: {fig_path}",
                                RuntimeWarning,
                            )
                        # Use clearer namespaces for images in W&B
                        if variant_key in {"paga", "fdl"} or variant_key.startswith("diagnostic/"):
                            key = f"diagnostic/{variant_key.split('/', 1)[-1]}"
                        else:
                            key = f"streamline/{variant_key}"
                        wandb_run.log({key: self._wandb.Image(str(path_obj))})
                    except Exception as exc:  # pragma: no cover
                        warnings.warn(f"Failed to log figure {fig_path} to wandb: {exc}", RuntimeWarning)
                    finally:
                        if ephemeral:
                            with contextlib.suppress(Exception):
                                path_obj.unlink()
                            parent = path_obj.parent
                            if parent.exists() and parent.name.startswith(("velovi_stream_", "velovi_diag_")):
                                with contextlib.suppress(OSError):
                                    parent.rmdir()
            if benchmark_figures and self._wandb is not None:
                for name, fig_path in benchmark_figures.items():
                    try:
                        wandb_run.log({f"benchmark/{name}": self._wandb.Image(str(fig_path))})
                    except Exception as exc:  # pragma: no cover
                        warnings.warn(f"Failed to log benchmark figure {fig_path}: {exc}", RuntimeWarning)
            wandb_run.finish()
        return record


def main():
    parser = argparse.ArgumentParser(description="Evaluate VELOVI improvement ideas across datasets.")
    parser.add_argument("data_dir", type=Path, help="Directory containing VELOVI benchmark datasets (h5ad files).")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("velovi_improvement_results"),
        help="Directory where experiment summaries will be written.",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Optional subset of dataset names to run (defaults to all).",
    )
    parser.add_argument("--warmup-epochs", type=int, default=100)
    parser.add_argument("--total-epochs", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--latent-dim", type=int, default=10)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument(
        "--baseline-encoder",
        choices=["mlp", "transformer"],
        default="mlp",
        help="Backbone for the baseline encoder stage.",
    )
    parser.add_argument(
        "--transformer-encoder-hidden-dim",
        type=int,
        default=256,
        help="Hidden size for the transformer baseline encoder.",
    )
    parser.add_argument(
        "--transformer-encoder-layers",
        type=int,
        default=2,
        help="Number of transformer layers for the baseline encoder.",
    )
    parser.add_argument(
        "--transformer-encoder-heads",
        type=int,
        default=4,
        help="Attention heads for the transformer baseline encoder.",
    )
    parser.add_argument(
        "--transformer-encoder-dropout",
        type=float,
        default=0.1,
        help="Dropout applied inside the transformer baseline encoder.",
    )
    parser.add_argument(
        "--transformer-encoder-max-neighbors",
        type=int,
        default=None,
        help="Optional cap on neighbors per token sequence for the baseline transformer encoder.",
    )
    parser.add_argument(
        "--transformer-encoder-neighbor-weight",
        type=float,
        default=0.0,
        help="Strength of the neighbor-alignment penalty for the baseline transformer encoder.",
    )
    parser.add_argument("--gnn-hidden-dim", type=int, default=128)
    parser.add_argument("--gnn-dropout", type=float, default=0.1)
    parser.add_argument(
        "--gnn-epochs",
        type=int,
        default=None,
        help="Override total training epochs for the GNN encoder (defaults to total epochs).",
    )
    parser.add_argument(
        "--gnn-batch-size",
        type=int,
        default=None,
        help="Override batch size for GNN training (defaults to baseline batch size).",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--disable-latent-smoothing",
        action="store_true",
        help="Skip latent embedding post-hoc smoothing.",
    )
    parser.add_argument(
        "--disable-gnn",
        action="store_true",
        help="Skip GNN-augmented encoder training.",
    )
    parser.add_argument(
        "--enable-gnn-latent",
        action="store_true",
        help="After GNN training, run latent smoothing on GNN velocities.",
    )
    parser.add_argument(
        "--plot-results",
        action="store_true",
        help="Generate comparison plots for each dataset.",
    )
    parser.add_argument(
        "--disable-scvelo-dynamic",
        action="store_true",
        help="Skip running the scVelo dynamical model baseline.",
    )
    parser.add_argument(
        "--scvelo-n-jobs",
        type=int,
        default=32,
        help="Number of parallel jobs to use for scVelo dynamical baseline computations.",
    )
    parser.add_argument(
        "--gnn-neighbor-source",
        choices=["expression", "latent", "both"],
        default="both",
        help="Source of neighbors for GNN encoder (expression, latent, or both).",
    )
    parser.add_argument(
        "--gnn-attention",
        action="store_true",
        help="Use attention weights when aggregating neighbor messages in the GNN encoder.",
    )
    parser.add_argument(
        "--gnn-gate",
        action="store_true",
        help="Apply layer norm and learnable gating to the neighbor message before concatenation.",
    )
    parser.add_argument(
        "--gnn-continuity-weight",
        type=float,
        default=0.0,
        help="Weight for the continuity loss that encourages forward extrapolation consistency.",
    )
    parser.add_argument(
        "--gnn-continuity-horizon",
        type=float,
        default=1.0,
        help="Time horizon (in latent units) used when projecting velocities for continuity loss.",
    )
    parser.add_argument(
        "--velocity-laplacian-weight",
        type=float,
        default=0.0,
        help="Weight for Laplacian smoothness penalty on velocities during training.",
    )
    parser.add_argument(
        "--velocity-angle-weight",
        type=float,
        default=0.0,
        help="Weight for angular consistency penalty on neighboring velocities.",
    )
    parser.add_argument(
        "--velocity-angle-eps",
        type=float,
        default=1e-6,
        help="Stabilizing epsilon used in the angular velocity penalty.",
    )
    parser.add_argument(
        "--stream-embed",
        choices=["pca", "umap"],
        default="pca",
        help="Embedding method to use for velocity streamline plots.",
    )
    parser.add_argument(
        "--stream-embed-pca-components",
        type=int,
        default=8,
        help="Number of PCA components to retain when constructing UMAP stream embeddings.",
    )
    parser.add_argument(
        "--disable-stream-standardize",
        action="store_true",
        help="Skip feature standardization before computing the streamline embedding.",
    )
    parser.add_argument("--stream-umap-neighbors", type=int, default=30)
    parser.add_argument("--stream-umap-min-dist", type=float, default=0.3)
    parser.add_argument("--stream-umap-spread", type=float, default=1.0)
    parser.add_argument(
        "--latent-metric-n-neighbors",
        type=int,
        default=15,
        help="Number of latent-space neighbors used for metric evaluation and plotting.",
    )
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="Force training/evaluation on CPU even if CUDA is available.",
    )
    parser.add_argument(
        "--skip-preprocess",
        action="store_true",
        help="Assume datasets are already preprocessed; skip scVelo preprocessing stage.",
    )
    parser.add_argument(
        "--use-tivelo-cluster-edges",
        action="store_true",
        help="Derive cluster edges from TiVelo's PAGA path (for training only; metrics keep static edges).",
    )
    parser.add_argument(
        "--use-paga-for-refinements",
        action="store_true",
        help="Run PAGA after baseline warmup to derive dynamic edges for refinement modules.",
    )
    parser.add_argument(
        "--enable-transformer-refinement",
        action="store_true",
        help="Train a transformer-based velocity refiner on top of baseline velocities.",
    )
    parser.add_argument("--transformer-epochs", type=int, default=10)
    parser.add_argument("--transformer-hidden-dim", type=int, default=128)
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--transformer-heads", type=int, default=4)
    parser.add_argument("--transformer-dropout", type=float, default=0.1)
    parser.add_argument("--transformer-batch-size", type=int, default=128)
    parser.add_argument("--transformer-learning-rate", type=float, default=1e-3)
    parser.add_argument("--transformer-weight-smooth", type=float, default=0.1)
    parser.add_argument("--transformer-weight-direction", type=float, default=0.1)
    parser.add_argument("--transformer-weight-celltype", type=float, default=0.0)
    parser.add_argument("--transformer-weight-celltype-dir", type=float, default=0.0)
    parser.add_argument("--transformer-weight-celltype-mag", type=float, default=0.0)
    parser.add_argument(
        "--transformer-weight-alignment",
        type=float,
        default=1.0,
        help="Weight applied to the alignment loss between predicted velocities and expression offsets.",
    )
    parser.add_argument(
        "--transformer-weight-supervised",
        type=float,
        default=0.0,
        help="Weight for supervised regression towards external velocity targets (e.g., TIVelo).",
    )
    parser.add_argument(
        "--transformer-weight-smooth-same",
        type=float,
        default=0.0,
        help="Weight for same-cluster smoothing term in the transformer refiner.",
    )
    parser.add_argument(
        "--transformer-weight-boundary-align",
        type=float,
        default=0.0,
        help="Weight for aligning embedding velocities with boundary directions.",
    )
    parser.add_argument(
        "--transformer-weight-boundary-contrast",
        type=float,
        default=0.0,
        help="Weight for discouraging alignment with invalid boundary directions.",
    )
    parser.add_argument(
        "--transformer-celltype-penalty",
        choices=["cosine", "mse", "both"],
        default="cosine",
    )
    parser.add_argument("--transformer-aux-cluster-loss-weight", type=float, default=0.0)
    parser.add_argument("--transformer-neighbor-max-distance", type=float, default=None)
    parser.add_argument(
        "--transformer-max-neighbors",
        type=int,
        default=None,
        help="Optional cap on neighbors used in transformer refinement sequences.",
    )
    parser.add_argument(
        "--transformer-no-residual",
        action="store_true",
        help="Predict absolute velocities instead of residuals relative to baseline.",
    )
    parser.add_argument(
        "--transformer-use-tivelo",
        action="store_true",
        help="Use TIVelo's directed trajectory inference to supervise the transformer.",
    )
    parser.add_argument(
        "--transformer-tivelo-resolution",
        type=float,
        default=0.6,
        help="Leiden resolution used when clustering cells for TIVelo guidance (if labels are missing).",
    )
    parser.add_argument(
        "--transformer-tivelo-threshold",
        type=float,
        default=0.1,
        help="Edge weight threshold applied when building the directed cluster tree in TIVelo guidance.",
    )
    parser.add_argument(
        "--transformer-tivelo-threshold-trans",
        type=float,
        default=1.0,
        help="Transition threshold used when building the directed tree in TIVelo guidance.",
    )
    parser.add_argument(
        "--tivelo-prior-strength",
        type=float,
        default=0.4,
        help="Blending factor applied when fusing TIVelo guidance into baseline velocities.",
    )
    parser.add_argument(
        "--tivelo-loss-fun",
        type=str,
        default="mse",
        choices={"mse", "cos"},
        help="Loss function used by the TIVelo model.",
    )
    parser.add_argument(
        "--tivelo-only-spliced",
        action="store_true",
        help="Optimise only spliced velocities in the TIVelo model.",
    )
    parser.add_argument(
        "--tivelo-no-constrain",
        dest="tivelo_constrain",
        action="store_false",
        help="Disable the velocity constraint term during TIVelo optimisation.",
    )
    parser.set_defaults(tivelo_constrain=True)
    parser.add_argument("--tivelo-alpha1", type=float, default=1.0, help="Weight for spliced loss in TIVelo.")
    parser.add_argument("--tivelo-alpha2", type=float, default=0.1, help="Weight for cosine regulariser in TIVelo.")
    parser.add_argument("--tivelo-batch-size", type=int, default=1024, help="Batch size for TIVelo training.")
    parser.add_argument("--tivelo-epochs", type=int, default=100, help="Epochs for TIVelo training.")
    parser.add_argument(
        "--tivelo-no-filter-genes",
        dest="tivelo_filter_genes",
        action="store_false",
        help="Disable gene filtering before TIVelo training.",
    )
    parser.set_defaults(tivelo_filter_genes=True)
    parser.add_argument(
        "--tivelo-show-fig",
        action="store_true",
        help="Display TIVelo diagnostic figures during execution.",
    )
    parser.add_argument(
        "--tivelo-show-dti",
        action="store_true",
        help="Save and (optionally) display TIVelo DTI plots.",
    )
    parser.add_argument(
        "--tivelo-adjust-dti",
        action="store_true",
        help="Run the TIVelo DTI adjustment sweep.",
    )
    parser.add_argument(
        "--tivelo-tree-gene",
        type=str,
        default="Cplx2",
        help="Marker gene passed to TIVelo's path correction step.",
    )
    parser.add_argument(
        "--disable-checkpoints",
        action="store_true",
        help="Do not save/load checkpoints for baseline and GNN models.",
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Load existing checkpoints instead of retraining when available.",
    )
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        help="Log metrics and streamline images to Weights & Biases.",
    )
    parser.add_argument("--wandb-project", type=str, default="velovi_improvements")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-api-key", type=str, default=None)
    parser.add_argument("--wandb-run-group", type=str, default=None)
    parser.add_argument(
        "--disable-local-figures",
        action="store_true",
        help="Skip persisting plots/diagnostics on disk (still logs to W&B if enabled).",
    )
    parser.add_argument(
        "--disable-gnn-residual",
        action="store_true",
        help="Skip residual fusion of the central cell features in the GNN encoder.",
    )
    parser.add_argument(
        "--disable-gnn-differences",
        action="store_true",
        help="Skip aggregating neighbor difference messages in the GNN encoder.",
    )

    args = parser.parse_args()

    config = TrainingConfig(
        warmup_epochs=args.warmup_epochs,
        total_epochs=args.total_epochs,
        batch_size=args.batch_size,
        n_latent=args.latent_dim,
        n_hidden=args.hidden_dim,
        baseline_encoder=args.baseline_encoder,
        transformer_encoder_hidden_dim=args.transformer_encoder_hidden_dim,
        transformer_encoder_layers=args.transformer_encoder_layers,
        transformer_encoder_heads=args.transformer_encoder_heads,
        transformer_encoder_dropout=args.transformer_encoder_dropout,
        transformer_encoder_max_neighbors=args.transformer_encoder_max_neighbors,
        transformer_encoder_neighbor_weight=args.transformer_encoder_neighbor_weight,
        gnn_hidden_dim=args.gnn_hidden_dim,
        gnn_dropout_rate=args.gnn_dropout,
        gnn_epochs=args.gnn_epochs,
        gnn_batch_size=args.gnn_batch_size,
        gnn_continuity_weight=args.gnn_continuity_weight,
        gnn_continuity_horizon=args.gnn_continuity_horizon,
        num_workers=args.num_workers,
        enable_latent_smoothing=not args.disable_latent_smoothing,
        enable_gnn=not args.disable_gnn,
        enable_gnn_latent_smoothing=args.enable_gnn_latent,
        produce_plots=args.plot_results,
        gnn_neighbor_source=args.gnn_neighbor_source,
        gnn_use_attention=args.gnn_attention,
        gnn_use_gate=args.gnn_gate,
        gnn_use_residual=not args.disable_gnn_residual,
        gnn_use_differences=not args.disable_gnn_differences,
        velocity_laplacian_weight=args.velocity_laplacian_weight,
        velocity_angle_weight=args.velocity_angle_weight,
        velocity_angle_eps=args.velocity_angle_eps,
        stream_embed_method=args.stream_embed,
        stream_embed_pca_components=args.stream_embed_pca_components,
        stream_embed_standardize=not args.disable_stream_standardize,
        stream_umap_neighbors=args.stream_umap_neighbors,
        stream_umap_min_dist=args.stream_umap_min_dist,
        stream_umap_spread=args.stream_umap_spread,
        enable_transformer_refinement=args.enable_transformer_refinement,
        enable_scvelo_dynamic=not args.disable_scvelo_dynamic,
        transformer_epochs=args.transformer_epochs,
        transformer_hidden_dim=args.transformer_hidden_dim,
        transformer_layers=args.transformer_layers,
        transformer_heads=args.transformer_heads,
        transformer_dropout=args.transformer_dropout,
        transformer_batch_size=args.transformer_batch_size,
        transformer_learning_rate=args.transformer_learning_rate,
        transformer_weight_smooth=args.transformer_weight_smooth,
        transformer_weight_direction=args.transformer_weight_direction,
        transformer_weight_celltype=args.transformer_weight_celltype,
        transformer_weight_celltype_dir=args.transformer_weight_celltype_dir,
        transformer_weight_celltype_mag=args.transformer_weight_celltype_mag,
        transformer_weight_alignment=args.transformer_weight_alignment,
        transformer_weight_supervised=args.transformer_weight_supervised,
        transformer_weight_smooth_same=args.transformer_weight_smooth_same,
        transformer_weight_boundary_align=args.transformer_weight_boundary_align,
        transformer_weight_boundary_contrast=args.transformer_weight_boundary_contrast,
        transformer_celltype_penalty=args.transformer_celltype_penalty,
        transformer_aux_cluster_loss_weight=args.transformer_aux_cluster_loss_weight,
        transformer_neighbor_max_distance=args.transformer_neighbor_max_distance,
        transformer_max_neighbors=args.transformer_max_neighbors,
        transformer_residual_to_baseline=not args.transformer_no_residual,
        transformer_use_tivelo=args.transformer_use_tivelo,
        transformer_tivelo_resolution=args.transformer_tivelo_resolution,
        transformer_tivelo_threshold=args.transformer_tivelo_threshold,
        transformer_tivelo_threshold_trans=args.transformer_tivelo_threshold_trans,
        tivelo_prior_strength=args.tivelo_prior_strength,
        tivelo_loss_fun=args.tivelo_loss_fun,
        tivelo_only_spliced=args.tivelo_only_spliced,
        tivelo_constrain=args.tivelo_constrain,
        tivelo_alpha1=args.tivelo_alpha1,
        tivelo_alpha2=args.tivelo_alpha2,
        tivelo_batch_size=args.tivelo_batch_size,
        tivelo_epochs=args.tivelo_epochs,
        tivelo_filter_genes=args.tivelo_filter_genes,
        tivelo_show_fig=args.tivelo_show_fig,
        tivelo_show_dti=args.tivelo_show_dti,
        tivelo_adjust_dti=args.tivelo_adjust_dti,
        tivelo_tree_gene=args.tivelo_tree_gene,
        load_pretrained=args.pretrained,
        use_checkpoints=not args.disable_checkpoints,
        checkpoint_dir=str(args.checkpoint_dir) if args.checkpoint_dir is not None else None,
        scvelo_dynamics_n_jobs=args.scvelo_n_jobs,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_api_key=args.wandb_api_key,
        wandb_run_group=args.wandb_run_group,
        latent_metric_n_neighbors=args.latent_metric_n_neighbors,
        use_gpu=not args.cpu_only,
        skip_preprocess=args.skip_preprocess,
        save_figures_locally=not args.disable_local_figures,
        use_tivelo_cluster_edges=args.use_tivelo_cluster_edges,
        use_paga_for_refinements=args.use_paga_for_refinements,
    )

    runner = VELOVIImprovementRunner(args.data_dir, args.output_dir, config)
    results = runner.run(dataset_names=args.datasets)
    print(results)
if __name__ == "__main__":
    main()
