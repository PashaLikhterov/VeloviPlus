from __future__ import annotations

import warnings
import tempfile
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scvelo as scv
import scanpy as sc
from scipy.sparse import csr_matrix

from .config import StreamEmbeddingResult
from .datasets import DatasetConfig


def _select_embedding_for_plot(adata, dataset_config: DatasetConfig) -> Tuple[Optional[str], Optional[np.ndarray]]:
    def _normalize_basis(value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        basis = value.strip()
        if basis.lower().startswith("x_"):
            basis = basis[2:]
        return basis

    candidates: List[Optional[str]] = [
        dataset_config.embedding_key,
        dataset_config.plot_basis,
    ]
    candidates.extend(["X_umap", "umap", "X_tsne", "tsne", "X_pca", "pca"])
    for cand in candidates:
        if cand is None:
            continue
        basis_name = _normalize_basis(str(cand))
        if basis_name is None:
            continue
        obsm_key = cand if str(cand).lower().startswith("x_") else f"X_{basis_name}"
        if obsm_key in adata.obsm:
            coords = np.asarray(adata.obsm[obsm_key], dtype=np.float32)
            if coords.ndim == 2 and coords.shape[1] >= 2:
                return basis_name, coords[:, :2]

    # Fallback: reuse any existing UMAP-like embedding key (do not recompute).
    # Many datasets store variants like `X_umap_2` or `X_umap_orig`.
    obsm_keys = list(getattr(adata, "obsm", {}).keys())
    umap_keys = [k for k in obsm_keys if k.lower().startswith("x_umap")]
    if "X_umap" in adata.obsm:
        umap_keys = ["X_umap"] + [k for k in umap_keys if k != "X_umap"]
    if not umap_keys:
        umap_keys = [k for k in obsm_keys if "umap" in k.lower() and k.lower().startswith("x_")]
    if umap_keys:
        chosen = sorted(umap_keys, key=lambda x: (0 if x == "X_umap" else 1, x.lower()))[0]
        coords = np.asarray(adata.obsm[chosen], dtype=np.float32)
        if coords.ndim == 2 and coords.shape[1] >= 2:
            basis_name = chosen[2:] if chosen.lower().startswith("x_") else chosen
            return basis_name, coords[:, :2]
    return None, None


def generate_variant_streamplots(
    dataset_name: str,
    adata,
    dataset_config: DatasetConfig,
    stream_embedding: StreamEmbeddingResult,
    variant_velocities: Dict[str, np.ndarray],
    neighbor_indices: np.ndarray,
    neighbor_weights: np.ndarray,
    cell_types: Optional[np.ndarray],
    color_key: Optional[str],
    output_dir: Path,
    context_label: str = "expression",
    save_locally: bool = True,
    figsize: Tuple[float, float] = (8.5, 6.5),
    legend_right_margin: float = 0.78,
) -> Dict[str, Tuple[Path, bool]]:
    scv.settings.set_figure_params(dpi=220, frameon=False)
    scv.settings.verbosity = 0

    dest_dir = Path(output_dir) / f"velovi_{dataset_name}"
    if context_label and context_label != "expression":
        dest_dir = dest_dir / context_label
    if save_locally:
        dest_dir.mkdir(parents=True, exist_ok=True)
    else:
        dest_dir = Path(tempfile.mkdtemp(prefix="velovi_stream_"))

    variant_titles = {
        "baseline": "Baseline",
        "latent": "Baseline + Latent",
        "gnn": "GNN Encoder",
        "gnn_latent": "GNN + Latent",
        "transformer": "Transformer",
        "scvelo_dynamic": "scVelo Dynamic",
        "tivelo": "TIVelo Guidance",
    }
    if context_label == "expression":
        space_label = (dataset_config.plot_basis or "embedding").capitalize()
    else:
        space_label = context_label.capitalize()

    ordered_keys = [key for key in variant_titles if key in variant_velocities]
    if not ordered_keys:
        return {}

    template = adata.copy()
    if "neighbors" not in template.uns:
        try:
            sc.pp.neighbors(template, n_neighbors=30)
        except Exception:
            pass
    basis_name, _ = _select_embedding_for_plot(template, dataset_config)
    if basis_name is None:
        try:
            sc.tl.umap(template)
            basis_name = "umap"
        except Exception:
            basis_name = "pca"
            sc.tl.pca(template)

    saved_paths: Dict[str, Tuple[Path, bool]] = {}
    requested_color_key = color_key or dataset_config.plot_color_key or dataset_config.celltype_key

    for variant_key in ordered_keys:
        velocity_matrix = np.asarray(variant_velocities[variant_key], dtype=np.float32)
        if velocity_matrix.ndim != 2 or velocity_matrix.shape[0] != template.n_obs:
            continue

        adata_stream = template.copy()
        adata_stream.layers["velocity"] = velocity_matrix
        if "Ms" not in adata_stream.layers:
            adata_stream.layers["Ms"] = adata_stream.layers[dataset_config.spliced_layer]

        color_key_value = requested_color_key
        if cell_types is not None:
            series = pd.Categorical(cell_types)
            if color_key_value is None:
                color_key_value = "celltype"
                adata_stream.obs[color_key_value] = series
            elif color_key_value not in adata_stream.obs:
                adata_stream.obs[color_key_value] = series
        if color_key_value is not None and color_key_value not in adata_stream.obs:
            color_key_value = None
        palette_key = f"{color_key_value}_colors" if color_key_value else None
        if color_key_value and color_key_value in adata.obs:
            original = adata.obs[color_key_value]
            if pd.api.types.is_categorical_dtype(original):
                adata_stream.obs[color_key_value] = pd.Categorical(
                    adata_stream.obs[color_key_value],
                    categories=original.cat.categories,
                    ordered=original.cat.ordered,
                )
        if palette_key and palette_key in adata.uns:
            adata_stream.uns[palette_key] = adata.uns[palette_key]

        scv.tl.velocity_graph(
            adata_stream,
            vkey="velocity",
            xkey=dataset_config.spliced_layer,
            n_jobs=32,
            mode_neighbors="distances",
            approx=False,
            show_progress_bar=False,
        )
        scv.tl.velocity_embedding(
            adata_stream,
            basis=basis_name,
            vkey="velocity",
            retain_scale=True,
            use_negative_cosines=False,
        )

        fig, ax = plt.subplots(figsize=figsize)
        scv.pl.velocity_embedding_stream(
            adata_stream,
            basis=basis_name,
            color=color_key_value,
            legend_loc="right margin",
            legend_fontsize=12,
            colorbar=False,
            linewidth=1.8,
            arrow_size=1.6,
            density=1.6,
            alpha=0.9,
            ax=ax,
            show=False,
        )
        method_title = variant_titles.get(variant_key, variant_key.replace("_", " ").title())
        ax.set_title(f"{space_label} - {method_title}", fontsize=11)

        if legend_right_margin is not None:
            try:
                fig.subplots_adjust(right=float(legend_right_margin))
            except Exception:
                pass
        fig.tight_layout()
        filename = f"{dataset_name}_{variant_key}_velocity_streamlines.png"
        if context_label and context_label != "expression":
            filename = f"{dataset_name}_{variant_key}_{context_label}_velocity_streamlines.png"
        ephemeral = not save_locally
        if save_locally:
            stream_filename = dest_dir / filename
        else:
            tmp_file = tempfile.NamedTemporaryFile(
                prefix=f"{dataset_name}_{variant_key}_",
                suffix=".png",
                delete=False,
                dir=dest_dir,
            )
            stream_filename = Path(tmp_file.name)
            tmp_file.close()
        fig.savefig(stream_filename, dpi=220, bbox_inches="tight")
        plt.close(fig)
        dict_key = variant_key if context_label == "expression" else f"{variant_key}_{context_label}"
        saved_paths[dict_key] = (stream_filename, ephemeral)

    return saved_paths


def generate_graph_diagnostics(
    dataset_name: str,
    adata,
    dataset_config: DatasetConfig,
    output_dir: Path,
    color_key: Optional[str],
    save_locally: bool = True,
    basis_override: Optional[str] = None,
    label_suffix: Optional[str] = None,
) -> Dict[str, Tuple[Path, bool]]:
    """Create and save PAGA and FDL diagnostics.

    We render onto a provided Matplotlib figure and save explicitly instead of
    relying on scanpy/scvelo's implicit save mechanism. This avoids surprises
    with ``sc.settings.figdir`` and makes paths predictable for W&B uploads.
    """
    saved: Dict[str, Tuple[Path, bool]] = {}
    suffix_str = f"_{label_suffix}" if label_suffix else ""

    if save_locally:
        dest_dir = Path(output_dir) / f"velovi_{dataset_name}" / "diagnostics"
        dest_dir.mkdir(parents=True, exist_ok=True)
    else:
        dest_dir = Path(tempfile.mkdtemp(prefix="velovi_diag_"))

    print(
        f"[VELOVI][DIAGNOSTIC] Preparing PAGA/FDL (dataset={dataset_name}, save_locally={save_locally}, dest_dir={dest_dir})"
    )

    adata_plot = adata.copy()
    group_key = color_key or dataset_config.celltype_key or dataset_config.plot_color_key
    if group_key not in adata_plot.obs:
        group_key = None
    palette_key = f"{group_key}_colors" if group_key else None
    if palette_key and palette_key in adata.uns:
        adata_plot.uns[palette_key] = adata.uns[palette_key]
    print(f"[VELOVI][DIAGNOSTIC] group_key={group_key}, palette_key={palette_key}")

    preprocess_cfg = dataset_config.preprocess
    if "neighbors" not in adata_plot.uns:
        try:
            sc.pp.neighbors(
                adata_plot,
                n_neighbors=preprocess_cfg.n_neighbors if preprocess_cfg else 30,
                use_rep="X_pca" if "X_pca" in adata_plot.obsm else None,
            )
            print("[VELOVI][DIAGNOSTIC] Computed neighbors for PAGA")
        except Exception as exc:
            print(f"[VELOVI][DIAGNOSTIC] Neighbors computation failed with {exc}; retrying with defaults")
            sc.pp.neighbors(adata_plot, n_neighbors=30)

    basis_name, _ = _select_embedding_for_plot(adata_plot, dataset_config)
    if basis_override:
        override_key = f"X_{basis_override}" if not basis_override.lower().startswith("x_") else basis_override
        if override_key in adata_plot.obsm:
            basis_name = basis_override if not basis_override.lower().startswith("x_") else basis_override[2:]
    if basis_name is None and "X_umap" in adata_plot.obsm:
        basis_name = "umap"
    print(f"[VELOVI][DIAGNOSTIC] Using basis={basis_name} for PAGA plots")

    try:
        # Ensure we have groups; if not, compute Leiden quickly
        if group_key is None:
            sc.pp.neighbors(adata_plot, n_neighbors=30)
            sc.tl.leiden(adata_plot, resolution=0.6)
            group_key = "leiden"
            palette_key = f"{group_key}_colors"
            print("[VELOVI][DIAGNOSTIC] No group_key provided; computed leiden groups for PAGA")

        # Compute velocity graph so transitions are available to paga visualizer
        try:
            scv.tl.velocity_graph(adata_plot, vkey="velocity", n_jobs=32)
            print("[VELOVI][DIAGNOSTIC] Computed velocity graph for PAGA overlays")
        except Exception as exc:
            print(f"[VELOVI][DIAGNOSTIC] velocity_graph failed ({exc}); continuing without it")

        # Compute PAGA on groups
        scv.tl.paga(adata_plot, groups=group_key)
        print("[VELOVI][DIAGNOSTIC] Computed PAGA connectivity")

        # Plot PAGA (cluster graph) – default layout
        try:
            import matplotlib.pyplot as plt

            paga_path = dest_dir / f"paga_{dataset_name}{suffix_str}.png"
            fig1, ax1 = plt.subplots(figsize=(5.5, 4.5))
            paga_kwargs = dict(
                adata=adata_plot,
                color=group_key,
                show=False,
                plot=True,
                ax=ax1,
            )
            if basis_name is not None:
                paga_kwargs["basis"] = basis_name
            scv.pl.paga(**paga_kwargs)
            fig1.tight_layout()
            fig1.savefig(paga_path, dpi=220, bbox_inches="tight")
            plt.close(fig1)
            key_name = "paga" + (f"_{label_suffix}" if label_suffix else "")
            saved[key_name] = (paga_path, not save_locally)
            print(
                f"[VELOVI][DIAGNOSTIC] PAGA saved to {paga_path} (exists={paga_path.exists()})"
            )
        except Exception as exc:
            print(f"[VELOVI][DIAGNOSTIC] Failed to render/save PAGA: {exc}")

        # Plot PAGA with Fruchterman–Reingold layout (FDL visualization of cluster graph)
        try:
            import matplotlib.pyplot as plt

            fdl_path = dest_dir / f"paga_{dataset_name}_fdl{suffix_str}.png"
            fig2, ax2 = plt.subplots(figsize=(5.5, 4.5))
            fdl_kwargs = dict(
                adata=adata_plot,
                color=group_key,
                layout="fr",
                show=False,
                plot=True,
                ax=ax2,
            )
            if basis_name is not None:
                fdl_kwargs["basis"] = basis_name
            scv.pl.paga(**fdl_kwargs)
            fig2.tight_layout()
            fig2.savefig(fdl_path, dpi=220, bbox_inches="tight")
            plt.close(fig2)
            key_name = "fdl" + (f"_{label_suffix}" if label_suffix else "")
            saved[key_name] = (fdl_path, not save_locally)
            print(f"[VELOVI][DIAGNOSTIC] FDL saved to {fdl_path} (exists={fdl_path.exists()})")
        except Exception as exc:
            print(f"[VELOVI][DIAGNOSTIC] Failed to render/save FDL: {exc}")
    except Exception as exc:
        warnings.warn(
            f"Failed to generate PAGA/FDL for {dataset_name}: {exc}", RuntimeWarning
        )

    # draw_graph-based FDL removed to align with scVelo docs (FDL via pl.paga with layout='fr')

    return saved


def plot_paga_guidance_overlay(
    dataset_name: str,
    adata,
    dataset_config: DatasetConfig,
    cluster_key: Optional[str],
    path_dict: Optional[Dict[str, List[str]]],
    main_path: Optional[List[str]],
    output_dir: Path,
    save_locally: bool = True,
) -> Optional[Tuple[Path, bool]]:
    if path_dict is None or not path_dict:
        return None
    if cluster_key is None or cluster_key not in adata.obs:
        print("[VELOVI][DIAGNOSTIC] Cannot plot PAGA guidance without a valid cluster key")
        return None

    basis_name, coords = _select_embedding_for_plot(adata, dataset_config)
    if coords is None:
        print("[VELOVI][DIAGNOSTIC] Cannot plot PAGA guidance without embedding coordinates")
        return None

    cluster_series = pd.Categorical(adata.obs[cluster_key].astype(str))
    categories = list(cluster_series.categories)
    palette_key = f"{cluster_key}_colors"
    if palette_key in adata.uns and len(adata.uns[palette_key]) >= len(categories):
        palette = list(adata.uns[palette_key])
    else:
        cmap = plt.get_cmap("tab20")
        palette = [cmap(i % cmap.N) for i in range(len(categories))]
    label_to_color = {label: palette[idx % len(palette)] for idx, label in enumerate(categories)}
    point_colors = [label_to_color.get(label, "#bbbbbb") for label in cluster_series.astype(str)]

    centroids: Dict[str, np.ndarray] = {}
    for idx, label in enumerate(categories):
        mask = (cluster_series == label).values
        if np.any(mask):
            centroids[label] = coords[mask].mean(axis=0)

    main_edge_set = set()
    if main_path is not None and len(main_path) >= 2:
        for i in range(len(main_path) - 1):
            main_edge_set.add((main_path[i], main_path[i + 1]))

    dest_dir = Path(output_dir) / f"velovi_{dataset_name}" / "diagnostics"
    if save_locally:
        dest_dir.mkdir(parents=True, exist_ok=True)
        fig_path = dest_dir / f"paga_{dataset_name}_guidance.png"
    else:
        tmp_dir = Path(tempfile.mkdtemp(prefix="velovi_guidance_"))
        fig_path = tmp_dir / f"paga_{dataset_name}_guidance.png"

    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    ax.scatter(coords[:, 0], coords[:, 1], c=point_colors, s=8, alpha=0.4, linewidths=0)

    for parent, children in path_dict.items():
        for child in children:
            start = centroids.get(parent)
            end = centroids.get(child)
            if start is None or end is None:
                continue
            highlight = (parent, child) in main_edge_set
            color = "#d62728" if highlight else "#4c78a8"
            lw = 2.5 if highlight else 1.2
            ax.annotate(
                "",
                xy=end,
                xytext=start,
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=color,
                    lw=lw,
                    alpha=0.9 if highlight else 0.65,
                    shrinkA=2,
                    shrinkB=2,
                ),
            )

    ax.set_title(f"{dataset_name} – PAGA Guidance ({basis_name or 'embedding'})")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[VELOVI][DIAGNOSTIC] Guidance plot saved to {fig_path} (exists={fig_path.exists()})")
    return fig_path, not save_locally
