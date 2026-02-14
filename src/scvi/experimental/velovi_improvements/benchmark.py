from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import scanpy as sc

try:  # pragma: no cover - optional dependency guard
    import scvelo as scv
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ModuleNotFoundError(
        "scvelo is required for benchmarking utilities. Please install scvelo."
    ) from exc

from scvi.experimental.tivelo.utils import metrics as tivelo_metrics


def _coerce_neighbor_indices(
    neighbor_indices: np.ndarray,
    n_obs: int,
    *,
    expression: Optional[np.ndarray] = None,
    n_neighbors_fallback: int = 30,
) -> np.ndarray:
    """Ensure neighbor indices are shaped (n_obs, k).

    The runner is expected to pass a 2D integer array, but we occasionally see
    1D arrays due to accidental variable shadowing or flattened storage.
    This helper makes benchmarking robust by coercing/repairing the input.
    """
    import warnings

    arr = np.asarray(neighbor_indices)

    if arr.ndim == 2 and arr.shape[0] == n_obs:
        return arr.astype(np.int64, copy=False)

    if arr.ndim == 1:
        if arr.shape[0] == n_obs:
            warnings.warn(
                "neighbor_indices provided as 1D array; coercing to (n_obs, 1). "
                "This usually indicates a bug upstream (e.g., variable shadowing).",
                RuntimeWarning,
            )
            return arr.reshape(n_obs, 1).astype(np.int64, copy=False)
        if n_obs > 0 and arr.shape[0] % n_obs == 0:
            k = int(arr.shape[0] // n_obs)
            warnings.warn(
                f"neighbor_indices provided as flattened 1D array; reshaping to (n_obs, {k}).",
                RuntimeWarning,
            )
            return arr.reshape(n_obs, k).astype(np.int64, copy=False)

    if expression is None:
        raise ValueError(
            f"Invalid neighbor_indices shape {arr.shape}; expected (n_obs, k) with n_obs={n_obs}. "
            "Provide a valid kNN index matrix."
        )

    warnings.warn(
        f"neighbor_indices has invalid shape {arr.shape}; rebuilding kNN graph from expression "
        f"(k={n_neighbors_fallback}).",
        RuntimeWarning,
    )
    # Last resort: rebuild a kNN graph from expression directly.
    from sklearn.neighbors import NearestNeighbors

    feats = np.asarray(expression, dtype=np.float32)
    nn = NearestNeighbors(n_neighbors=min(n_neighbors_fallback + 1, max(2, n_obs)), metric="euclidean")
    nn.fit(feats)
    _, idx = nn.kneighbors(feats)
    idx = idx[:, 1:]
    return idx.astype(np.int64, copy=False)

@dataclass
class EdgeMetricCollection:
    cbdir_gene: Dict[Tuple[str, str], float]
    cbdir_umap: Dict[Tuple[str, str], float]
    trans_cosine: Dict[Tuple[str, str], float]


@dataclass
class MethodBenchmarkResult:
    cbdir: float
    cbdir2: float
    trans_cosine: float
    trans_probability: float
    icvcoh: float
    icvcoh2: float
    velocoh: float
    velocoh_values: Dict[int, float]
    edge_metrics: EdgeMetricCollection
    fucci_sign_accuracy: Optional[pd.DataFrame] = None
    cell_cycle_velocity_accuracy: Optional[float] = None


def _ensure_dense(matrix):
    return matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _select_embedding_basis(adata) -> str:
    """Pick an existing embedding basis for scvelo plotting.

    Prefers UMAP-like embeddings if present; otherwise falls back to PCA (computing it if needed).
    """
    obsm_keys = list(getattr(adata, "obsm", {}).keys())
    if "X_umap" in adata.obsm:
        return "umap"
    umap_like = [k for k in obsm_keys if str(k).lower().startswith("x_umap")]
    if umap_like:
        key = sorted(umap_like, key=lambda x: x.lower())[0]
        return key[2:] if key.lower().startswith("x_") else key
    if "X_tsne" in adata.obsm:
        return "tsne"
    if "X_pca" not in adata.obsm:
        sc.tl.pca(adata)
    return "pca"


def _collect_cluster_edges(clusters: np.ndarray, neighbor_indices: np.ndarray) -> List[Tuple[str, str]]:
    edges: set[Tuple[str, str]] = set()
    for idx, nbrs in enumerate(neighbor_indices):
        src = clusters[idx]
        for nbr in nbrs:
            dst = clusters[nbr]
            if src != dst:
                edges.add((src, dst))
    return sorted(edges)


def compute_edge_metrics(
    clusters: np.ndarray,
    expression: np.ndarray,
    velocities: np.ndarray,
    embedding: Optional[np.ndarray],
    embedding_velocity: Optional[np.ndarray],
    neighbor_indices: np.ndarray,
    cluster_edges: Optional[List[Tuple[str, str]]] = None,
) -> EdgeMetricCollection:
    cbdir_gene_sum: Dict[Tuple[str, str], float] = {}
    cbdir_gene_count: Dict[Tuple[str, str], int] = {}
    cbdir_umap_sum: Dict[Tuple[str, str], float] = {}
    cbdir_umap_count: Dict[Tuple[str, str], int] = {}
    trans_cosine_sum: Dict[Tuple[str, str], float] = {}
    trans_cosine_count: Dict[Tuple[str, str], int] = {}

    cluster_labels = np.asarray(clusters)
    neighbor_indices = _coerce_neighbor_indices(
        neighbor_indices,
        n_obs=int(velocities.shape[0]),
        expression=expression,
    )

    if cluster_edges is None:
        for idx in range(velocities.shape[0]):
            src_cluster = cluster_labels[idx]
            vel = velocities[idx]
            exp_anchor = expression[idx]
            emb_anchor = embedding[idx] if embedding is not None else None
            vel_embed = embedding_velocity[idx] if embedding_velocity is not None else None
            for nbr in neighbor_indices[idx]:
                dst_cluster = cluster_labels[nbr]
                if src_cluster == dst_cluster:
                    continue
                edge = (src_cluster, dst_cluster)
                diff_exp = expression[nbr] - exp_anchor
                val_gene = _cosine(vel, diff_exp)
                cbdir_gene_sum[edge] = cbdir_gene_sum.get(edge, 0.0) + val_gene
                cbdir_gene_count[edge] = cbdir_gene_count.get(edge, 0) + 1

                val_trans = _cosine(diff_exp, vel)
                trans_cosine_sum[edge] = trans_cosine_sum.get(edge, 0.0) + val_trans
                trans_cosine_count[edge] = trans_cosine_count.get(edge, 0) + 1

                if emb_anchor is not None and vel_embed is not None and embedding is not None:
                    diff_embed = embedding[nbr] - emb_anchor
                    val_umap = _cosine(vel_embed, diff_embed)
                    cbdir_umap_sum[edge] = cbdir_umap_sum.get(edge, 0.0) + val_umap
                    cbdir_umap_count[edge] = cbdir_umap_count.get(edge, 0) + 1
    else:
        cluster_to_indices: Dict[str, np.ndarray] = {}
        for idx, label in enumerate(cluster_labels):
            cluster_to_indices.setdefault(label, []).append(idx)
        for key in cluster_to_indices:
            cluster_to_indices[key] = np.asarray(cluster_to_indices[key], dtype=int)

        for src_cluster, dst_cluster in cluster_edges:
            src_indices = cluster_to_indices.get(src_cluster)
            if src_indices is None:
                continue
            for idx in src_indices:
                vel = velocities[idx]
                exp_anchor = expression[idx]
                emb_anchor = embedding[idx] if embedding is not None else None
                vel_embed = embedding_velocity[idx] if embedding_velocity is not None else None
                neighbors = neighbor_indices[idx]
                if neighbors.size == 0:
                    continue
                dst_mask = cluster_labels[neighbors] == dst_cluster
                if not np.any(dst_mask):
                    continue
                target_nodes = neighbors[dst_mask]
                for nbr in target_nodes:
                    diff_exp = expression[nbr] - exp_anchor
                    val_gene = _cosine(vel, diff_exp)
                    edge = (src_cluster, dst_cluster)
                    cbdir_gene_sum[edge] = cbdir_gene_sum.get(edge, 0.0) + val_gene
                    cbdir_gene_count[edge] = cbdir_gene_count.get(edge, 0) + 1

                    val_trans = _cosine(diff_exp, vel)
                    trans_cosine_sum[edge] = trans_cosine_sum.get(edge, 0.0) + val_trans
                    trans_cosine_count[edge] = trans_cosine_count.get(edge, 0) + 1

                    if emb_anchor is not None and vel_embed is not None and embedding is not None:
                        diff_embed = embedding[nbr] - emb_anchor
                        val_umap = _cosine(vel_embed, diff_embed)
                        cbdir_umap_sum[edge] = cbdir_umap_sum.get(edge, 0.0) + val_umap
                        cbdir_umap_count[edge] = cbdir_umap_count.get(edge, 0) + 1

    cbdir_gene = {
        edge: cbdir_gene_sum[edge] / cbdir_gene_count[edge]
        for edge in cbdir_gene_sum
        if cbdir_gene_count[edge] > 0
    }
    cbdir_umap = {
        edge: cbdir_umap_sum[edge] / cbdir_umap_count[edge]
        for edge in cbdir_umap_sum
        if cbdir_umap_count[edge] > 0
    }
    trans_cosine = {
        edge: trans_cosine_sum[edge] / trans_cosine_count[edge]
        for edge in trans_cosine_sum
        if trans_cosine_count[edge] > 0
    }

    return EdgeMetricCollection(cbdir_gene=cbdir_gene, cbdir_umap=cbdir_umap, trans_cosine=trans_cosine)


def compute_velocoh(
    expression: np.ndarray,
    velocities: np.ndarray,
    neighbor_indices: np.ndarray,
    sigma: Optional[float] = None,
) -> Dict[int, float]:
    neighbor_indices = _coerce_neighbor_indices(
        neighbor_indices,
        n_obs=int(velocities.shape[0]),
        expression=expression,
    )
    if sigma is None:
        sigma = float(np.std(velocities))
        if sigma < 1e-6:
            sigma = 1e-6
    velocoh: Dict[int, float] = {}
    for idx in range(velocities.shape[0]):
        vel = velocities[idx]
        anchor_exp = expression[idx]
        nbrs = neighbor_indices[idx]
        delta = expression[nbrs] - anchor_exp
        gamma = np.array([_cosine(vec, vel) for vec in delta])
        weights = np.exp(gamma / (2.0 * sigma**2))
        weight_sum = weights.sum()
        if weight_sum <= 1e-12:
            weights = np.ones_like(weights) / len(weights)
        else:
            weights /= weight_sum
        predicted = (weights[:, None] * expression[nbrs]).sum(axis=0)
        velocoh[idx] = _cosine(predicted - anchor_exp, vel)
    return velocoh


def compute_fucci_sign_accuracy(
    clusters: np.ndarray,
    expression: np.ndarray,
    velocities: np.ndarray,
) -> pd.DataFrame:
    unique_positions, inverse = np.unique(clusters, return_inverse=True)
    n_groups = len(unique_positions)
    if n_groups < 2:
        return pd.DataFrame(columns=["position", "sign_accuracy"])
    accuracy_rows: List[Dict[str, float | str]] = []
    for idx in range(n_groups):
        current_mask = inverse == idx
        next_mask = inverse == ((idx + 1) % n_groups)
        empirical = expression[next_mask].mean(axis=0) - expression[current_mask].mean(axis=0)
        estimated = velocities[next_mask].mean(axis=0) - velocities[current_mask].mean(axis=0)
        sign_match = np.mean(np.sign(empirical) == np.sign(estimated))
        accuracy_rows.append({"position": unique_positions[idx], "sign_accuracy": float(sign_match)})
    return pd.DataFrame(accuracy_rows)


def compute_cell_cycle_velocity_accuracy(
    adata,
    radial_key: str,
    expression: np.ndarray,
    velocities: np.ndarray,
) -> Optional[float]:
    if radial_key not in adata.obs:
        return None
    radial_values = np.asarray(adata.obs[radial_key], dtype=float)
    unique_pos = np.sort(np.unique(radial_values))
    if unique_pos.size == 0:
        return None
    if unique_pos.size > 1 and np.isclose(unique_pos[-1], 2 * np.pi):
        positions = unique_pos[:-1]
    else:
        positions = unique_pos
    aggregated_counts = []
    aggregated_velocities = []
    valid_positions = []
    for pos in positions:
        if np.isclose(pos, 0.0):
            mask = np.isclose(radial_values, 0.0) | np.isclose(radial_values, 2 * np.pi)
        else:
            mask = np.isclose(radial_values, pos)
        if not np.any(mask):
            continue
        aggregated_counts.append(np.median(expression[mask], axis=0))
        aggregated_velocities.append(np.median(velocities[mask], axis=0))
        valid_positions.append(pos)
    if len(valid_positions) < 2:
        return None
    aggregated_counts = np.asarray(aggregated_counts)
    aggregated_velocities = np.asarray(aggregated_velocities)
    valid_positions = np.asarray(valid_positions, dtype=float)
    reorder = np.arange(1, len(valid_positions)).tolist() + [0]
    diffs = (valid_positions[reorder] - valid_positions) % (2 * np.pi)
    diffs[np.isclose(diffs, 0.0)] = 1e-8
    empirical = (aggregated_counts[reorder] - aggregated_counts) / diffs[:, None]
    empirical_sign = np.sign(empirical)
    predicted_sign = np.sign(aggregated_velocities)
    accuracy_per_gene = (empirical_sign == predicted_sign).mean(axis=0)
    return float(np.mean(accuracy_per_gene))


def _compute_cluster_coherence(
    velocities: np.ndarray,
    clusters: np.ndarray,
) -> float:
    cluster_means: Dict[str, np.ndarray] = {}
    for cl in np.unique(clusters):
        mask = clusters == cl
        mean_vec = velocities[mask].mean(axis=0)
        cluster_means[cl] = mean_vec
    cosines = []
    for idx, vel in enumerate(velocities):
        mean_vec = cluster_means[clusters[idx]]
        cosines.append(_cosine(vel, mean_vec))
    return float(np.mean(cosines)) if cosines else 0.0


def benchmark_methods(
    adata,
    velocities: Dict[str, np.ndarray],
    neighbor_indices: np.ndarray,
    clusters: np.ndarray,
    output_dir: Path,
    method_embedding: Optional[Dict[str, np.ndarray]] = None,
    embedding_coordinates: Optional[np.ndarray] = None,
    fucci_key: Optional[str] = None,
    cluster_edges: Optional[List[Tuple[str, str]]] = None,
    cluster_key: Optional[str] = None,
    method_anndatas: Optional[Dict[str, object]] = None,
    cell_cycle_rad_key: Optional[str] = None,
) -> Dict[str, MethodBenchmarkResult]:
    output_dir.mkdir(parents=True, exist_ok=True)
    expression = _ensure_dense(adata.layers["Ms"])
    neighbor_indices = _coerce_neighbor_indices(
        neighbor_indices,
        n_obs=int(adata.n_obs),
        expression=expression,
    )
    results: Dict[str, MethodBenchmarkResult] = {}
    sigma = float(np.std(expression))
    fucci_clusters = adata.obs[fucci_key].astype(str).to_numpy() if fucci_key and fucci_key in adata.obs else None
    use_cell_cycle_metric = cell_cycle_rad_key is not None and cell_cycle_rad_key in adata.obs

    for method, vel in velocities.items():
        tivelo_override = False
        tivelo_cbdir_gene = {}
        tivelo_cbdir_umap = {}
        tivelo_trans_scores = {}
        tivelo_cbdir = tivelo_cbdir2 = tivelo_trans_prob = None
        tivelo_icvcoh = tivelo_icvcoh2 = None
        tivelo_velocoh = None
        cell_cycle_accuracy = None
        if (
            method_anndatas is not None
            and method in method_anndatas
            and cluster_edges
            and cluster_key
            and cluster_key in method_anndatas[method].obs
        ):
            try:
                adata_method = method_anndatas[method]
                tivelo_cbdir_umap, tivelo_cbdir2 = tivelo_metrics.cross_boundary_correctness(
                    adata_method,
                    cluster_key=cluster_key,
                    velocity_key="velocity",
                    cluster_edges=cluster_edges,
                )
                tivelo_cbdir_gene, tivelo_cbdir = tivelo_metrics.cross_boundary_correctness2(
                    adata_method,
                    cluster_key=cluster_key,
                    velocity_key="velocity",
                    cluster_edges=cluster_edges,
                )
                tivelo_trans_scores, tivelo_trans_prob = tivelo_metrics.cross_boundary_scvelo_probs(
                    adata_method,
                    cluster_key=cluster_key,
                    cluster_edges=cluster_edges,
                    trans_g_key="velocity_graph",
                )
                _, tivelo_icvcoh = tivelo_metrics.inner_cluster_coh(
                    adata_method,
                    cluster_key=cluster_key,
                    velocity_key="velocity",
                )
                _, tivelo_icvcoh2 = tivelo_metrics.inner_cluster_coh2(
                    adata_method,
                    cluster_key=cluster_key,
                    velocity_key="velocity",
                    x_emb="X_umap",
                )
                tivelo_velocoh = tivelo_metrics.velo_coh(
                    adata_method,
                    velocity_key="velocity",
                    trans_g_key="velocity_graph",
                )
                tivelo_override = True
            except Exception:
                tivelo_override = False
        emb_vel = None
        if method_embedding and method in method_embedding:
            emb_vel = method_embedding[method]
        if tivelo_override:
            edge_metrics = EdgeMetricCollection(
                cbdir_gene=tivelo_cbdir_gene,
                cbdir_umap=tivelo_cbdir_umap,
                trans_cosine=tivelo_trans_scores,
            )
            cbdir = float(tivelo_cbdir) if tivelo_cbdir is not None else 0.0
            cbdir2 = float(tivelo_cbdir2) if tivelo_cbdir2 is not None else 0.0
            trans_prob = float(tivelo_trans_prob) if tivelo_trans_prob is not None else 0.0
            trans_cos = float(2.0 * trans_prob - 1.0)
            icvcoh = float(tivelo_icvcoh) if tivelo_icvcoh is not None else 0.0
            icvcoh2 = float(tivelo_icvcoh2) if tivelo_icvcoh2 is not None else 0.0
            velocoh_mean = float(tivelo_velocoh) if tivelo_velocoh is not None else 0.0
        else:
            edge_metrics = compute_edge_metrics(
                clusters=clusters,
                expression=expression,
                velocities=vel,
                embedding=embedding_coordinates,
                embedding_velocity=emb_vel,
                neighbor_indices=neighbor_indices,
                cluster_edges=cluster_edges,
            )
            velocoh = compute_velocoh(expression=expression, velocities=vel, neighbor_indices=neighbor_indices, sigma=sigma)
            cbdir = float(np.mean(list(edge_metrics.cbdir_gene.values()))) if edge_metrics.cbdir_gene else 0.0
            cbdir2 = float(np.mean(list(edge_metrics.cbdir_umap.values()))) if edge_metrics.cbdir_umap else 0.0
            trans_cos = float(np.mean(list(edge_metrics.trans_cosine.values()))) if edge_metrics.trans_cosine else 0.0
            trans_prob = float(np.mean([(val + 1.0) / 2.0 for val in edge_metrics.trans_cosine.values()])) if edge_metrics.trans_cosine else 0.0
            icvcoh = _compute_cluster_coherence(vel, clusters)
            icvcoh2 = 0.0
            if emb_vel is not None:
                icvcoh2 = _compute_cluster_coherence(emb_vel, clusters)
            velocoh_mean = float(np.mean(list(velocoh.values()))) if velocoh else 0.0
        if use_cell_cycle_metric:
            cell_cycle_accuracy = compute_cell_cycle_velocity_accuracy(
                adata=adata,
                radial_key=cell_cycle_rad_key,
                expression=expression,
                velocities=vel,
            )
            edge_metrics = EdgeMetricCollection(cbdir_gene={}, cbdir_umap={}, trans_cosine={})
            cbdir = float("nan")
            cbdir2 = float("nan")
            trans_cos = float("nan")
            trans_prob = float("nan")
        fucci_df = None
        if fucci_clusters is not None:
            fucci_df = compute_fucci_sign_accuracy(fucci_clusters, expression, vel)
        results[method] = MethodBenchmarkResult(
            cbdir=cbdir,
            cbdir2=cbdir2,
            trans_cosine=trans_cos,
            trans_probability=trans_prob,
            icvcoh=icvcoh,
            icvcoh2=icvcoh2,
            velocoh=velocoh_mean,
            velocoh_values=velocoh if not tivelo_override else {},
            edge_metrics=edge_metrics,
            fucci_sign_accuracy=fucci_df,
            cell_cycle_velocity_accuracy=cell_cycle_accuracy,
        )
    return results


def plot_stream_grid(
    dataset_id: str,
    adata,
    velocities: Dict[str, np.ndarray],
    cluster_key: Optional[str],
    output_path: Path,
) -> Path:
    methods = list(velocities.keys())
    fig, axes = plt.subplots(1, len(methods), figsize=(6 * len(methods), 5.5), squeeze=False)
    for ax, method in zip(axes.flat, methods):
        adata_tmp = adata.copy()
        adata_tmp.layers["velocity"] = velocities[method]
        scv.tl.velocity_graph(adata_tmp, vkey="velocity", n_jobs=32, mode_neighbors="distances")
        basis = _select_embedding_basis(adata_tmp)
        scv.tl.velocity_embedding(adata_tmp, basis=basis, vkey="velocity")
        color_key = cluster_key if (cluster_key is not None and cluster_key in adata_tmp.obs) else None
        scv.pl.velocity_embedding_stream(
            adata_tmp,
            basis=basis,
            color=color_key,
            legend_loc="right margin" if color_key else None,
            legend_fontsize=12,
            colorbar=False,
            linewidth=1.8,
            arrow_size=1.6,
            density=1.6,
            alpha=0.9,
            title=method,
            ax=ax,
            show=False,
        )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path
def plot_comparison_panel(
    dataset_id: str,
    adata,
    pairs: Dict[str, Tuple[np.ndarray, np.ndarray]],
    cluster_key: Optional[str],
    output_path: Path,
) -> Path:
    methods = list(pairs.keys())
    fig, axes = plt.subplots(len(methods), 2, figsize=(14, 5.5 * len(methods)), squeeze=False)
    for row, method in enumerate(methods):
        before, after = pairs[method]
        adata_before = adata.copy()
        adata_before.layers["velocity"] = before
        scv.tl.velocity_graph(adata_before, vkey="velocity", n_jobs=32, mode_neighbors="distances")
        basis = _select_embedding_basis(adata_before)
        scv.tl.velocity_embedding(adata_before, basis=basis, vkey="velocity")
        scv.pl.velocity_embedding_stream(
            adata_before,
            basis=basis,
            color=cluster_key if (cluster_key is not None and cluster_key in adata_before.obs) else None,
            legend_loc="right margin" if (cluster_key is not None and cluster_key in adata_before.obs) else None,
            legend_fontsize=12,
            colorbar=False,
            linewidth=1.8,
            arrow_size=1.6,
            density=1.6,
            alpha=0.9,
            title=f"{method} - Before",
            ax=axes[row, 0],
            show=False,
        )

        adata_after = adata.copy()
        adata_after.layers["velocity"] = after
        scv.tl.velocity_graph(adata_after, vkey="velocity", n_jobs=32, mode_neighbors="distances")
        scv.tl.velocity_embedding(adata_after, basis=basis, vkey="velocity")
        scv.pl.velocity_embedding_stream(
            adata_after,
            basis=basis,
            color=cluster_key if (cluster_key is not None and cluster_key in adata_after.obs) else None,
            legend_loc="right margin" if (cluster_key is not None and cluster_key in adata_after.obs) else None,
            legend_fontsize=12,
            colorbar=False,
            linewidth=1.8,
            arrow_size=1.6,
            density=1.6,
            alpha=0.9,
            title=f"{method} - After TIVelo",
            ax=axes[row, 1],
            show=False,
        )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def plot_performance_bars(
    dataset_id: str,
    results: Dict[str, MethodBenchmarkResult],
    output_path: Path,
) -> Path:
    methods = list(results.keys())
    cbdir = [results[m].cbdir for m in methods]
    icvcoh = [results[m].icvcoh for m in methods]
    trans_probs = [results[m].trans_probability for m in methods]
    velocoh = [results[m].velocoh for m in methods]

    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    axes[0].bar(methods, cbdir)
    axes[0].set_ylabel("CBDir")
    axes[1].bar(methods, icvcoh)
    axes[1].set_ylabel("ICVCoh")
    axes[2].bar(methods, trans_probs)
    axes[2].set_ylabel("TransProb")
    axes[3].bar(methods, velocoh)
    axes[3].set_ylabel("VeloCoh")
    axes[3].tick_params(axis="x", rotation=35)
    fig.suptitle(f"Velocity metrics: {dataset_id}")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="pdf")
    plt.close(fig)
    return output_path


def plot_fucci_violin(
    dataset_id: str,
    results: Dict[str, MethodBenchmarkResult],
    output_path: Path,
) -> Optional[Path]:
    fucci_frames = []
    for method, res in results.items():
        if res.fucci_sign_accuracy is None or res.fucci_sign_accuracy.empty:
            continue
        df = res.fucci_sign_accuracy.copy()
        df["method"] = method
        fucci_frames.append(df)
    if not fucci_frames:
        return None
    combined = pd.concat(fucci_frames, ignore_index=True)
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    sns.violinplot(data=combined, x="method", y="sign_accuracy", ax=ax, inner=None)
    sns.boxplot(
        data=combined,
        x="method",
        y="sign_accuracy",
        ax=ax,
        showcaps=True,
        boxprops={"facecolor": (0, 0, 0, 0)},
    )
    means = combined.groupby("method")["sign_accuracy"].mean()
    for i, method in enumerate(means.index):
        ax.text(i, means[method] - 0.05, f"{means[method]:.2f}", ha="center", va="top")
    ax.set_ylabel("Sign Accuracy")
    ax.set_xlabel("")
    fig.suptitle(f"FUCCI sign accuracy: {dataset_id}")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="pdf")
    plt.close(fig)
    return output_path


def plot_gene_level_panels(
    dataset_id: str,
    adata,
    genes: Iterable[str],
    methods: Dict[str, np.ndarray],
    cluster_key: Optional[str],
    output_dir: Path,
) -> List[Path]:
    saved_paths: List[Path] = []
    genes = [g for g in genes if g in adata.var_names]
    if not genes:
        return saved_paths
    method_names = list(methods.keys())
    for gene in genes:
        fig, axes = plt.subplots(2, len(method_names), figsize=(4 * len(method_names), 8), squeeze=False)
        for col, method in enumerate(method_names):
            adata_tmp = adata.copy()
            adata_tmp.layers["velocity"] = methods[method]
            scv.tl.velocity_graph(adata_tmp, vkey="velocity", n_jobs=32, mode_neighbors="distances")
            basis = _select_embedding_basis(adata_tmp)
            try:
                scv.tl.velocity_embedding(adata_tmp, basis=basis, vkey="velocity")
            except Exception:
                # If the basis isn't available and can't be computed for some reason, fall back to PCA.
                basis = "pca"
                if "X_pca" not in adata_tmp.obsm:
                    sc.tl.pca(adata_tmp)
                scv.tl.velocity_embedding(adata_tmp, basis=basis, vkey="velocity")
            color_key = cluster_key if (cluster_key is not None and cluster_key in adata_tmp.obs) else None
            scatter_kwargs = dict(
                layer="Ms",
                color=color_key,
                legend_loc="right margin",
                legend_fontsize=12,
                legend_align_text=True,
                use_raw=False,
                ax=axes[0, col],
                title=f"{method}: {gene}",
                show=False,
            )
            scv.pl.scatter(adata_tmp, var_names=[gene], **scatter_kwargs)
            scv.pl.velocity_embedding(
                adata_tmp,
                basis=basis,
                color=color_key,
                legend_loc="right margin",
                legend_fontsize=12,
                legend_align_text=True,
                ax=axes[1, col],
                show=False,
            )
        fig.tight_layout()
        out_path = output_dir / f"{dataset_id}_{gene}_gene_panels.pdf"
        fig.savefig(out_path, format="pdf")
        plt.close(fig)
        saved_paths.append(out_path)
    return saved_paths


def build_tables(
    dataset_id: str,
    results: Dict[str, MethodBenchmarkResult],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    edge_rows = []
    for method, res in results.items():
        summary_rows.append(
            {
                "dataset": dataset_id,
                "method": method,
                "cbdir": res.cbdir,
                "cbdir2": res.cbdir2,
                "transcosine": res.trans_cosine,
                "transprob": res.trans_probability,
                "icvcoh": res.icvcoh,
                "icvcoh2": res.icvcoh2,
                "velocoh": res.velocoh,
                "cell_cycle_accuracy": res.cell_cycle_velocity_accuracy,
            }
        )
        for edge, value in res.edge_metrics.cbdir_gene.items():
            edge_rows.append(
                {
                    "dataset": dataset_id,
                    "method": method,
                    "edge_src": edge[0],
                    "edge_dst": edge[1],
                    "cbdir": value,
                    "cbdir2": res.edge_metrics.cbdir_umap.get(edge, np.nan),
                    "trans_cosine": res.edge_metrics.trans_cosine.get(edge, np.nan),
                }
            )
    return pd.DataFrame(summary_rows), pd.DataFrame(edge_rows)


def save_tables(
    dataset_id: str,
    results: Dict[str, MethodBenchmarkResult],
    output_dir: Path,
) -> Tuple[Path, Path]:
    summary_df, edge_df = build_tables(dataset_id, results)
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    summary_path = tables_dir / f"{dataset_id}_advanced_summary.csv"
    edges_path = tables_dir / f"{dataset_id}_advanced_edges.csv"
    summary_df.to_csv(summary_path, index=False)
    edge_df.to_csv(edges_path, index=False)
    return summary_path, edges_path
