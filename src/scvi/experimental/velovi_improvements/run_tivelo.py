from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import scanpy as sc

try:  # SciPy ≥1.14 removed convenient `A`/`A1` attributes on sparse matrices
    import scipy.sparse as sp

    if not hasattr(sp.spmatrix, "A"):
        sp.spmatrix.A = property(lambda self: self.toarray())  # type: ignore[attr-defined]
    if not hasattr(sp.spmatrix, "A1"):
        sp.spmatrix.A1 = property(lambda self: self.toarray().ravel())  # type: ignore[attr-defined]
except Exception:  # pragma: no cover - best-effort compatibility shim
    pass

try:  # pragma: no cover - optional dependency
    import scvelo as scv
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ModuleNotFoundError(
        "Running the standalone TIVelo pipeline requires the `scvelo` package. "
        "Install it via `pip install scvelo`."
    ) from exc

from scvi.experimental.tivelo.main import tivelo


def _ensure_layers(adata, spliced_key: str, unspliced_key: str) -> None:
    fallback_pairs = [
        ("spliced", "unspliced"),
        ("spliced_counts", "unspliced_counts"),
        ("spliced_norm", "unspliced_norm"),
    ]

    missing = [key for key in (spliced_key, unspliced_key) if key not in adata.layers]
    if missing:
        for fallback_spliced, fallback_unspliced in fallback_pairs:
            if fallback_spliced in adata.layers and fallback_unspliced in adata.layers:
                if spliced_key not in adata.layers:
                    adata.layers[spliced_key] = adata.layers[fallback_spliced]
                if unspliced_key not in adata.layers:
                    adata.layers[unspliced_key] = adata.layers[fallback_unspliced]
                missing = []
                break

    if missing:
        available = list(adata.layers.keys())
        raise KeyError(
            f"Layers {missing} not found in AnnData object. "
            f"Available layers: {available}. Use `--spliced-layer` / `--unspliced-layer` to match your dataset."
        )


def _maybe_preprocess(
    adata,
    *,
    min_shared_counts: int,
    n_top_genes: int,
    n_pcs: int,
    n_neighbors: int,
    log1p: bool,
) -> None:
    scv.pp.filter_and_normalize(
        adata,
        min_shared_counts=min_shared_counts,
        n_top_genes=n_top_genes,
        log=True,
    )
    scv.pp.moments(adata, n_pcs=n_pcs, n_neighbors=n_neighbors)
    if not log1p:
        # Undo log1p if user prefers raw counts downstream.
        adata.layers["Ms"] = (adata.layers["Ms"].expm1() if hasattr(adata.layers["Ms"], "expm1") else adata.layers["Ms"])


def run_tivelo_cli(
    input_path: Optional[Path],
    output_dir: Path,
    *,
    dataset_name: Optional[str],
    scvelo_loader: Optional[str],
    spliced_layer: str,
    unspliced_layer: str,
    group_key: Optional[str],
    embedding_key: Optional[str],
    resolution: float,
    njobs: int,
    start_mode: str,
    rev_stat: str,
    threshold: float,
    threshold_trans: float,
    t1: float,
    t2: float,
    loss_fun: str,
    only_s: bool,
    constrain: bool,
    alpha_1: float,
    alpha_2: float,
    batch_size: int,
    n_epochs: int,
    filter_genes: bool,
    preprocess: bool,
    min_shared_counts: int,
    n_top_genes: int,
    n_pcs: int,
    n_neighbors: int,
    show_fig: bool,
    cluster_edges: Optional[List[Tuple[str, str]]],
    device: str,
) -> None:
    if scvelo_loader is not None and hasattr(scv.datasets, scvelo_loader):
        loader_fn = getattr(scv.datasets, scvelo_loader)
        adata = loader_fn()
        dataset_tag = dataset_name or scvelo_loader
    elif input_path is not None:
        try:
            adata = sc.read_h5ad(input_path)
        except TypeError as exc:
            # Work around mixed/partial `anndata` installs sometimes seen on clusters where
            # `anndata.read_h5ad` passes `attrs=` but the loaded AnnData class doesn't accept it.
            # We monkeypatch AnnData.__init__ to ignore `attrs` and retry.
            msg = str(exc)
            if "unexpected keyword argument 'attrs'" in msg:
                try:
                    import anndata as ad

                    original_init = ad.AnnData.__init__

                    def _patched_init(self, *args, attrs=None, **kwargs):  # noqa: ARG001
                        return original_init(self, *args, **kwargs)

                    ad.AnnData.__init__ = _patched_init  # type: ignore[assignment]
                    print(
                        "[TIVELO] WARNING: Detected `anndata` reader/init mismatch (`attrs` kwarg). "
                        "Applying a local monkeypatch to ignore `attrs` and retrying read_h5ad."
                    )
                    adata = sc.read_h5ad(input_path)
                except Exception as retry_exc:  # pragma: no cover
                    raise TypeError(
                        "Failed to read the provided .h5ad due to an `anndata` incompatibility "
                        "in the current environment (AnnData.__init__ does not accept `attrs`).\n"
                        "Workarounds:\n"
                        "  1) Use a different input file (saved with an older anndata), or\n"
                        "  2) Run this job in an env with a consistent anndata/scanpy install.\n\n"
                        f"Original error: {exc}\nRetry error: {retry_exc}"
                    ) from retry_exc
            else:
                raise
        dataset_tag = dataset_name or input_path.stem
    else:
        raise ValueError("Provide either an input .h5ad path or --scvelo-loader.")

    _ensure_layers(adata, spliced_layer, unspliced_layer)
    adata.layers.setdefault("Ms", adata.layers[spliced_layer])
    adata.layers.setdefault("Mu", adata.layers[unspliced_layer])

    if preprocess:
        _maybe_preprocess(
            adata,
            min_shared_counts=min_shared_counts,
            n_top_genes=n_top_genes,
            n_pcs=n_pcs,
            n_neighbors=n_neighbors,
            log1p=True,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    data_name = dataset_tag

    tivelo(
        adata,
        group_key=group_key,
        emb_key=embedding_key,
        res=resolution,
        data_name=data_name,
        save_folder=str(output_dir),
        njobs=njobs,
        start_mode=start_mode,
        rev_stat=rev_stat,
        t1=t1,
        t2=t2,
        show_fig=show_fig,
        filter_genes=filter_genes,
        constrain=constrain,
        loss_fun=loss_fun,
        only_s=only_s,
        alpha_1=alpha_1,
        alpha_2=alpha_2,
        batch_size=batch_size,
        n_epochs=n_epochs,
        velocity_key="velocity",
        measure_performance=True,
        cluster_edges=cluster_edges,
        device=device,
    )

    # Persist the updated AnnData with TIVelo outputs so downstream plotting/benchmarking
    # can reuse it without rerunning TIVelo.
    try:
        out_path = output_dir / f"{data_name}_tivelo.h5ad"
        adata.write_h5ad(out_path)
        print(f"[TIVELO] Wrote updated AnnData to {out_path}")
    except Exception as exc:  # pragma: no cover
        print(f"[TIVELO] WARNING: failed to save AnnData: {exc}")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the standalone TIVelo pipeline on an AnnData (.h5ad) dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        help="Path to AnnData .h5ad file with spliced/unspliced layers (optional if --scvelo-loader is set).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tivelo_results"),
        help="Directory to store TIVelo outputs (figures + updated AnnData).",
    )
    parser.add_argument("--dataset-name", type=str, default=None, help="Optional dataset tag for output folders.")
    parser.add_argument(
        "--scvelo-loader",
        type=str,
        default=None,
        help="Name of scvelo.datasets loader (e.g., pancreas, hindbrain). Overrides input path.",
    )
    parser.add_argument("--spliced-layer", type=str, default="Ms", help="Layer key for spliced counts.")
    parser.add_argument("--unspliced-layer", type=str, default="Mu", help="Layer key for unspliced counts.")
    parser.add_argument("--group-key", type=str, default=None, help="Precomputed cluster labels in adata.obs.")
    parser.add_argument("--embedding-key", type=str, default=None, help="Embedding key (e.g., X_umap) to reuse.")
    parser.add_argument("--resolution", type=float, default=0.6, help="Resolution used if Leiden clustering is needed.")
    parser.add_argument("--njobs", type=int, default=-1, help="Number of parallel jobs for velocity graph steps.")
    parser.add_argument(
        "--device",
        type=str,
        choices={"auto", "cpu", "cuda"},
        default="auto",
        help="Device to run TIVelo model fitting on.",
    )
    parser.add_argument("--start-mode", type=str, default="stochastic", help="TIVelo path inference start mode.")
    parser.add_argument("--rev-stat", type=str, default="mean", help="Statistic used in path correction.")
    parser.add_argument("--threshold", type=float, default=0.1, help="Edge-weight threshold in directed tree.")
    parser.add_argument("--threshold-trans", type=float, default=1.0, help="Transition threshold in directed tree.")
    parser.add_argument("--t1", type=float, default=0.1, help="DTI threshold t1 passed to TIVelo.")
    parser.add_argument("--t2", type=float, default=1.0, help="DTI transition threshold t2 passed to TIVelo.")
    parser.add_argument("--loss-fun", type=str, default="mse", choices={"mse", "cos"}, help="Loss used for TIVelo model.")
    parser.add_argument("--only-s", action="store_true", help="Optimise only spliced velocities in TIVelo model.")
    parser.add_argument("--no-constrain", dest="constrain", action="store_false", help="Disable TIVelo velocity constraint.")
    parser.set_defaults(constrain=True)
    parser.add_argument("--alpha-1", type=float, default=1.0, help="Weight for spliced loss in TIVelo optimisation.")
    parser.add_argument("--alpha-2", type=float, default=0.1, help="Weight for cosine regulariser in TIVelo optimisation.")
    parser.add_argument("--batch-size", type=int, default=1024, help="Batch size for TIVelo model training.")
    parser.add_argument("--n-epochs", type=int, default=100, help="Number of epochs for TIVelo model training.")
    parser.add_argument("--filter-genes", action="store_true", help="Restrict to velocity genes before training.")
    parser.add_argument("--no-preprocess", dest="preprocess", action="store_false", help="Skip scVelo preprocessing step.")
    parser.set_defaults(preprocess=True)
    parser.add_argument("--min-shared-counts", type=int, default=30, help="Minimum shared counts in preprocessing.")
    parser.add_argument("--n-top-genes", type=int, default=2000, help="Number of highly variable genes to keep.")
    parser.add_argument("--n-pcs", type=int, default=30, help="Number of principal components for moments.")
    parser.add_argument("--n-neighbors", type=int, default=30, help="Number of neighbors for moments.")
    parser.add_argument("--no-show-fig", dest="show_fig", action="store_false", help="Disable interactive figures.")
    parser.set_defaults(show_fig=True)
    parser.add_argument(
        "--cluster-edge",
        action="append",
        default=None,
        metavar="SRC:DEST",
        help=(
            "Specify a directed transition edge (e.g., 'Pre-endocrine:Alpha'). "
            "Pass multiple times to provide several edges."
        ),
    )
    return parser


def main(args: Optional[list[str]] = None) -> None:
    parser = build_argparser()
    namespace = parser.parse_args(args=args)

    cluster_edges: Optional[List[Tuple[str, str]]] = None
    if namespace.cluster_edge:
        cluster_edges = []
        for spec in namespace.cluster_edge:
            if ":" in spec:
                src, dst = spec.split(":", 1)
            elif "," in spec:
                src, dst = spec.split(",", 1)
            else:
                raise ValueError(
                    f"Cluster edge '{spec}' must be in 'source:target' format."
                )
            src = src.strip()
            dst = dst.strip()
            if not src or not dst:
                raise ValueError(f"Invalid cluster edge specification '{spec}'.")
            cluster_edges.append((src, dst))

    run_tivelo_cli(
        namespace.input,
        namespace.output_dir,
        dataset_name=namespace.dataset_name,
        scvelo_loader=namespace.scvelo_loader,
        spliced_layer=namespace.spliced_layer,
        unspliced_layer=namespace.unspliced_layer,
        group_key=namespace.group_key,
        embedding_key=namespace.embedding_key,
        resolution=namespace.resolution,
        njobs=namespace.njobs,
        start_mode=namespace.start_mode,
        rev_stat=namespace.rev_stat,
        threshold=namespace.threshold,
        threshold_trans=namespace.threshold_trans,
        t1=namespace.t1,
        t2=namespace.t2,
        loss_fun=namespace.loss_fun,
        only_s=namespace.only_s,
        constrain=namespace.constrain,
        alpha_1=namespace.alpha_1,
        alpha_2=namespace.alpha_2,
        batch_size=namespace.batch_size,
        n_epochs=namespace.n_epochs,
        filter_genes=namespace.filter_genes,
        preprocess=namespace.preprocess,
        min_shared_counts=namespace.min_shared_counts,
        n_top_genes=namespace.n_top_genes,
        n_pcs=namespace.n_pcs,
        n_neighbors=namespace.n_neighbors,
        show_fig=namespace.show_fig,
        cluster_edges=cluster_edges,
        device=namespace.device,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
