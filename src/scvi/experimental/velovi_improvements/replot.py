from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional

try:  # optional dependency
    import wandb
except ModuleNotFoundError:  # pragma: no cover
    wandb = None  # type: ignore

from .analysis import generate_variant_streamplots
from .datasets import get_dataset_config
from .plot_bundle import load_plot_bundle
from .training import compute_stream_embedding
from .config import TrainingConfig
import numpy as np
import scanpy as sc


def _download_bundle_from_wandb(
    *,
    run_path: str,
    artifact_name: str,
    output_dir: Path,
) -> Path:
    if wandb is None:
        raise ModuleNotFoundError("wandb is required to download plot bundles. Install via `pip install wandb`.")

    # Avoid writing to a non-writable shared HOME cache on clusters.
    cache_root = Path(output_dir) / "wandb_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("WANDB_CACHE_DIR", str(cache_root))
    os.environ.setdefault("WANDB_CONFIG_DIR", str(cache_root / "config"))
    os.environ.setdefault("WANDB_DIR", str(cache_root / "run"))

    api = wandb.Api()
    run = api.run(run_path)
    artifacts = list(run.logged_artifacts())
    for art in artifacts:
        if art.name.split(":")[0] == artifact_name:
            dest = output_dir / "wandb_artifacts" / artifact_name
            dest.mkdir(parents=True, exist_ok=True)
            return Path(art.download(root=str(dest)))
    raise ValueError(f"No artifact named '{artifact_name}' found in run {run_path}.")


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Re-generate streamline plots from a saved plot bundle.")
    parser.add_argument("--bundle-dir", type=Path, default=None, help="Local plot bundle directory.")
    parser.add_argument(
        "--wandb-run",
        type=str,
        default=None,
        help="W&B run path in the form 'ENTITY/PROJECT/RUN_ID' (alternative to --bundle-dir).",
    )
    parser.add_argument("--wandb-artifact", type=str, default="plot_bundle", help="Artifact name to download.")
    parser.add_argument("--output-dir", type=Path, default=Path("replot_outputs"), help="Where to write figures.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (for dataset_config lookup).")
    parser.add_argument("--color-key", type=str, default=None, help="Override obs key used for coloring.")
    parser.add_argument(
        "--adata-path",
        type=Path,
        default=None,
        help=(
            "Optional path to an AnnData .h5ad to use for plotting (to preserve the dataset's original UMAP). "
            "Must contain the same cells as the bundle (matching `obs_names`)."
        ),
    )
    parser.add_argument(
        "--figsize",
        type=str,
        default="12,8",
        help="Figure size for each streamline plot as 'width,height' (inches).",
    )
    parser.add_argument(
        "--legend-right",
        type=float,
        default=0.78,
        help="Right margin reserved for legend (0-1 figure fraction).",
    )
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[REplot] output_dir={args.output_dir.resolve()}")

    bundle_dir = args.bundle_dir
    if bundle_dir is None:
        if args.wandb_run is None:
            raise SystemExit("Provide either --bundle-dir or --wandb-run.")
        bundle_dir = _download_bundle_from_wandb(
            run_path=args.wandb_run,
            artifact_name=args.wandb_artifact,
            output_dir=args.output_dir,
        )
    print(f"[REplot] bundle_dir={Path(bundle_dir).resolve()}")

    adata, velocities, meta = load_plot_bundle(bundle_dir)
    dataset_config = get_dataset_config(args.dataset)

    if args.adata_path is not None:
        print(f"[REplot] loading adata_path={args.adata_path}")
        adata_plot = sc.read_h5ad(args.adata_path)
        if adata_plot.n_obs != adata.n_obs or not np.array_equal(adata_plot.obs_names.values, adata.obs_names.values):
            # Try aligning by obs_names
            common = adata_plot.obs_names.intersection(adata.obs_names)
            if len(common) != adata.n_obs:
                raise SystemExit(
                    f"--adata-path does not match bundle cells. "
                    f"bundle_n_obs={adata.n_obs} common={len(common)} adata_path_n_obs={adata_plot.n_obs}"
                )
            adata_plot = adata_plot[adata.obs_names].copy()
        adata = adata_plot

    color_key = args.color_key or meta.get("plot_color_key") or dataset_config.plot_color_key
    config = TrainingConfig(produce_plots=True)
    stream_embedding = compute_stream_embedding(adata, config, dataset_config)

    try:
        w_str, h_str = args.figsize.split(",", 1)
        figsize = (float(w_str.strip()), float(h_str.strip()))
    except Exception:
        raise SystemExit("--figsize must be in 'width,height' format, e.g. 12,8")

    neighbor_indices = adata.obsm.get("velovi_gnn_indices")
    neighbor_weights = adata.obsm.get("velovi_gnn_weights")
    if neighbor_indices is None:
        neighbor_indices = np.zeros((adata.n_obs, 1), dtype=np.int64)
    if neighbor_weights is None:
        neighbor_weights = np.ones((adata.n_obs, 1), dtype=np.float32)

    generate_variant_streamplots(
        dataset_name=args.dataset,
        adata=adata,
        dataset_config=dataset_config,
        stream_embedding=stream_embedding,
        variant_velocities=velocities,
        neighbor_indices=neighbor_indices,
        neighbor_weights=neighbor_weights,
        cell_types=None,
        color_key=color_key,
        output_dir=args.output_dir,
        save_locally=True,
        figsize=figsize,
        legend_right_margin=args.legend_right,
    )
    final_dir = args.output_dir / f"velovi_{args.dataset}"
    print(f"[REplot] wrote figures under {final_dir.resolve()}")


if __name__ == "__main__":  # pragma: no cover
    main()
