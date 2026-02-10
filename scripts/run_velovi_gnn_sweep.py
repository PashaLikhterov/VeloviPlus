#!/usr/bin/env python
from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

import pandas as pd

from scvi.experimental.velovi_improvements.runner import TrainingConfig, VELOVIImprovementRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Grid sweep for VELOVI GNN neighbour configurations."
    )
    parser.add_argument("data_dir", type=Path, help="Directory containing benchmark datasets.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="pancreas_endocrinogenesis",
        help="Dataset key defined in VELOVI_DATASETS.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("velovi_gnn_sweep"),
        help="Directory where sweep artefacts and summaries are stored.",
    )
    parser.add_argument("--warmup-epochs", type=int, default=100)
    parser.add_argument("--total-epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--latent-dim", type=int, default=10)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--gnn-hidden-dim", type=int, default=128)
    parser.add_argument("--gnn-dropout", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--neighbor-sources",
        nargs="+",
        default=["expression", "latent", "both"],
        help="Neighbor sources to evaluate. Defaults to expression/latent/both.",
    )
    parser.add_argument(
        "--sweep-attention",
        action="store_true",
        help="Include attention on/off when sweeping message passing variants.",
    )
    parser.add_argument(
        "--sweep-gate",
        action="store_true",
        help="Include gate on/off when sweeping message passing variants.",
    )
    parser.add_argument(
        "--velocity-laplacian-weight",
        type=float,
        default=0.0,
        help="Weight for Laplacian velocity regularisation.",
    )
    parser.add_argument(
        "--velocity-angle-weight",
        type=float,
        default=0.0,
        help="Weight for angular consistency penalty.",
    )
    parser.add_argument(
        "--velocity-angle-eps",
        type=float,
        default=1e-6,
        help="Stability epsilon used for the angular penalty.",
    )
    parser.add_argument(
        "--stream-embed",
        choices=["pca", "umap"],
        default="pca",
        help="Embedding method applied when saving streamline figures.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    attention_options = [False, True] if args.sweep_attention else [False]
    gate_options = [False, True] if args.sweep_gate else [False]

    sweep_records: list[pd.DataFrame] = []

    for neighbor_source, use_attention, use_gate in product(
        args.neighbor_sources, attention_options, gate_options
    ):
        variant_slug = f"{neighbor_source}_att{int(use_attention)}_gate{int(use_gate)}"
        variant_dir = args.output_dir / variant_slug
        variant_dir.mkdir(parents=True, exist_ok=True)

        config = TrainingConfig(
            warmup_epochs=args.warmup_epochs,
            total_epochs=args.total_epochs,
            batch_size=args.batch_size,
            n_latent=args.latent_dim,
            n_hidden=args.hidden_dim,
            gnn_hidden_dim=args.gnn_hidden_dim,
            gnn_dropout_rate=args.gnn_dropout,
            num_workers=args.num_workers,
            enable_latent_smoothing=True,
            enable_gnn=True,
            enable_gnn_latent_smoothing=True,
            produce_plots=True,
            gnn_neighbor_source=neighbor_source,
            gnn_use_attention=use_attention,
            gnn_use_gate=use_gate,
            gnn_use_residual=True,
            gnn_use_differences=True,
            velocity_laplacian_weight=args.velocity_laplacian_weight,
            velocity_angle_weight=args.velocity_angle_weight,
            velocity_angle_eps=args.velocity_angle_eps,
            stream_embed_method=args.stream_embed,
        )

        runner = VELOVIImprovementRunner(args.data_dir, variant_dir, config)
        results = runner.run(dataset_names=[args.dataset])
        results["neighbor_source"] = neighbor_source
        results["use_attention"] = use_attention
        results["use_gate"] = use_gate
        sweep_records.append(results)

    summary = pd.concat(sweep_records, ignore_index=True)
    summary_path = args.output_dir / f"{args.dataset}_gnn_sweep_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved sweep summary to {summary_path}")


if __name__ == "__main__":
    main()
