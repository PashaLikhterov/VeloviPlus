#!/usr/bin/env python
"""Sweep VELOVI preprocessing options within a single training run."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import replace
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from scvi.experimental.velovi_improvements.datasets import (
    DATASET_ALIASES,
    DatasetConfig,
    PreprocessConfig,
    VELOVI_DATASETS,
    resolve_dataset_name,
)
from scvi.experimental.velovi_improvements.runner import TrainingConfig, VELOVIImprovementRunner


def _coerce_value(token: str) -> Any:
    token_strip = token.strip()
    lower = token_strip.lower()
    if lower in {"true", "false"}:
        return lower == "true"
    try:
        return int(token_strip)
    except ValueError:
        try:
            return float(token_strip)
        except ValueError:
            return token_strip


def _parse_grid_args(grid_args: Sequence[str]) -> Dict[str, Sequence[Any]]:
    grid: Dict[str, Sequence[Any]] = {}
    for spec in grid_args:
        if "=" not in spec:
            raise ValueError(f"Grid specification `{spec}` must be of the form key=val1,val2,...")
        key, raw_values = spec.split("=", 1)
        values = [_coerce_value(v) for v in raw_values.split(",") if v.strip()]
        if not values:
            raise ValueError(f"No values provided for grid key `{key}`.")
        grid[key.strip()] = values
    return grid


def _load_variant_specs(path: Path) -> List[Dict[str, Any]]:
    raw = path.read_text()
    suffix = path.suffix.lower()
    data: Any
    if suffix in {".yml", ".yaml"}:
        try:
            import yaml  # type: ignore
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise ModuleNotFoundError(
                f"PyYAML is required to read variant definitions from {path}. "
                "Install it with `pip install pyyaml`."
            ) from exc
        data = yaml.safe_load(raw)
    else:
        data = json.loads(raw)

    if isinstance(data, dict) and "variants" in data:
        data = data["variants"]
    if not isinstance(data, list):
        raise ValueError(
            f"Variant specification at {path} must be a list or contain a top-level `variants` list."
        )
    specs: List[Dict[str, Any]] = []
    for entry in data:
        if not isinstance(entry, dict):
            raise ValueError("Each variant entry must be a mapping of preprocessing parameters.")
        specs.append(dict(entry))
    return specs


def _format_suffix(updates: Dict[str, Any]) -> str:
    parts: List[str] = []
    for key, value in updates.items():
        if isinstance(value, bool):
            value_str = "true" if value else "false"
        else:
            value_str = str(value).replace(" ", "")
        parts.append(f"{key}{value_str}")
    return "_".join(parts)


def _slugify(label: str) -> str:
    return re.sub(r"[^0-9A-Za-z._-]+", "_", label)


def _build_variants(
    base_config: PreprocessConfig,
    include_default: bool,
    grid: Dict[str, Sequence[Any]],
    variant_specs: Iterable[Dict[str, Any]],
) -> List[PreprocessConfig]:
    variants: List[PreprocessConfig] = []
    seen_signatures = set()

    def register(cfg: PreprocessConfig):
        sig = cfg.signature()
        if sig in seen_signatures:
            return
        seen_signatures.add(sig)
        variants.append(cfg)

    if include_default:
        default_cfg = replace(base_config)
        if default_cfg.name is None:
            default_cfg.name = "default"
        register(default_cfg)

    if grid:
        keys = list(grid.keys())
        for combo in product(*(grid[k] for k in keys)):
            updates = dict(zip(keys, combo))
            cfg = replace(base_config, **updates)
            cfg.name = f"grid_{_format_suffix(updates)}"
            register(cfg)

    for spec in variant_specs:
        name = spec.get("name")
        params = {k: v for k, v in spec.items() if k != "name"}
        cfg = replace(base_config, **params)
        cfg.name = name or cfg.name
        register(cfg)

    if not variants:
        raise ValueError("No preprocessing variants were generated. Provide a grid or variant file.")

    return variants


def _build_dataset_mapping(
    dataset_names: Sequence[str],
    variants_per_dataset: Dict[str, List[PreprocessConfig]],
) -> Dict[str, DatasetConfig]:
    dataset_configs: Dict[str, DatasetConfig] = {}
    for dataset in dataset_names:
        canonical = resolve_dataset_name(dataset)
        base_config = VELOVI_DATASETS[canonical]
        for variant in variants_per_dataset[dataset]:
            label = variant.display_name()
            slug = _slugify(label)
            key = f"{base_config.name}__{slug}"
            dataset_configs[key] = replace(base_config, preprocess=variant)
    return dataset_configs


def _build_training_config(args: argparse.Namespace) -> TrainingConfig:
    kwargs: Dict[str, Any] = {}
    for key in (
        "warmup_epochs",
        "total_epochs",
        "batch_size",
        "latent_dim",
        "hidden_dim",
        "gnn_hidden_dim",
        "gnn_dropout",
        "num_workers",
    ):
        value = getattr(args, key, None)
        if value is not None:
            mapped_key = key
            if key == "latent_dim":
                mapped_key = "n_latent"
            elif key == "hidden_dim":
                mapped_key = "n_hidden"
            elif key == "gnn_dropout":
                mapped_key = "gnn_dropout_rate"
            kwargs[mapped_key] = value

    if args.disable_latent_smoothing:
        kwargs["enable_latent_smoothing"] = False
    if args.disable_gnn:
        kwargs["enable_gnn"] = False
    if args.enable_gnn_latent:
        kwargs["enable_gnn_latent_smoothing"] = True
    if args.plot_results:
        kwargs["produce_plots"] = True
    if args.disable_local_figures:
        kwargs["save_figures_locally"] = False
    if args.disable_checkpoints:
        kwargs["use_checkpoints"] = False
    if args.pretrained:
        kwargs["load_pretrained"] = True
    if args.gnn_neighbor_source is not None:
        kwargs["gnn_neighbor_source"] = args.gnn_neighbor_source
    if args.checkpoint_dir is not None:
        kwargs["checkpoint_dir"] = str(args.checkpoint_dir)
    if args.use_wandb:
        kwargs["use_wandb"] = True
    for attr in ("wandb_project", "wandb_entity", "wandb_api_key", "wandb_run_group"):
        value = getattr(args, attr, None)
        if value is not None:
            kwargs[attr] = value

    return TrainingConfig(**kwargs)


def main():
    parser = argparse.ArgumentParser(
        description="Sweep VELOVI preprocess configurations within a single training run."
    )
    parser.add_argument("data_dir", type=Path, help="Directory containing benchmark datasets.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("velovi_preprocess_sweep"),
        help="Destination directory for sweep outputs (default: velovi_preprocess_sweep).",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(set(VELOVI_DATASETS.keys()) | set(DATASET_ALIASES.keys())),
        required=True,
        help="Datasets to include in the sweep.",
    )
    parser.add_argument(
        "--grid",
        action="append",
        default=[],
        help="Grid specification of the form key=val1,val2. Repeat for multiple keys.",
    )
    parser.add_argument(
        "--variant-file",
        type=Path,
        help="Optional JSON or YAML file with explicit preprocessing variant definitions.",
    )
    parser.add_argument(
        "--skip-default",
        action="store_true",
        help="Do not include the dataset's default preprocessing configuration.",
    )
    parser.add_argument(
        "--selection-metric",
        default="baseline_gene_likelihood_mean",
        help="Metric column used to select the best preprocessing variant.",
    )
    parser.add_argument(
        "--selection-mode",
        choices=["max", "min"],
        default="max",
        help="Whether to maximise or minimise the selection metric when ranking variants.",
    )
    parser.add_argument("--warmup-epochs", type=int)
    parser.add_argument("--total-epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--latent-dim", type=int)
    parser.add_argument("--hidden-dim", type=int)
    parser.add_argument("--gnn-hidden-dim", type=int)
    parser.add_argument("--gnn-dropout", type=float)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--disable-latent-smoothing", action="store_true")
    parser.add_argument("--disable-gnn", action="store_true")
    parser.add_argument("--enable-gnn-latent", action="store_true")
    parser.add_argument("--plot-results", action="store_true")
    parser.add_argument(
        "--disable-local-figures",
        action="store_true",
        help="Skip saving plots to disk (diagnostics still uploaded to W&B if enabled).",
    )
    parser.add_argument("--disable-checkpoints", action="store_true")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument(
        "--gnn-neighbor-source",
        choices=["expression", "latent", "both"],
        help="Override the GNN neighbour construction strategy.",
    )
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str)
    parser.add_argument("--wandb-entity", type=str)
    parser.add_argument("--wandb-api-key", type=str)
    parser.add_argument("--wandb-run-group", type=str)

    args = parser.parse_args()

    grid = _parse_grid_args(args.grid)
    variant_specs: Iterable[Dict[str, Any]] = []
    if args.variant_file is not None:
        variant_specs = _load_variant_specs(args.variant_file)

    variants_per_dataset: Dict[str, List[PreprocessConfig]] = {}
    for dataset in args.datasets:
        canonical = resolve_dataset_name(dataset)
        base_config = VELOVI_DATASETS[canonical].preprocess or PreprocessConfig()
        variants_per_dataset[dataset] = _build_variants(
            base_config=base_config,
            include_default=not args.skip_default,
            grid=grid,
            variant_specs=variant_specs,
        )

    dataset_mapping = _build_dataset_mapping(args.datasets, variants_per_dataset)

    training_config = _build_training_config(args)
    runner = VELOVIImprovementRunner(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        config=training_config,
        dataset_configs=dataset_mapping,
    )
    results = runner.run(dataset_names=list(dataset_mapping.keys()))

    summary_path = args.output_dir / "preprocess_sweep_summary.csv"
    results.to_csv(summary_path, index=False)

    metric = args.selection_metric
    if metric not in results.columns:
        raise ValueError(
            f"Selection metric `{metric}` not found in results columns: {list(results.columns)}"
        )
    metric_series = results[metric].dropna()
    if metric_series.empty:
        raise ValueError(f"No valid values for selection metric `{metric}`; cannot rank variants.")

    if args.selection_mode == "max":
        best_idx = metric_series.idxmax()
    else:
        best_idx = metric_series.idxmin()
    best_row = results.loc[best_idx]

    print("Preprocess sweep completed.")
    print(f"Summary saved to: {summary_path}")
    print("Top variant:")
    summary_cols = [
        "dataset",
        "dataset_variant",
        "preprocess",
        "preprocess_signature",
        metric,
    ]
    available_cols = [c for c in summary_cols if c in results.columns]
    print(best_row[available_cols].to_string())


if __name__ == "__main__":
    main()
