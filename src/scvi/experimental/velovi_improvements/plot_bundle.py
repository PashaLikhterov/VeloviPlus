from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

import scanpy as sc

from .config import TrainingConfig
from .datasets import DatasetConfig


def _safe_key(key: str) -> str:
    return key.replace("/", "_").replace(" ", "_")


def save_plot_bundle(
    *,
    dataset_name: str,
    adata,
    dataset_config: DatasetConfig,
    training_config: TrainingConfig,
    plot_color_key: Optional[str],
    velocities: Dict[str, np.ndarray],
    output_dir: Path,
) -> Tuple[Path, Path, Path]:
    """Persist minimal information needed to reproduce streamline plots.

    Files written under:
      `<output_dir>/velovi_<dataset>/plot_bundle/{adata.h5ad,velocities.npz,metadata.json}`
    """
    bundle_dir = Path(output_dir) / f"velovi_{dataset_name}" / "plot_bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    adata_path = bundle_dir / "adata.h5ad"
    vel_path = bundle_dir / "velocities.npz"
    meta_path = bundle_dir / "metadata.json"

    # Save AnnData with embeddings + obs + needed layers.
    adata_to_save = adata.copy()
    keep_layers = {dataset_config.spliced_layer, dataset_config.unspliced_layer, "Ms", "Mu"}
    for key in list(adata_to_save.layers.keys()):
        if key not in keep_layers:
            del adata_to_save.layers[key]
    if dataset_config.spliced_layer not in adata_to_save.layers and "Ms" in adata_to_save.layers:
        adata_to_save.layers[dataset_config.spliced_layer] = adata_to_save.layers["Ms"]
    if dataset_config.unspliced_layer not in adata_to_save.layers and "Mu" in adata_to_save.layers:
        adata_to_save.layers[dataset_config.unspliced_layer] = adata_to_save.layers["Mu"]

    adata_to_save.write_h5ad(adata_path)

    # Save velocity matrices (float32) in a single compressed archive.
    npz_payload = {}
    for key, value in velocities.items():
        arr = np.asarray(value, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] != adata_to_save.n_obs:
            continue
        npz_payload[_safe_key(key)] = arr
    np.savez_compressed(vel_path, **npz_payload)

    metadata = {
        "dataset": dataset_name,
        "plot_color_key": plot_color_key,
        "dataset_config": {
            "spliced_layer": dataset_config.spliced_layer,
            "unspliced_layer": dataset_config.unspliced_layer,
            "plot_basis": dataset_config.plot_basis,
            "embedding_key": dataset_config.embedding_key,
            "plot_color_key": dataset_config.plot_color_key,
            "celltype_key": dataset_config.celltype_key,
        },
        "training_config": {
            k: v
            for k, v in asdict(training_config).items()
            if isinstance(v, (int, float, str, bool)) or v is None
        },
        "velocity_keys": sorted(npz_payload.keys()),
    }
    meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))
    return adata_path, vel_path, meta_path


def load_plot_bundle(bundle_dir: Path):
    bundle_dir = Path(bundle_dir)
    adata_path = bundle_dir / "adata.h5ad"
    vel_path = bundle_dir / "velocities.npz"
    meta_path = bundle_dir / "metadata.json"
    if not (adata_path.exists() and vel_path.exists() and meta_path.exists()):
        raise FileNotFoundError(
            f"Missing plot bundle files in {bundle_dir}. Expected adata.h5ad, velocities.npz, metadata.json."
        )
    meta = json.loads(meta_path.read_text())
    adata = sc.read_h5ad(adata_path)
    vel_npz = np.load(vel_path)
    velocities = {key: np.asarray(vel_npz[key], dtype=np.float32) for key in vel_npz.files}
    return adata, velocities, meta

