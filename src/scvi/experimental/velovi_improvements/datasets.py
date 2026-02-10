from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple, List, Any
import inspect
import warnings

import numpy as np
import scanpy as sc

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    yaml = None

_DATASET_CONFIG_DIR = Path(__file__).with_name("dataset_configs")


try:  # SciPy >=1.14 removed .A/.A1 convenience attributes from sparse matrices
    import scipy.sparse as sp

    if not hasattr(sp.spmatrix, "A"):
        sp.spmatrix.A = property(lambda self: self.toarray())  # type: ignore[attr-defined]
    if not hasattr(sp.spmatrix, "A1"):
        sp.spmatrix.A1 = property(lambda self: self.toarray().ravel())  # type: ignore[attr-defined]
except Exception:  # pragma: no cover - best effort compatibility shim
    pass


@dataclass
class PreprocessConfig:
    min_counts: int = 1
    min_counts_u: int = 0
    min_cells: int = 0
    min_cells_u: int = 0
    min_shared_counts: int = 20
    n_top_genes: int = 2000
    flavor: str = "seurat"
    log1p: bool = True
    scale: bool = False
    n_pcs: int = 30
    n_neighbors: int = 30
    name: Optional[str] = None

    def signature(self) -> str:
        """Construct a short string that uniquely identifies this configuration."""
        parts = [
            f"mc{self.min_counts}",
            f"mcu{self.min_counts_u}",
            f"mcell{self.min_cells}",
            f"mcellu{self.min_cells_u}",
            f"msc{self.min_shared_counts}",
            f"hvg{self.n_top_genes}",
            f"pcs{self.n_pcs}",
            f"nn{self.n_neighbors}",
            f"flv{self.flavor}",
        ]
        if self.log1p:
            parts.append("log1p")
        if self.scale:
            parts.append("scale")
        return "_".join(parts)

    def display_name(self) -> str:
        """Human readable label for logging/reporting."""
        return self.name or self.signature()

    def apply(self, adata):
        import scvelo as scv

        fn = scv.pp.filter_and_normalize
        valid_args = set(inspect.signature(fn).parameters.keys())

        candidate_kwargs = {
            "min_counts": self.min_counts,
            "min_counts_u": self.min_counts_u,
            "min_cells": self.min_cells,
            "min_cells_u": self.min_cells_u,
            "min_shared_counts": self.min_shared_counts,
            "n_top_genes": self.n_top_genes,
            "flavor": self.flavor,
            "log1p": self.log1p,
        }
        kwargs = {k: v for k, v in candidate_kwargs.items() if k in valid_args}

        try:
            fn(adata, **kwargs)
        except TypeError as exc:
            filtered_kwargs = {
                k: v
                for k, v in kwargs.items()
                if k in valid_args and k not in {"log1p"}
            }
            warnings.warn(
                f"Retrying filter_and_normalize without unsupported arguments due to: {exc}",
                RuntimeWarning,
            )
            fn(adata, **filtered_kwargs)

        if self.scale:
            sc.pp.scale(adata)

        sc.tl.pca(adata, n_comps=self.n_pcs, svd_solver="arpack")
        sc.pp.neighbors(adata, n_neighbors=self.n_neighbors, n_pcs=self.n_pcs)
        scv.pp.moments(adata, n_pcs=self.n_pcs, n_neighbors=self.n_neighbors)
        try:
            sc.tl.umap(adata, n_components=2)
        except ModuleNotFoundError:
            warnings.warn(
                "UMAP is not installed; skipping `sc.tl.umap` during preprocessing.",
                RuntimeWarning,
            )
        except Exception as exc:
            warnings.warn(
                f"Failed to compute UMAP during preprocessing: {exc}",
                RuntimeWarning,
            )


@dataclass
class DatasetConfig:
    name: str
    scvelo_loader: Optional[str]
    filename: str
    description: str
    spliced_layer: str = "Ms"
    unspliced_layer: str = "Mu"
    group_key: Optional[str] = None
    plot_color_key: Optional[str] = None
    embedding_key: Optional[str] = "X_umap"
    plot_basis: Optional[str] = None
    pseudotime_key: Optional[str] = None
    fucci_key: Optional[str] = None
    cell_cycle_rad_key: Optional[str] = None
    cluster_edges: Optional[List[Tuple[str, str]]] = None
    alternative_filenames: Tuple[str, ...] = ()
    aliases: Tuple[str, ...] = ()
    preprocess_enabled: bool = True
    preprocess_config: Optional[PreprocessConfig] = None

    @property
    def celltype_key(self) -> Optional[str]:
        return self.group_key

    @property
    def preprocess(self) -> Optional[PreprocessConfig]:
        return self.preprocess_config


DEFAULT_PREPROCESS = PreprocessConfig()
_PREPROCESS_FIELD_NAMES = {f.name for f in fields(PreprocessConfig)}


def _infer_plot_basis(embedding_key: Optional[str], explicit: Optional[str]) -> Optional[str]:
    if explicit:
        return explicit
    if embedding_key is None:
        return None
    if embedding_key.lower().startswith("x_"):
        return embedding_key[2:].lower()
    return embedding_key.lower()


def _parse_preprocess_entry(entry: Any) -> Tuple[bool, Optional[PreprocessConfig]]:
    if entry is None:
        return True, None
    if isinstance(entry, bool):
        return bool(entry), None
    if isinstance(entry, dict):
        enabled = bool(entry.get("enabled", True))
        params_source = entry.get("params")
        if params_source is None:
            candidates = {k: v for k, v in entry.items() if k != "enabled"}
        else:
            candidates = params_source or {}
        params: Dict[str, Any] = {}
        for key, value in candidates.items():
            if key in _PREPROCESS_FIELD_NAMES:
                params[key] = value
        cfg = PreprocessConfig(**params) if params else None
        return enabled, cfg
    raise ValueError(f"Invalid preprocess specification: {entry!r}")


def _dataset_config_from_yaml(data: Dict[str, Any], *, source: Path) -> DatasetConfig:
    files = data.get("files") or []
    if not files:
        raise ValueError(f"Dataset config {source} must specify at least one file path.")
    filename = str(files[0])
    alternatives = tuple(str(path) for path in files[1:])
    preprocess_enabled, preprocess_cfg = _parse_preprocess_entry(data.get("preprocess"))
    cluster_edges = data.get("cluster_edges")
    if cluster_edges is not None:
        cluster_edges = [(str(src), str(dst)) for src, dst in cluster_edges]

    embedding_key = data.get("embedding_key", "X_umap")
    plot_basis = _infer_plot_basis(embedding_key, data.get("plot_basis"))
    plot_color_key = data.get("plot_color_key") or data.get("group_key")

    return DatasetConfig(
        name=str(data["name"]),
        scvelo_loader=data.get("scvelo_loader"),
        filename=filename,
        description=data.get("description", ""),
        group_key=data.get("group_key"),
        plot_color_key=plot_color_key,
        embedding_key=embedding_key,
        plot_basis=plot_basis,
        pseudotime_key=data.get("pseudotime_key"),
        fucci_key=data.get("fucci_key"),
        cell_cycle_rad_key=data.get("cell_cycle_rad_key"),
        cluster_edges=cluster_edges,
        alternative_filenames=alternatives,
        aliases=tuple(str(alias) for alias in data.get("aliases", []) if alias),
        preprocess_enabled=preprocess_enabled,
        preprocess_config=preprocess_cfg,
    )


def _load_dataset_registry() -> Tuple[Dict[str, DatasetConfig], Dict[str, str]]:
    if yaml is None:
        raise ModuleNotFoundError(
            "PyYAML is required to read dataset configs. Install it with `pip install pyyaml`."
        )
    canonical: Dict[str, DatasetConfig] = {}
    alias_lookup: Dict[str, str] = {}
    for config_path in sorted(_DATASET_CONFIG_DIR.glob("*.yaml")):
        with config_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        cfg = _dataset_config_from_yaml(raw, source=config_path)
        if cfg.name in canonical and canonical[cfg.name] is not cfg:
            warnings.warn(f"Duplicate dataset definition for {cfg.name}; overriding previous entry.")
        canonical[cfg.name] = cfg
        for alias in cfg.aliases:
            if not alias or alias == cfg.name:
                continue
            if alias in alias_lookup and alias_lookup[alias] != cfg.name:
                warnings.warn(
                    f"Dataset alias '{alias}' defined in {config_path.name} overwrites an existing entry.",
                    RuntimeWarning,
                )
            alias_lookup[alias] = cfg.name
    return canonical, alias_lookup


VELOVI_DATASETS, DATASET_ALIASES = _load_dataset_registry()


def _candidate_relative_paths(config: DatasetConfig) -> Tuple[Path, ...]:
    """Return unique relative paths that may contain the dataset."""
    primary = Path(config.filename)
    extras = tuple(Path(name) for name in config.alternative_filenames)
    seen: Dict[str, Path] = {}
    for path in (primary, *extras):
        key = str(path)
        if key not in seen:
            seen[key] = path
    return tuple(seen.values())


def _search_by_basename(data_dir: Path, candidates: Iterable[Path]) -> Optional[Path]:
    """Search recursively for files that match any candidate basename."""
    basenames = {path.name.lower() for path in candidates}
    if not basenames:
        return None
    for suffix in ("*.h5ad", "*.loom"):
        for path in data_dir.glob(suffix):
            if path.name.lower() in basenames:
                return path
    for suffix in ("*.h5ad", "*.loom"):
        for path in data_dir.rglob(suffix):
            if path.name.lower() in basenames:
                return path
    return None


def _download_dataset_via_scvelo(data_path: Path, config: DatasetConfig) -> Path:
    if not config.scvelo_loader:
        raise FileNotFoundError(
            f"Dataset `{config.name}` is not available locally and no scvelo loader is configured. "
            f"Place one of {config.alternative_filenames or (config.filename,)} inside {data_path.parent}."
        )

    import scvelo as scv

    loader = getattr(scv.datasets, config.scvelo_loader, None)
    if loader is None:
        raise ValueError(f"Unknown scvelo dataset loader `{config.scvelo_loader}`.")
    adata = loader()
    _harmonize_velocity_layers(adata, config)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    if data_path.suffix == ".loom":
        adata.write_loom(data_path)
    else:
        adata.write_h5ad(data_path)
    return data_path


def ensure_dataset_file(data_dir: Path, config: DatasetConfig, force: bool = False) -> Path:
    data_dir = Path(data_dir).expanduser()
    candidate_relative = _candidate_relative_paths(config)
    candidate_paths = [(data_dir / rel).expanduser() for rel in candidate_relative]

    if not force:
        for path in candidate_paths:
            if path.exists():
                return path
        found = _search_by_basename(data_dir, candidate_relative)
        if found is not None and found.exists():
            return found

    target_path = candidate_paths[0]
    if target_path.exists() and not force:
        return target_path
    return _download_dataset_via_scvelo(target_path, config)


def _read_anndata(data_path: Path, reader: str):
    suffix = data_path.suffix.lower()
    if reader == "auto":
        if suffix == ".h5ad":
            return sc.read_h5ad(data_path)
        if suffix == ".loom":
            try:
                return sc.read_loom(data_path, sparse=True)
            except ModuleNotFoundError as exc:
                raise ModuleNotFoundError(
                    "Reading .loom files requires the `loompy` package. "
                    "Install it with `pip install loompy` inside your environment."
                ) from exc
        raise ValueError(f"Unsupported file format for {data_path}.")
    if reader == "loom":
        try:
            return sc.read_loom(data_path, sparse=True)
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Reading .loom files requires the `loompy` package. "
                "Install it with `pip install loompy` inside your environment."
            ) from exc
    if reader == "h5ad":
        return sc.read_h5ad(data_path)
    raise ValueError(f"Unknown reader `{reader}` for dataset `{data_path}`.")


def _harmonize_velocity_layers(adata, config: DatasetConfig):
    spliced_fallback = (config.spliced_layer, "Ms", "spliced", "X_spliced", "spliced_counts")
    unspliced_fallback = (config.unspliced_layer, "Mu", "unspliced", "X_unspliced", "unspliced_counts")

    def _ensure_layer(target_name: str, candidates: Tuple[str, ...]):
        if target_name in adata.layers:
            return
        for cand in candidates:
            if cand in adata.layers:
                adata.layers[target_name] = adata.layers[cand]
                return
        raise KeyError(f"Unable to locate layer for {target_name} in dataset {config.name}.")

    _ensure_layer(config.spliced_layer, spliced_fallback)
    _ensure_layer(config.unspliced_layer, unspliced_fallback)


def _standardize_layer(adata, layer_name: str):
    """Ensure layer is dense, float32, and shaped (n_obs, n_vars)."""
    layer = adata.layers[layer_name]
    if hasattr(layer, "toarray"):
        layer = layer.toarray()
    else:
        layer = np.asarray(layer)

    if layer.ndim != 2:
        raise ValueError(
            f"Layer `{layer_name}` must be 2D after conversion, found shape {layer.shape}."
        )

    if layer.shape == (adata.n_vars, adata.n_obs):
        layer = layer.T
    elif layer.shape != (adata.n_obs, adata.n_vars):
        raise ValueError(
            f"Layer `{layer_name}` has unexpected shape {layer.shape}; "
            f"expected ({adata.n_obs}, {adata.n_vars})."
        )

    adata.layers[layer_name] = layer.astype(np.float32, copy=False)


def load_dataset(
    data_dir: Path,
    config: DatasetConfig,
    apply_preprocess: bool = True,
):
    """Load dataset from disk, applying optional preprocessing hook."""

    data_dir = Path(data_dir).expanduser()
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Provided data directory {data_dir.resolve(strict=False)} does not exist."
        )

    data_path = ensure_dataset_file(data_dir, config)
    if data_path.is_dir():
        candidates = list(data_path.glob("*.h5ad")) + list(data_path.glob("*.loom"))
        if not candidates:
            raise FileNotFoundError(
                f"No .h5ad or .loom files found inside directory {data_path} for dataset "
                f"`{config.name}`."
            )
        data_path = candidates[0]
    elif not data_path.exists():
        candidate_relatives = _candidate_relative_paths(config)
        fallback = _search_by_basename(data_dir, candidate_relatives)
        if fallback is not None and fallback.exists():
            data_path = fallback
        else:
            names = ", ".join(p.name for p in candidate_relatives)
            raise FileNotFoundError(
                f"Expected dataset `{config.name}` in {data_dir}. "
                f"Place one of [{names}] in the directory or allow automatic download."
            )

    reader = "auto"
    print(f"[VELOVI][DATA] Using dataset file {data_path} for dataset {config.name}")
    try:
        adata = _read_anndata(data_path, reader)
    except (OSError, ValueError) as exc:
        if config.scvelo_loader is not None:
            warnings.warn(
                f"Failed to read dataset {config.name} at {data_path}: {exc}. "
                "Attempting to re-download via scvelo.",
                RuntimeWarning,
            )
            data_path = _download_dataset_via_scvelo((data_dir / Path(config.filename)).expanduser(), config)
            adata = _read_anndata(data_path, "h5ad")
        else:
            raise

    _harmonize_velocity_layers(adata, config)

    _standardize_layer(adata, config.spliced_layer)
    _standardize_layer(adata, config.unspliced_layer)
    adata.X = adata.layers[config.spliced_layer]

    if apply_preprocess:
        preprocess_cfg = config.preprocess or DEFAULT_PREPROCESS
        preprocess_cfg.apply(adata)
    return adata


def resolve_dataset_name(name: str) -> str:
    """Return the canonical dataset key for a possibly aliased name."""
    return DATASET_ALIASES.get(name, name)


def get_dataset_config(name: str) -> DatasetConfig:
    """Fetch dataset config by canonical or aliased name."""
    canonical = resolve_dataset_name(name)
    if canonical not in VELOVI_DATASETS:
        raise KeyError(f"Unknown dataset '{name}'. Available keys: {', '.join(sorted(VELOVI_DATASETS))}")
    return VELOVI_DATASETS[canonical]


__all__ = [
    "PreprocessConfig",
    "DatasetConfig",
    "DEFAULT_PREPROCESS",
    "VELOVI_DATASETS",
    "DATASET_ALIASES",
    "resolve_dataset_name",
    "get_dataset_config",
    "ensure_dataset_file",
    "load_dataset",
]
