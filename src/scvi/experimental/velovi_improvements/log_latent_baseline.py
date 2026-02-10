from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from lightning.pytorch.callbacks import Callback

from .config import TrainingConfig
from .datasets import DatasetConfig, VELOVI_DATASETS, load_dataset, resolve_dataset_name
from .training import (
    infer_cell_type_labels,
    log_training_history,
    start_wandb_run,
)
from scvi.external.velovi import VELOVI


def _get_dataset_config(name: str) -> DatasetConfig:
    canonical = resolve_dataset_name(name)
    if canonical not in VELOVI_DATASETS:
        raise KeyError(f"Unknown dataset '{name}'. Available: {', '.join(sorted(VELOVI_DATASETS))}")
    return VELOVI_DATASETS[canonical]


def _build_color_map(adata, dataset_config: DatasetConfig) -> Optional[Dict[str, str]]:
    color_map: Optional[Dict[str, str]] = None
    key = dataset_config.plot_color_key
    if key and key in adata.obs:
        palette_key = f"{key}_colors"
        if palette_key in adata.uns:
            raw_colors = adata.uns[palette_key]
            unique_labels = np.unique(adata.obs[key].astype(str))
            if isinstance(raw_colors, dict):
                color_map = {str(k): v for k, v in raw_colors.items()}
            else:
                color_map = {
                    str(label): raw_colors[idx % len(raw_colors)]
                    for idx, label in enumerate(unique_labels)
                }
    return color_map


def _persist_image(src: Path, dst_dir: Path, filename: str) -> Path:
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst_path = dst_dir / filename
    shutil.copy2(src, dst_path)
    return dst_path


def _latent_pca2(latent: np.ndarray) -> tuple[np.ndarray, tuple[float, float] | None]:
    """Compute a 2D PCA of the latent matrix.

    Returns (coords, (var_exp_pc1, var_exp_pc2)) when available.
    """
    X = np.asarray(latent, dtype=np.float32)
    X = X - X.mean(axis=0, keepdims=True)
    try:
        from sklearn.decomposition import PCA

        pca = PCA(n_components=2, random_state=0)
        coords = pca.fit_transform(X)
        ev = pca.explained_variance_ratio_
        var_exp = (float(ev[0]), float(ev[1])) if ev is not None and len(ev) >= 2 else None
        return coords.astype(np.float32), var_exp
    except ModuleNotFoundError:
        # Fallback to SVD
        try:
            _, _, vt = np.linalg.svd(X, full_matrices=False)
            components = vt[:2]
            coords = X @ components.T
            return coords.astype(np.float32), None
        except np.linalg.LinAlgError:
            return np.zeros((X.shape[0], 2), dtype=np.float32), None


def _plot_latent(
    embedding: np.ndarray,
    labels: np.ndarray | None,
    title: str,
    color_map: Dict[str, str] | None = None,
    elbo_value: Optional[float] = None,
    suffix_text: Optional[str] = None,
) -> Path:
    fig, ax = plt.subplots(figsize=(5, 4))
    scatter_kwargs = dict(s=6, alpha=0.6, linewidths=0)
    if labels is None:
        ax.scatter(embedding[:, 0], embedding[:, 1], **scatter_kwargs)
        centroid_distance = 0.0
    else:
        unique = np.unique(labels)
        centroids = []
        for lab in unique:
            mask = labels == lab
            color = color_map.get(str(lab)) if color_map else None
            centroids.append(embedding[mask].mean(axis=0))
            ax.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                label=str(lab),
                color=color,
                **scatter_kwargs,
            )
        if unique.size <= 20:
            ax.legend(loc="best", fontsize=7, ncol=2)
        centroids = np.asarray(centroids, dtype=np.float32)
        dists = np.linalg.norm(centroids[:, None, :] - centroids[None, :, :], axis=-1)
        # Unique pairs sum
        centroid_distance = float(np.triu(dists, 1).sum())
    ax.set_xlabel("Latent-1")
    ax.set_ylabel("Latent-2")
    ax.set_xticks([])
    ax.set_yticks([])
    subtitle = f"Centroid Sum Dist: {centroid_distance:.2f}"
    if elbo_value is not None:
        subtitle += f" | ELBO: {elbo_value:.2f}"
    if suffix_text:
        subtitle += f" | {suffix_text}"
    ax.set_title(f"{title}\n{subtitle}")
    fig.tight_layout()
    tmp_dir = Path(tempfile.mkdtemp(prefix="velovi_latent_"))
    out_path = tmp_dir / f"{title.replace(' ', '_')}.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _extract_last_elbo(model) -> float | None:
    history = getattr(model, "history_", None)
    if history is None:
        history = getattr(model, "history", None)
    if history is None:
        return None

    def _get(history_obj, key):
        if history_obj is None:
            return None
        try:
            series = history_obj[key]
        except Exception:
            return None
        try:
            if hasattr(series, "iloc"):
                if len(series) == 0:
                    return None
                return float(series.iloc[-1])
            if isinstance(series, (list, tuple)):
                if not series:
                    return None
                return float(series[-1])
            return float(series)
        except Exception:
            return None

    for candidate in ("elbo_train", "elbo_validation", "train_loss_epoch"):
        value = _get(history, candidate)
        if value is not None:
            return value
    return None


def _extract_loss_components(model) -> Dict[str, Optional[float]]:
    """Extract recon, KL, ELBO (best-effort) from history."""
    history = getattr(model, "history_", None)
    if history is None:
        history = getattr(model, "history", None)
    result: Dict[str, Optional[float]] = {"recon": None, "kl": None, "elbo": None}
    if history is None:
        return result

    def _pick(keys):
        for k in keys:
            if k in history and history[k] is not None and len(history[k]) > 0:
                try:
                    series = history[k]
                    if hasattr(series, "iloc"):
                        return float(series.iloc[-1])
                    if isinstance(series, (list, tuple)):
                        return float(series[-1])
                    return float(series)
                except Exception:
                    continue
        return None

    # Populate result dict using common key names
    result["elbo"] = _pick(["elbo_train", "elbo_validation", "train_loss_epoch"])  # fallback
    result["recon"] = _pick([
        "reconstruction_loss_train",
        "reconstruction_loss_validation",
        "reconstruction_loss",
        "reconstruction_train",
    ])
    result["kl"] = _pick([
        "kl_local_train",
        "kl_local_validation",
        "kl_local",
        "kl_loss_train",
        "kl_train",
    ])
    return result


class LatentLoggingCallback(Callback):
    def __init__(
        self,
        model: VELOVI,
        adata,
        dataset_config: DatasetConfig,
        log_interval: int,
        total_epochs: int,
        output_dir: Path,
        dataset_name: str,
        wandb_run,
    ) -> None:
        super().__init__()
        self.model = model
        self.adata = adata
        self.dataset_config = dataset_config
        self.log_interval = max(1, log_interval)
        self.total_epochs = total_epochs
        self.output_dir = output_dir
        self.dataset_name = dataset_name
        self.wandb_run = wandb_run
        self.cell_types, _ = infer_cell_type_labels(adata, dataset_config)
        self.color_map = _build_color_map(adata, dataset_config)

    def _should_log(self, epoch: int) -> bool:
        if epoch == self.total_epochs:
            return True
        return epoch % self.log_interval == 0

    def _log_image(self, path: Path, kind: str, epoch: int) -> None:
        if self.output_dir is not None:
            figures_dir = self.output_dir / "figures" / "latent_snapshots"
            filename = f"{self.dataset_name}_epoch{epoch:04d}_{kind}.png"
            _persist_image(path, figures_dir, filename)
        if self.wandb_run is not None:
            import wandb

            self.wandb_run.log({f"{kind}/{epoch:03d}": wandb.Image(str(path))}, step=epoch)

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        epoch = trainer.current_epoch + 1
        if not self._should_log(epoch):
            return
        # Ensure latent vectors are computed on a consistent device
        prev_flag = self.model.is_trained_
        if not prev_flag:
            self.model.is_trained = True
        # Temporarily move module to CPU to avoid device mismatches during logging
        has_module = hasattr(self.model, "module")
        prev_device = None
        if has_module:
            try:
                first_param = next(self.model.module.parameters())
                prev_device = first_param.device
            except StopIteration:
                prev_device = None
            if prev_device is not None and prev_device.type != "cpu":
                self.model.module.to("cpu")
        try:
            latent = self.model.get_latent_representation(batch_size=256)
            latent = np.asarray(latent, dtype=np.float32)
        finally:
            if has_module and prev_device is not None and prev_device.type != "cpu":
                self.model.module.to(prev_device)
            self.model.is_trained = prev_flag
        embedding, _ = _latent_pca2(latent)
        losses = _extract_loss_components(self.model)
        logged = getattr(trainer, "logged_metrics", None)
        if logged:
            if losses["recon"] is None and "reconstruction_loss_train" in logged:
                losses["recon"] = float(logged["reconstruction_loss_train"].cpu())
            if losses["kl"] is None and "kl_local_train" in logged:
                losses["kl"] = float(logged["kl_local_train"].cpu())
            if losses["elbo"] is None and "elbo_train" in logged:
                losses["elbo"] = float(logged["elbo_train"].cpu())
        recon_v = losses.get("recon")
        kl_v = losses.get("kl")
        suffix_parts = []
        if recon_v is not None:
            suffix_parts.append(f"Recon {recon_v:.2f}")
        if kl_v is not None:
            suffix_parts.append(f"KL {kl_v:.2f}")
        suffix = " | ".join(suffix_parts) if suffix_parts else None

        title = f"{self.dataset_name} Epoch {epoch}"
        plot_path = _plot_latent(
            embedding,
            self.cell_types if self.cell_types is not None else None,
            title=title,
            color_map=self.color_map,
            elbo_value=losses.get("elbo"),
            suffix_text=suffix,
        )
        self._log_image(plot_path, "latent", epoch)

        if recon_v is not None:
            recon_path = _plot_latent(
                embedding,
                self.cell_types if self.cell_types is not None else None,
                title=f"{self.dataset_name} Epoch {epoch} (Recon)",
                color_map=self.color_map,
                elbo_value=None,
                suffix_text=f"Recon {recon_v:.2f}",
            )
            self._log_image(recon_path, "latent_recon", epoch)

        kl_v = losses.get("kl")
        if kl_v is not None:
            kl_path = _plot_latent(
                embedding,
                self.cell_types if self.cell_types is not None else None,
                title=f"{self.dataset_name} Epoch {epoch} (KL)",
                color_map=self.color_map,
                elbo_value=None,
                suffix_text=f"KL {kl_v:.2f}",
            )
            self._log_image(kl_path, "latent_kl", epoch)

        if self.wandb_run is not None:
            logs = {}
            if losses.get("elbo") is not None:
                logs["loss/elbo"] = losses["elbo"]
            if recon_v is not None:
                logs["loss/reconstruction"] = recon_v
            if kl_v is not None:
                logs["loss/kl"] = kl_v
            if logs:
                self.wandb_run.log(logs, step=epoch)


def _plot_training_summary(history: dict | None, title: str, output_path: Path) -> Optional[Path]:
    """Render a compact summary figure for training losses from a scvi-style history dict."""
    try:
        if history is None or not isinstance(history, dict):
            return None
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
    except Exception:
        return None

    result["elbo"] = _pick(["elbo_train", "elbo_validation", "train_loss_epoch"])
    result["recon"] = _pick([
        "reconstruction_loss_train",
        "reconstruction_loss_validation",
        "reconstruction_loss",
        "reconstruction_train",
    ])
    result["kl"] = _pick([
        "kl_local_train",
        "kl_local_validation",
        "kl_local",
        "kl_loss_train",
        "kl_train",
    ])
    return result


def main():
    parser = argparse.ArgumentParser(description="Log VELOVI latent space during baseline training.")
    parser.add_argument("data_dir", type=Path, help="Root directory containing datasets.")
    parser.add_argument("--dataset", type=str, default="pancreas_endocrinogenesis")
    parser.add_argument("--output-dir", type=Path, default=Path("./results/velovi_latent_logging"))
    parser.add_argument("--total-epochs", type=int, default=150)
    parser.add_argument("--log-interval", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--latent-dim", type=int, default=10)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--skip-preprocess", action="store_true")
    parser.add_argument("--warmup-epochs", type=int, default=20, help="KL warmup epochs for VELOVI.")
    parser.add_argument("--wandb-project", type=str, default="RNA-Velocity")
    parser.add_argument("--wandb-run-group", type=str, default=None)
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-api-key", type=str, default=None)
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--plot-basis", type=str, default=None)
    args = parser.parse_args()

    dataset_config = _get_dataset_config(args.dataset)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    preprocess_active = dataset_config.preprocess_enabled and not args.skip_preprocess

    adata = load_dataset(
        args.data_dir,
        dataset_config,
        apply_preprocess=preprocess_active,
    )
    VELOVI.setup_anndata(
        adata,
        spliced_layer=dataset_config.spliced_layer,
        unspliced_layer=dataset_config.unspliced_layer,
    )
    model = VELOVI(
        adata,
        n_hidden=args.hidden_dim,
        n_latent=args.latent_dim,
        dropout_rate=0.1,
    )

    config = TrainingConfig(
        warmup_epochs=args.warmup_epochs,
        total_epochs=args.total_epochs,
        batch_size=args.batch_size,
        n_hidden=args.hidden_dim,
        n_latent=args.latent_dim,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_group=args.wandb_run_group,
    )
    config.stream_embed_standardize = True
    config.stream_embed_method = "pca"

    wandb_run = None
    if args.use_wandb:
        if args.wandb_api_key and "WANDB_API_KEY" not in os.environ:
            os.environ["WANDB_API_KEY"] = args.wandb_api_key
        wandb_run = start_wandb_run(config=config, dataset_name=args.dataset)

    total = args.total_epochs
    interval = max(1, args.log_interval)
    device = "gpu" if torch.cuda.is_available() else "cpu"

    latent_callback = LatentLoggingCallback(
        model=model,
        adata=adata,
        dataset_config=dataset_config,
        log_interval=interval,
        total_epochs=total,
        output_dir=args.output_dir,
        dataset_name=args.dataset,
        wandb_run=wandb_run,
    )

    model.train(
        max_epochs=total,
        batch_size=args.batch_size,
        accelerator=device,
        devices="auto",
        early_stopping=False,
        plan_kwargs={"n_epochs_kl_warmup": args.warmup_epochs},
        callbacks=[latent_callback],
    )

    history = getattr(model, "history_", None)
    if history is None:
        history = getattr(model, "history", None)
    log_training_history(wandb_run, history, "baseline/train_latent")
    # Training summary figure
    summary_path = _plot_training_summary(
        history,
        title=f"{args.dataset} – Baseline Training Summary",
        output_path=args.output_dir / f"{args.dataset}_train_summary.png",
    )
    if summary_path is not None and wandb_run is not None:
        import wandb

        wandb_run.log({"train/summary": wandb.Image(str(summary_path))})
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
