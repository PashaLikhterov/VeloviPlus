import wandb
import os
import subprocess, sys

def main():
    wandb.init(project="RNA-Velocity", settings=wandb.Settings(init_timeout=600))
    cfg = wandb.config
    # Build the python command using cfg values
    cmd = [
        "python",
        "-m",
        "scvi.experimental.velovi_improvements.runner",
        "/gpfs0/bgu-ofircohen/users/likhtepi/files_adata",
        "--datasets", "gazit",
        "--disable-checkpoints",
        "--output-dir", f"/gpfs0/bgu-ofircohen/users/likhtepi/proj/scvi-tools/results/gazit_sweep/seed{wandb.run.id}",
        "--warmup-epochs", str(cfg.warmup_epochs),
        "--total-epochs", str(cfg.total_epochs),
        "--batch-size", str(cfg.batch_size),
        "--latent-dim", "10",
        "--hidden-dim", "256",
        "--gnn-hidden-dim", "128",
        "--gnn-dropout", "0.1",
        "--num-workers", "0",
        "--gnn-neighbor-source", "both",
        "--gnn-attention",
        "--gnn-gate",
        "--enable-gnn-latent",
        "--velocity-laplacian-weight", "0.05",
        "--velocity-angle-weight", "0.02",
        "--stream-embed", "umap",
        "--enable-transformer-refinement",
        "--transformer-epochs", str(cfg.transformer_epochs),
        "--transformer-hidden-dim", "256",
        "--transformer-layers", "3",
        "--transformer-heads", "8",
        "--transformer-dropout", "0.1",
        "--transformer-batch-size", "128",
        "--transformer-learning-rate", str(cfg.transformer_learning_rate),
        "--transformer-weight-smooth", str(cfg.transformer_weight_smooth),
        "--transformer-weight-smooth-same", str(cfg.transformer_weight_smooth_same),
        "--transformer-weight-boundary-align", str(cfg.transformer_weight_boundary_align),
        "--transformer-weight-boundary-contrast", str(cfg.transformer_weight_boundary_contrast),
        "--transformer-weight-direction", str(cfg.transformer_weight_direction),
        "--transformer-weight-celltype", str(cfg.transformer_weight_celltype),
        "--transformer-weight-celltype-dir", str(cfg.transformer_weight_celltype_dir),
        "--transformer-weight-celltype-mag", "0.0",
        "--transformer-weight-alignment", "1.0",
        "--transformer-aux-cluster-loss-weight", "0.2",
        "--transformer-neighbor-max-distance", str(cfg.transformer_neighbor_max_distance),
        "--transformer-max-neighbors", str(cfg.transformer_max_neighbors),
        "--scvelo-n-jobs", "32",
        "--latent-metric-n-neighbors", str(cfg.latent_metric_n_neighbors),
        "--checkpoint-dir", "/gpfs0/bgu-ofircohen/users/likhtepi/proj/scvi-tools/checkpoints_sweep",
        "--use-wandb",
        "--wandb-project", "RNA-Velocity",
        "--wandb-run-group", "gazit_sweep",
        "--plot-results",
    ]


    result = subprocess.run(" ".join(cmd), shell=True, check=True)
    if result.returncode != 0:
        print("Training command failed with code", result.returncode, file=sys.stderr)
        sys.exit(result.returncode)

if __name__ == "__main__":
    main()
