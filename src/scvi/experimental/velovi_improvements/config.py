from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional, Tuple

import numpy as np

from scvi.external.velovi import LatentGraphConfig


@dataclass
class StreamEmbeddingResult:
    """Container for velocities projected into plotting space."""

    embedding: Optional[np.ndarray]
    components: Optional[np.ndarray]
    projection: Optional[np.ndarray]


@dataclass
class TrainingConfig:
    warmup_epochs: int = 100
    total_epochs: int = 400
    batch_size: int = 256
    n_latent: int = 10
    n_hidden: int = 256
    n_layers: int = 1
    dropout_rate: float = 0.1
    baseline_encoder: Literal["mlp", "transformer"] = "mlp"
    transformer_encoder_hidden_dim: int = 256
    transformer_encoder_layers: int = 2
    transformer_encoder_heads: int = 4
    transformer_encoder_dropout: float = 0.1
    transformer_encoder_max_neighbors: Optional[int] = None
    transformer_encoder_neighbor_weight: float = 0.0
    latent_graph: LatentGraphConfig = field(default_factory=LatentGraphConfig)
    gnn_hidden_dim: int = 128
    gnn_dropout_rate: float = 0.1
    num_workers: int = 0
    enable_latent_smoothing: bool = True
    enable_gnn: bool = True
    enable_gnn_latent_smoothing: bool = False
    gnn_epochs: Optional[int] = None
    gnn_batch_size: Optional[int] = None
    gnn_continuity_weight: float = 0.0
    gnn_continuity_horizon: float = 1.0
    produce_plots: bool = False
    gnn_neighbor_source: Literal["expression", "latent", "both"] = "both"
    gnn_use_attention: bool = False
    gnn_use_gate: bool = False
    gnn_use_residual: bool = True
    gnn_use_differences: bool = True
    velocity_laplacian_weight: float = 0.0
    velocity_angle_weight: float = 0.0
    velocity_angle_eps: float = 1e-6
    stream_embed_method: Literal["pca", "umap"] = "pca"
    stream_embed_pca_components: int = 8
    stream_embed_standardize: bool = True
    stream_umap_neighbors: int = 30
    stream_umap_min_dist: float = 0.3
    stream_umap_spread: float = 1.0
    enable_transformer_refinement: bool = False
    enable_scvelo_dynamic: bool = True
    transformer_epochs: int = 10
    transformer_hidden_dim: int = 128
    transformer_layers: int = 2
    transformer_heads: int = 4
    transformer_dropout: float = 0.1
    transformer_batch_size: int = 128
    transformer_learning_rate: float = 1e-3
    transformer_weight_smooth: float = 0.1
    transformer_weight_direction: float = 0.1
    transformer_weight_celltype: float = 0.0
    transformer_weight_celltype_dir: float = 0.0
    transformer_weight_celltype_mag: float = 0.0
    transformer_weight_alignment: float = 1.0
    transformer_weight_supervised: float = 0.0
    transformer_weight_smooth_same: float = 0.0
    transformer_weight_boundary_align: float = 0.0
    transformer_weight_boundary_contrast: float = 0.0
    transformer_celltype_penalty: Literal["cosine", "mse", "both"] = "cosine"
    transformer_aux_cluster_loss_weight: float = 0.0
    transformer_neighbor_max_distance: Optional[float] = None
    transformer_max_neighbors: Optional[int] = None
    transformer_residual_to_baseline: bool = True
    transformer_use_tivelo: bool = False
    transformer_tivelo_resolution: float = 0.6
    transformer_tivelo_threshold: float = 0.1
    transformer_tivelo_threshold_trans: float = 1.0
    tivelo_prior_strength: float = 0.4
    tivelo_loss_fun: str = "mse"
    tivelo_only_spliced: bool = False
    tivelo_constrain: bool = True
    tivelo_alpha1: float = 1.0
    tivelo_alpha2: float = 0.1
    tivelo_batch_size: int = 1024
    tivelo_epochs: int = 100
    tivelo_filter_genes: bool = True
    tivelo_show_fig: bool = False
    tivelo_show_dti: bool = False
    tivelo_adjust_dti: bool = False
    tivelo_tree_gene: Optional[str] = "Cplx2"
    load_pretrained: bool = False
    use_checkpoints: bool = True
    checkpoint_dir: Optional[str] = None
    scvelo_dynamics_n_jobs: int = 32
    use_wandb: bool = False
    wandb_project: str = "RNA-Velocity"
    wandb_entity: Optional[str] = None
    wandb_api_key: Optional[str] = None
    wandb_run_group: Optional[str] = None
    latent_metric_n_neighbors: int = 15
    use_gpu: bool = True
    skip_preprocess: bool = False
    save_figures_locally: bool = True
    use_tivelo_cluster_edges: bool = False
    use_paga_cluster_edges: bool = False
    use_paga_for_refinements: bool = False
    paga_prior_threshold: float = 0.3
    paga_min_alignment: float = 0.0
    paga_projection_alpha: float = 0.0
