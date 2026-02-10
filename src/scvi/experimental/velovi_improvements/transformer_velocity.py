from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional

try:
    import wandb  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover
    wandb = None

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset


@dataclass
class TransformerConfig:
    n_layers: int = 2
    n_heads: int = 4
    hidden_dim: int = 128
    dropout: float = 0.1
    batch_size: int = 128
    epochs: int = 10
    learning_rate: float = 1e-3
    weight_alignment: float = 1.0
    weight_smooth: float = 0.0
    weight_direction: float = 0.1
    weight_celltype: float = 0.0
    weight_celltype_dir: float = 0.0
    weight_celltype_mag: float = 0.0
    weight_pseudotime: float = 0.0
    weight_magnitude: float = 0.0
    target_velocity_norm: Optional[float] = None
    weight_supervised: float = 0.0
    celltype_penalty: Literal["cosine", "mse", "both"] = "cosine"
    aux_cluster_loss_weight: float = 0.0
    max_neighbors: Optional[int] = None
    neighbor_max_distance: Optional[float] = None
    device: str = "auto"
    use_layer_norm: bool = True
    max_grad_norm: float = 5.0
    residual_to_baseline: bool = True
    weight_smooth_same: float = 0.0
    weight_boundary_align: float = 0.0
    weight_boundary_contrast: float = 0.0


class _VelocitySequenceDataset(Dataset):
    def __init__(
        self,
        latent: np.ndarray,
        embedding: np.ndarray,
        baseline_velocity: np.ndarray,
        neighbor_indices: np.ndarray,
        velocity_components: Optional[np.ndarray],
        projection: Optional[np.ndarray],
        config: TransformerConfig,
        cell_type_ids: Optional[np.ndarray] = None,
        type_means: Optional[np.ndarray] = None,
        cluster_labels: Optional[np.ndarray] = None,
        cluster_edge_list: Optional[List[tuple[int, int]]] = None,
        pseudotime: Optional[np.ndarray] = None,
        alignment_vectors: Optional[np.ndarray] = None,
        supervised_target: Optional[np.ndarray] = None,
        supervised_weight: Optional[np.ndarray] = None,
    ):
        self.latent = latent.astype(np.float32)
        self.embedding = embedding.astype(np.float32)
        self.baseline_velocity = baseline_velocity.astype(np.float32)
        self.neighbor_indices = neighbor_indices
        self.velocity_components = velocity_components.astype(np.float32) if velocity_components is not None else None
        self.projection = projection.astype(np.float32) if projection is not None else None
        self.max_neighbors = (
            config.max_neighbors if config.max_neighbors is not None else neighbor_indices.shape[1]
        )
        if alignment_vectors is not None:
            self.alignment_vectors = alignment_vectors.astype(np.float32)
        else:
            self.alignment_vectors = None
        if supervised_target is not None:
            self.supervised_target = supervised_target.astype(np.float32)
        else:
            self.supervised_target = None
        if supervised_weight is not None:
            self.supervised_weight = supervised_weight.astype(np.float32)
        else:
            self.supervised_weight = None
        if cell_type_ids is not None and type_means is not None:
            self.cell_type_ids = cell_type_ids.astype(np.int64)
            self.type_means = type_means.astype(np.float32)
        else:
            self.cell_type_ids = None
            self.type_means = None

        if cluster_labels is not None:
            self.cluster_labels = cluster_labels.astype(np.int64)
            self.n_clusters = int(self.cluster_labels.max()) + 1
        else:
            self.cluster_labels = None
            self.n_clusters = None
        self.cluster_edge_map: Optional[Dict[int, set[int]]] = None
        if cluster_edge_list and self.n_clusters is not None:
            edge_map: Dict[int, set[int]] = {}
            for src, dst in cluster_edge_list:
                edge_map.setdefault(int(src), set()).add(int(dst))
            self.cluster_edge_map = edge_map

        if pseudotime is not None:
            self.pseudotime = pseudotime.astype(np.float32)
        else:
            self.pseudotime = None

        self.feature_matrix = self.latent

    def __len__(self) -> int:
        return self.latent.shape[0]

    def _project_velocity(self, velocity: np.ndarray) -> np.ndarray:
        if self.velocity_components is None:
            return np.zeros((velocity.shape[0], 2), dtype=np.float32)
        comps = self.velocity_components
        proj = velocity @ comps[: min(comps.shape[0], velocity.shape[1])].T
        if self.projection is not None:
            proj = proj @ self.projection
        return proj

    def __getitem__(self, idx: int):
        neighbor_ids = self.neighbor_indices[idx][: self.max_neighbors]
        sequence_ids = np.concatenate([[idx], neighbor_ids])
        features = self.feature_matrix[sequence_ids]

        target_velocity = self.baseline_velocity[idx]
        neighbor_velocity = self.baseline_velocity[neighbor_ids]
        direction_vectors = self.embedding[neighbor_ids] - self.embedding[idx]
        velocity_proj = self._project_velocity(self.baseline_velocity[sequence_ids])
        sequence_velocity = self.baseline_velocity[sequence_ids]
        anchor_direction = np.zeros((1, direction_vectors.shape[1]), dtype=np.float32)
        sequence_direction = np.concatenate([anchor_direction, direction_vectors], axis=0)
        token_type = np.zeros(sequence_ids.shape[0], dtype=np.int64)
        token_type[1:] = 1

        if self.cell_type_ids is not None and self.type_means is not None:
            ct_id = int(self.cell_type_ids[idx])
            same_type_velocity = self.type_means[ct_id]
            same_type_flag = 1.0
        else:
            same_type_velocity = target_velocity
            same_type_flag = 0.0

        sample = {
            "features": torch.from_numpy(features),
            "target": torch.from_numpy(target_velocity),
            "neighbor_velocity": torch.from_numpy(neighbor_velocity),
            "direction": torch.from_numpy(direction_vectors),
            "velocity_proj": torch.from_numpy(velocity_proj),
            "same_type_velocity": torch.from_numpy(same_type_velocity.astype(np.float32)),
            "same_type_flag": torch.tensor(same_type_flag, dtype=torch.float32),
            "sequence_velocity": torch.from_numpy(sequence_velocity),
            "sequence_direction": torch.from_numpy(sequence_direction.astype(np.float32)),
            "token_type": torch.from_numpy(token_type),
        }
        if self.cluster_labels is not None:
            anchor_label = int(self.cluster_labels[idx])
            neighbor_labels = self.cluster_labels[neighbor_ids]
            same_mask = (neighbor_labels == anchor_label).astype(np.float32)
            boundary_mask = 1.0 - same_mask
            if self.cluster_edge_map and anchor_label in self.cluster_edge_map:
                allowed_targets = self.cluster_edge_map[anchor_label]
                allowed_mask = np.zeros_like(boundary_mask, dtype=np.float32)
                for target in allowed_targets:
                    allowed_mask += (neighbor_labels == target).astype(np.float32)
                allowed_mask = np.clip(allowed_mask, 0.0, 1.0)
            else:
                allowed_mask = boundary_mask.copy()
            valid_boundary = boundary_mask * allowed_mask
            invalid_boundary = boundary_mask - valid_boundary
            sample["cluster_label"] = torch.tensor(anchor_label, dtype=torch.long)
            sample["same_cluster_mask"] = torch.from_numpy(same_mask.astype(np.float32))
            sample["valid_boundary_mask"] = torch.from_numpy(valid_boundary.astype(np.float32))
            sample["invalid_boundary_mask"] = torch.from_numpy(np.clip(invalid_boundary, 0.0, 1.0).astype(np.float32))
        else:
            sample["same_cluster_mask"] = torch.ones(self.max_neighbors, dtype=torch.float32)
            sample["valid_boundary_mask"] = torch.zeros(self.max_neighbors, dtype=torch.float32)
            sample["invalid_boundary_mask"] = torch.zeros(self.max_neighbors, dtype=torch.float32)
        if self.pseudotime is not None:
            sample["pseudotime_anchor"] = torch.tensor(self.pseudotime[idx], dtype=torch.float32)
            sample["pseudotime_neighbors"] = torch.from_numpy(
                self.pseudotime[neighbor_ids].astype(np.float32)
            )
        if self.alignment_vectors is not None:
            alignment = self.alignment_vectors[idx]
            sample["alignment"] = torch.from_numpy(alignment.astype(np.float32))
        if self.supervised_target is not None:
            sample["supervised_target"] = torch.from_numpy(self.supervised_target[idx])
            if self.supervised_weight is not None:
                weight_value = float(self.supervised_weight[idx])
            else:
                weight_value = 1.0
            sample["supervised_weight"] = torch.tensor(weight_value, dtype=torch.float32)
        return sample


def _filter_neighbors_by_distance(
    neighbor_indices: np.ndarray,
    embedding: np.ndarray,
    max_distance: float,
) -> np.ndarray:
    if max_distance is None:
        return neighbor_indices
    filtered = neighbor_indices.copy()
    n_cells, k = filtered.shape
    anchor_indices = np.arange(n_cells, dtype=filtered.dtype)
    for i in range(n_cells):
        neighbors = filtered[i]
        distances = np.linalg.norm(embedding[neighbors] - embedding[i], axis=1)
        order = np.argsort(distances)
        distances = distances[order]
        neighbors = neighbors[order]
        valid = distances <= max_distance
        if np.any(valid):
            kept = neighbors[valid]
            if kept.size < k:
                pad = np.full(k - kept.size, anchor_indices[i], dtype=filtered.dtype)
                kept = np.concatenate([kept, pad], axis=0)
            filtered[i] = kept[:k]
        else:
            filtered[i] = anchor_indices[i]
    return filtered


class VelocityTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        direction_dim: int,
        config: TransformerConfig,
        seq_len: int,
        num_clusters: Optional[int] = None,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, config.hidden_dim)
        self.velocity_proj = nn.Linear(output_dim, config.hidden_dim)
        self.direction_proj = nn.Linear(direction_dim, config.hidden_dim)
        self.token_type_emb = nn.Embedding(2, config.hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.n_heads,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layers)
        self.output_head = nn.Linear(config.hidden_dim, output_dim)
        self.aux_head = nn.Linear(config.hidden_dim, num_clusters) if num_clusters is not None else None
        self.pos_embedding = nn.Parameter(torch.zeros(seq_len, config.hidden_dim))
        self.dropout = nn.Dropout(config.dropout)
        self.use_layer_norm = config.use_layer_norm
        self.layer_norm = nn.LayerNorm(config.hidden_dim) if self.use_layer_norm else None
        self.residual_to_baseline = config.residual_to_baseline
        nn.init.normal_(self.pos_embedding, std=0.02)

    def forward(
        self,
        features: torch.Tensor,
        sequence_velocity: torch.Tensor,
        sequence_direction: torch.Tensor,
        token_type: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        seq_len = features.size(1)
        x = self.input_proj(features)
        x = x + self.velocity_proj(sequence_velocity)
        x = x + self.direction_proj(sequence_direction)
        if token_type is not None:
            x = x + self.token_type_emb(token_type)
        x = x + self.pos_embedding[:seq_len].unsqueeze(0)
        x = self.dropout(x)
        hidden = self.encoder(x)
        anchor_hidden = hidden[:, 0, :]
        if self.layer_norm is not None:
            anchor_hidden = self.layer_norm(anchor_hidden)
        delta = self.output_head(anchor_hidden)
        aux_logits = self.aux_head(anchor_hidden) if self.aux_head is not None else None
        anchor_baseline = sequence_velocity[:, 0, :]
        if self.residual_to_baseline:
            anchor_velocity = anchor_baseline + delta
        else:
            anchor_velocity = delta
        return anchor_velocity, delta, aux_logits


def _determine_device(config: TransformerConfig) -> torch.device:
    if config.device == "cpu":
        return torch.device("cpu")
    if config.device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if config.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(config.device)


def refine_velocities_with_transformer(
    latent: np.ndarray,
    embedding: np.ndarray,
    baseline_velocity: np.ndarray,
    neighbor_indices: np.ndarray,
    velocity_components: Optional[np.ndarray],
    projection: Optional[np.ndarray],
    config: TransformerConfig,
    wandb_run=None,
    wandb_prefix: str = "transformer",
    cell_type_ids: Optional[np.ndarray] = None,
    type_means: Optional[np.ndarray] = None,
    cluster_labels: Optional[np.ndarray] = None,
    cluster_edge_list: Optional[List[tuple[int, int]]] = None,
    pseudotime: Optional[np.ndarray] = None,
    alignment_vectors: Optional[np.ndarray] = None,
    supervised_target: Optional[np.ndarray] = None,
    supervised_weight: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
) -> np.ndarray:
    if config.neighbor_max_distance is not None:
        neighbor_indices = _filter_neighbors_by_distance(
            neighbor_indices,
            embedding,
            config.neighbor_max_distance,
        )

    dataset = _VelocitySequenceDataset(
        latent=latent,
        embedding=embedding,
        baseline_velocity=baseline_velocity,
        neighbor_indices=neighbor_indices,
        velocity_components=velocity_components,
        projection=projection,
        config=config,
        cell_type_ids=cell_type_ids,
        type_means=type_means,
        cluster_labels=cluster_labels,
        cluster_edge_list=cluster_edge_list,
        pseudotime=pseudotime,
        alignment_vectors=alignment_vectors,
        supervised_target=supervised_target,
        supervised_weight=supervised_weight,
    )

    seq_len = dataset.neighbor_indices.shape[1] + 1
    input_dim = dataset.feature_matrix.shape[1]
    output_dim = baseline_velocity.shape[1]
    direction_dim = dataset.embedding.shape[1]
    device = _determine_device(config)

    model = VelocityTransformer(
        input_dim=input_dim,
        output_dim=output_dim,
        direction_dim=direction_dim,
        config=config,
        seq_len=seq_len,
        num_clusters=dataset.n_clusters,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, drop_last=False)

    use_aux_head = config.aux_cluster_loss_weight > 0.0 and dataset.n_clusters is not None

    comps = dataset.velocity_components
    proj_mat = dataset.projection
    comp_tensor = None
    if comps is not None:
        comp_tensor = torch.from_numpy(comps[: min(comps.shape[0], output_dim)]).to(device)
    proj_tensor = torch.from_numpy(proj_mat).to(device) if proj_mat is not None else None

    for epoch in range(config.epochs):
        model.train()
        running_loss = 0.0
        running_steps = 0
        running_alignment = 0.0
        running_smooth = 0.0
        running_direction = 0.0
        running_celltype_dir = 0.0
        running_celltype_mag = 0.0
        running_aux = 0.0
        running_pseudotime = 0.0
        running_magnitude = 0.0
        running_supervised = 0.0
        running_boundary_align = 0.0
        running_boundary_contrast = 0.0
        for batch in dataloader:
            features = batch["features"].to(device)
            neighbor_velocity = batch["neighbor_velocity"].to(device)
            direction = batch["direction"].to(device)
            sequence_velocity = batch["sequence_velocity"].to(device)
            sequence_direction = batch["sequence_direction"].to(device)
            token_type = batch["token_type"].to(device)
            same_type_velocity = batch["same_type_velocity"].to(device)
            same_type_flag = batch["same_type_flag"].to(device)
            cluster_label = batch.get("cluster_label") if use_aux_head else None
            if cluster_label is not None:
                cluster_label = cluster_label.to(device)
            alignment_target = batch.get("alignment")
            if alignment_target is not None:
                alignment_target = alignment_target.to(device)
            pseudotime_anchor = batch.get("pseudotime_anchor")
            pseudotime_neighbors = batch.get("pseudotime_neighbors")
            if pseudotime_anchor is not None:
                pseudotime_anchor = pseudotime_anchor.to(device)
            if pseudotime_neighbors is not None:
                pseudotime_neighbors = pseudotime_neighbors.to(device)
            supervised_target_batch = batch.get("supervised_target")
            supervised_weight_batch = batch.get("supervised_weight")
            if supervised_target_batch is not None:
                supervised_target_batch = supervised_target_batch.to(device)
            if supervised_weight_batch is not None:
                supervised_weight_batch = supervised_weight_batch.to(device)
            same_cluster_mask = batch.get("same_cluster_mask")
            valid_boundary_mask = batch.get("valid_boundary_mask")
            invalid_boundary_mask = batch.get("invalid_boundary_mask")
            if same_cluster_mask is not None:
                same_cluster_mask = same_cluster_mask.to(device)
            if valid_boundary_mask is not None:
                valid_boundary_mask = valid_boundary_mask.to(device)
            if invalid_boundary_mask is not None:
                invalid_boundary_mask = invalid_boundary_mask.to(device)

            pred, _, aux_logits = model(features, sequence_velocity, sequence_direction, token_type)
            if config.weight_alignment > 0.0 and alignment_target is not None:
                pred_unit = F.normalize(pred, dim=1, eps=1e-8)
                alignment_unit = F.normalize(alignment_target, dim=1, eps=1e-8)
                cos = (pred_unit * alignment_unit).sum(dim=1)
                loss_alignment = torch.mean(1.0 - cos)
            else:
                loss_alignment = torch.tensor(0.0, device=device)

            if config.weight_supervised > 0.0 and supervised_target_batch is not None:
                diff = pred - supervised_target_batch
                if supervised_weight_batch is not None:
                    weights = supervised_weight_batch.view(-1, 1)
                    denom = torch.clamp(weights.sum(), min=1.0)
                    loss_supervised = (diff.pow(2) * weights).sum() / denom
                else:
                    loss_supervised = F.mse_loss(pred, supervised_target_batch)
            else:
                loss_supervised = torch.tensor(0.0, device=device)

            smooth_residual = pred.unsqueeze(1) - neighbor_velocity
            if config.weight_smooth_same > 0.0 and same_cluster_mask is not None:
                diff_sq = smooth_residual.pow(2).sum(dim=2)
                denom = same_cluster_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
                masked = (diff_sq * same_cluster_mask).sum(dim=1) / denom.squeeze(1)
                loss_smooth = masked.mean()
                smooth_weight = config.weight_smooth_same
            else:
                loss_smooth = F.mse_loss(smooth_residual, torch.zeros_like(smooth_residual))
                smooth_weight = config.weight_smooth

            proj = None
            if comp_tensor is not None:
                proj = pred @ comp_tensor.T
                if proj_tensor is not None:
                    proj = proj @ proj_tensor
                dir_norm = torch.norm(direction, dim=-1, keepdim=True).clamp_min(1e-5)
                direction_unit = direction / dir_norm
                proj_proj = proj.unsqueeze(1)
                proj_norm = torch.norm(proj_proj, dim=-1, keepdim=True).clamp_min(1e-5)
                proj_unit = proj_proj / proj_norm
                cos = (proj_unit * direction_unit).sum(dim=-1)
                loss_direction = torch.mean(1.0 - cos)
            else:
                loss_direction = torch.tensor(0.0, device=device)

            loss_boundary_align = torch.tensor(0.0, device=device)
            loss_boundary_contrast = torch.tensor(0.0, device=device)
            if (
                config.weight_boundary_align > 0.0
                and proj is not None
                and valid_boundary_mask is not None
            ):
                mask = valid_boundary_mask
                denom = mask.sum(dim=1, keepdim=True)
                valid_rows = denom.squeeze(1) > 1e-6
                if valid_rows.any():
                    mean_dir = (direction * mask.unsqueeze(-1)).sum(dim=1) / denom.clamp_min(1e-6)
                    mean_dir = mean_dir[valid_rows]
                    proj_valid = proj[valid_rows]
                    mean_dir_unit = F.normalize(mean_dir, dim=1, eps=1e-8)
                    proj_unit_valid = F.normalize(proj_valid, dim=1, eps=1e-8)
                    cos = (proj_unit_valid * mean_dir_unit).sum(dim=1)
                    loss_boundary_align = torch.mean(1.0 - cos)
            if (
                config.weight_boundary_contrast > 0.0
                and proj is not None
                and invalid_boundary_mask is not None
            ):
                mask = invalid_boundary_mask
                denom = mask.sum(dim=1, keepdim=True)
                valid_rows = denom.squeeze(1) > 1e-6
                if valid_rows.any():
                    mean_dir = (direction * mask.unsqueeze(-1)).sum(dim=1) / denom.clamp_min(1e-6)
                    mean_dir = mean_dir[valid_rows]
                    proj_valid = proj[valid_rows]
                    mean_dir_unit = F.normalize(mean_dir, dim=1, eps=1e-8)
                    proj_unit_valid = F.normalize(proj_valid, dim=1, eps=1e-8)
                    cos = (proj_unit_valid * mean_dir_unit).sum(dim=1)
                    loss_boundary_contrast = torch.mean(torch.clamp(cos, min=0.0))

            same_type_flag = same_type_flag.view(-1)
            dir_weight = config.weight_celltype_dir
            mag_weight = config.weight_celltype_mag
            base_weight = config.weight_celltype
            if dir_weight <= 0.0 and base_weight > 0.0 and config.celltype_penalty in ("cosine", "both"):
                dir_weight = base_weight
            if mag_weight <= 0.0 and base_weight > 0.0 and config.celltype_penalty in ("mse", "both"):
                mag_weight = base_weight

            if dir_weight > 0.0:
                proto_unit = F.normalize(same_type_velocity, dim=1, eps=1e-8)
                pred_unit = F.normalize(pred, dim=1, eps=1e-8)
                cos = (pred_unit * proto_unit).sum(dim=1)
                loss_celltype_dir = ((1.0 - cos) * same_type_flag).mean()
            else:
                loss_celltype_dir = torch.tensor(0.0, device=device)

            if mag_weight > 0.0:
                celltype_residual = F.mse_loss(pred, same_type_velocity, reduction="none").mean(dim=1)
                loss_celltype_mag = (celltype_residual * same_type_flag).mean()
            else:
                loss_celltype_mag = torch.tensor(0.0, device=device)

            if use_aux_head and aux_logits is not None and cluster_label is not None:
                loss_aux = F.cross_entropy(aux_logits, cluster_label, reduction="mean")
            else:
                loss_aux = torch.tensor(0.0, device=device)

            if config.weight_pseudotime > 0.0 and pseudotime_anchor is not None and pseudotime_neighbors is not None:
                delta_t = pseudotime_neighbors - pseudotime_anchor.unsqueeze(1)
                weights_pt = torch.relu(delta_t)
                valid = weights_pt.sum(dim=1) > 1e-5
                if valid.any():
                    grad_vec = torch.sum(weights_pt[valid].unsqueeze(-1) * direction[valid], dim=1)
                    grad_norm = torch.norm(grad_vec, dim=1, keepdim=True).clamp_min(1e-6)
                    grad_unit = grad_vec / grad_norm
                    pred_unit_valid = F.normalize(pred[valid], dim=1, eps=1e-8)
                    cos_grad = (pred_unit_valid * grad_unit).sum(dim=1)
                    loss_pseudotime = torch.mean(1.0 - cos_grad)
                else:
                    loss_pseudotime = torch.tensor(0.0, device=device)
            else:
                loss_pseudotime = torch.tensor(0.0, device=device)

            if config.weight_magnitude > 0.0:
                pred_norm = torch.norm(pred, dim=1)
                if config.target_velocity_norm is not None:
                    target_norm = torch.full_like(pred_norm, config.target_velocity_norm)
                else:
                    target_norm = torch.norm(direction, dim=2).mean(dim=1)
                loss_magnitude = F.mse_loss(pred_norm, target_norm)
            else:
                loss_magnitude = torch.tensor(0.0, device=device)

            loss = (
                config.weight_alignment * loss_alignment
                + smooth_weight * loss_smooth
                + config.weight_direction * loss_direction
                + dir_weight * loss_celltype_dir
                + mag_weight * loss_celltype_mag
                + (config.aux_cluster_loss_weight if use_aux_head else 0.0) * loss_aux
                + config.weight_pseudotime * loss_pseudotime
                + config.weight_magnitude * loss_magnitude
                + config.weight_supervised * loss_supervised
                + config.weight_boundary_align * loss_boundary_align
                + config.weight_boundary_contrast * loss_boundary_contrast
            )
            optimizer.zero_grad()
            loss.backward()
            if config.max_grad_norm is not None and config.max_grad_norm > 0:
                clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            running_loss += loss.detach().item()
            running_alignment += loss_alignment.detach().item()
            running_smooth += loss_smooth.detach().item()
            running_direction += loss_direction.detach().item()
            running_celltype_dir += loss_celltype_dir.detach().item()
            running_celltype_mag += loss_celltype_mag.detach().item()
            if use_aux_head and aux_logits is not None:
                running_aux += loss_aux.detach().item()
            running_pseudotime += loss_pseudotime.detach().item()
            running_magnitude += loss_magnitude.detach().item()
            running_supervised += loss_supervised.detach().item()
            running_boundary_align += loss_boundary_align.detach().item()
            running_boundary_contrast += loss_boundary_contrast.detach().item()
            running_steps += 1
        if wandb_run is not None:
            try:
                log_payload = {
                    f"{wandb_prefix}/loss": running_loss / max(running_steps, 1),
                    f"{wandb_prefix}/loss_alignment": running_alignment / max(running_steps, 1),
                    f"{wandb_prefix}/loss_smooth": running_smooth / max(running_steps, 1),
                    f"{wandb_prefix}/loss_direction": running_direction / max(running_steps, 1),
                }
                if dir_weight > 0.0:
                    log_payload[f"{wandb_prefix}/loss_celltype_dir"] = running_celltype_dir / max(
                        running_steps, 1
                    )
                if mag_weight > 0.0:
                    log_payload[f"{wandb_prefix}/loss_celltype_mag"] = running_celltype_mag / max(
                        running_steps, 1
                    )
                if use_aux_head:
                    log_payload[f"{wandb_prefix}/loss_aux_cluster"] = running_aux / max(running_steps, 1)
                if config.weight_pseudotime > 0.0:
                    log_payload[f"{wandb_prefix}/loss_pseudotime"] = running_pseudotime / max(
                        running_steps, 1
                    )
                if config.weight_magnitude > 0.0:
                    log_payload[f"{wandb_prefix}/loss_magnitude"] = running_magnitude / max(
                        running_steps, 1
                    )
                if config.weight_supervised > 0.0:
                    log_payload[f"{wandb_prefix}/loss_supervised"] = running_supervised / max(
                        running_steps, 1
                    )
                if config.weight_boundary_align > 0.0:
                    log_payload[f"{wandb_prefix}/loss_boundary_align"] = running_boundary_align / max(
                        running_steps, 1
                    )
                if config.weight_boundary_contrast > 0.0:
                    log_payload[f"{wandb_prefix}/loss_boundary_contrast"] = running_boundary_contrast / max(
                        running_steps, 1
                    )
                wandb_run.log(
                    log_payload,
                    step=epoch,
                )
            except Exception:  # pragma: no cover
                pass

    model.eval()
    refined = []
    inference_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)
    with torch.no_grad():
        for batch in inference_loader:
            features = batch["features"].to(device)
            sequence_velocity = batch["sequence_velocity"].to(device)
            sequence_direction = batch["sequence_direction"].to(device)
            token_type = batch["token_type"].to(device)
            preds, _, _ = model(features, sequence_velocity, sequence_direction, token_type)
            preds = preds.cpu().numpy()
            refined.append(preds)
    refined_velocity = np.vstack(refined)

    if save_path is not None:
        torch.save(model.state_dict(), save_path)

    return refined_velocity.astype(np.float32)
