from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal

from ._module import VELOVAE
from ._constants import VELOVI_REGISTRY_KEYS
from scvi.module._constants import MODULE_KEYS


def _identity(x: torch.Tensor) -> torch.Tensor:
    return x


class TransformerSequenceEncoder(nn.Module):
    """Transformer encoder operating over anchor + neighbor expression tokens."""

    def __init__(
        self,
        *,
        spliced_full: np.ndarray,
        unspliced_full: np.ndarray,
        n_latent: int,
        latent_distribution: str = "normal",
        var_activation=None,
        hidden_dim: int = 256,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        log_variational: bool = False,
        var_eps: float = 1e-4,
    ):
        super().__init__()
        spliced_tensor = torch.as_tensor(spliced_full, dtype=torch.float32)
        unspliced_tensor = torch.as_tensor(unspliced_full, dtype=torch.float32)
        self.register_buffer("spliced_full", spliced_tensor)
        self.register_buffer("unspliced_full", unspliced_tensor)
        self.feature_dim = spliced_tensor.shape[1] + unspliced_tensor.shape[1]
        self.hidden_dim = hidden_dim
        self.log_variational = log_variational
        self.var_activation = var_activation if var_activation is not None else torch.exp
        self.var_eps = var_eps
        self.distribution = latent_distribution
        self.input_proj = nn.Linear(self.feature_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dropout=dropout,
            norm_first=True,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.token_type_emb = nn.Embedding(2, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.mean_encoder = nn.Linear(hidden_dim, n_latent)
        self.var_encoder = nn.Linear(hidden_dim, n_latent)
        if latent_distribution == "ln":
            self.z_transformation = nn.Softmax(dim=-1)
        else:
            self.z_transformation = _identity
        self._pos_cache: dict[int, torch.Tensor] = {}

    def _positional_encoding(self, seq_len: int, device: torch.device) -> torch.Tensor:
        cached = self._pos_cache.get(seq_len)
        if cached is not None and cached.device == device:
            return cached
        position = torch.arange(seq_len, dtype=torch.float32, device=device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.hidden_dim, 2, dtype=torch.float32, device=device)
            * (-math.log(10000.0) / max(1, self.hidden_dim))
        )
        pe = torch.zeros(seq_len, self.hidden_dim, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self._pos_cache[seq_len] = pe
        return pe

    def _gather_neighbors(
        self,
        neighbor_index: torch.Tensor,
        target_device: torch.device,
    ) -> torch.Tensor:
        if neighbor_index.numel() == 0:
            return torch.zeros(
                neighbor_index.shape[0],
                neighbor_index.shape[1],
                self.feature_dim,
                device=target_device,
            )
        neighbor_index = neighbor_index.long().to(self.spliced_full.device)
        batch_size, seq_len = neighbor_index.shape
        flat_index = neighbor_index.reshape(-1)
        spliced_neighbors = torch.index_select(self.spliced_full, 0, flat_index)
        unspliced_neighbors = torch.index_select(self.unspliced_full, 0, flat_index)
        if self.log_variational:
            spliced_neighbors = torch.log(0.01 + spliced_neighbors)
            unspliced_neighbors = torch.log(0.01 + unspliced_neighbors)
        neighbor_features = torch.cat([spliced_neighbors, unspliced_neighbors], dim=-1)
        neighbor_features = neighbor_features.view(batch_size, seq_len, -1)
        return neighbor_features.to(target_device)

    def forward(
        self,
        encoder_input: torch.Tensor,
        neighbor_index: torch.Tensor,
        neighbor_weight: Optional[torch.Tensor] = None,
        neighbor_index2: Optional[torch.Tensor] = None,
        neighbor_weight2: Optional[torch.Tensor] = None,
    ):
        del neighbor_index2, neighbor_weight2  # Unused but kept for parity with graph encoders
        device = encoder_input.device
        if not torch.is_tensor(neighbor_index):
            neighbor_index = torch.as_tensor(neighbor_index, dtype=torch.long, device=device)
        else:
            neighbor_index = neighbor_index.to(device)

        if neighbor_weight is not None:
            if not torch.is_tensor(neighbor_weight):
                neighbor_weight = torch.as_tensor(
                    neighbor_weight,
                    dtype=encoder_input.dtype,
                    device=device,
                )
            else:
                neighbor_weight = neighbor_weight.to(device)
        else:
            if neighbor_index.shape[1] > 0:
                neighbor_weight = torch.full(
                    (neighbor_index.shape[0], neighbor_index.shape[1]),
                    fill_value=1.0 / float(neighbor_index.shape[1]),
                    dtype=encoder_input.dtype,
                    device=device,
                )
            else:
                neighbor_weight = torch.zeros(
                    neighbor_index.shape[0],
                    0,
                    dtype=encoder_input.dtype,
                    device=device,
                )

        neighbor_features = self._gather_neighbors(neighbor_index, device)
        neighbor_features = neighbor_features * neighbor_weight.unsqueeze(-1)

        anchor = encoder_input.unsqueeze(1)
        sequence = torch.cat([anchor, neighbor_features], dim=1)
        token_type = torch.zeros(sequence.shape[0], sequence.shape[1], dtype=torch.long, device=device)
        if token_type.shape[1] > 1:
            token_type[:, 1:] = 1

        hidden = self.input_proj(sequence)
        hidden = hidden + self.token_type_emb(token_type)
        hidden = hidden + self._positional_encoding(sequence.shape[1], device)
        hidden = self.dropout(hidden)
        encoded = self.encoder(hidden)
        anchor_hidden = self.layer_norm(encoded[:, 0, :])

        qz_m = self.mean_encoder(anchor_hidden)
        qz_v = self.var_activation(self.var_encoder(anchor_hidden)) + self.var_eps
        latent = Normal(qz_m, qz_v.sqrt()).rsample()
        z = self.z_transformation(latent)
        return qz_m, qz_v, z


class TransformerVELOVAE(VELOVAE):
    """VELOVAE variant using a transformer encoder over neighborhood tokens."""

    def __init__(
        self,
        *,
        spliced_full: np.ndarray,
        unspliced_full: np.ndarray,
        transformer_hidden_dim: int = 256,
        transformer_heads: int = 4,
        transformer_layers: int = 2,
        transformer_dropout: float = 0.1,
        transformer_max_neighbors: Optional[int] = None,
        transformer_neighbor_reg_weight: float = 0.0,
        var_activation=None,
        **super_kwargs,
    ):
        self._transformer_hidden_dim = transformer_hidden_dim
        self._transformer_heads = transformer_heads
        self._transformer_layers = transformer_layers
        self._transformer_dropout = transformer_dropout
        self._transformer_max_neighbors = transformer_max_neighbors
        self.transformer_neighbor_reg_weight = float(transformer_neighbor_reg_weight)
        feature_dim = spliced_full.shape[1] + unspliced_full.shape[1]
        activation_fn = var_activation if var_activation is not None else torch.nn.Softplus()
        super().__init__(var_activation=activation_fn, **super_kwargs)
        self.z_encoder = TransformerSequenceEncoder(
            spliced_full=spliced_full,
            unspliced_full=unspliced_full,
            n_latent=self.n_latent,
            latent_distribution=self.latent_distribution,
            var_activation=activation_fn,
            hidden_dim=transformer_hidden_dim,
            n_heads=transformer_heads,
            n_layers=transformer_layers,
            dropout=transformer_dropout,
            log_variational=self.log_variational,
        )
        self.neighbor_projection = nn.Linear(feature_dim, self.n_latent)

    def _clip_neighbors(self, neighbor_index: torch.Tensor, neighbor_weight: Optional[torch.Tensor]):
        if neighbor_index is None:
            return None, neighbor_weight
        if self._transformer_max_neighbors is None:
            return neighbor_index, neighbor_weight
        max_neighbors = min(self._transformer_max_neighbors, neighbor_index.shape[1])
        neighbor_index = neighbor_index[:, :max_neighbors]
        if neighbor_weight is not None:
            neighbor_weight = neighbor_weight[:, :max_neighbors]
        return neighbor_index, neighbor_weight

    def _get_inference_input(self, tensors):
        input_dict = super()._get_inference_input(tensors)
        input_dict[VELOVI_REGISTRY_KEYS.NEIGHBOR_INDEX_KEY] = tensors[
            VELOVI_REGISTRY_KEYS.NEIGHBOR_INDEX_KEY
        ]
        weight_tensor = tensors.get(VELOVI_REGISTRY_KEYS.NEIGHBOR_WEIGHT_KEY)
        if weight_tensor is not None:
            input_dict[VELOVI_REGISTRY_KEYS.NEIGHBOR_WEIGHT_KEY] = weight_tensor
        return input_dict

    def inference(
        self,
        spliced,
        unspliced,
        neighbor_index,
        neighbor_weight=None,
        n_samples: int = 1,
    ):
        spliced_ = spliced
        unspliced_ = unspliced
        if self.log_variational:
            spliced_ = torch.log(0.01 + spliced)
            unspliced_ = torch.log(0.01 + unspliced)

        encoder_input = torch.cat((spliced_, unspliced_), dim=-1)
        target_device = self.z_encoder.mean_encoder.weight.device
        if encoder_input.device != target_device:
            encoder_input = encoder_input.to(target_device)
        if not torch.is_tensor(neighbor_index):
            neighbor_index = torch.as_tensor(neighbor_index, dtype=torch.long, device=target_device)
        else:
            neighbor_index = neighbor_index.to(target_device)
        if neighbor_weight is not None:
            if not torch.is_tensor(neighbor_weight):
                neighbor_weight = torch.as_tensor(
                    neighbor_weight, dtype=encoder_input.dtype, device=target_device
                )
            else:
                neighbor_weight = neighbor_weight.to(target_device)

        neighbor_index, neighbor_weight = self._clip_neighbors(neighbor_index, neighbor_weight)

        qz_m, qz_v, z = self.z_encoder(
            encoder_input,
            neighbor_index,
            neighbor_weight=neighbor_weight,
        )

        if n_samples > 1:
            qz_m = qz_m.unsqueeze(0).expand((n_samples, qz_m.size(0), qz_m.size(1)))
            qz_v = qz_v.unsqueeze(0).expand((n_samples, qz_v.size(0), qz_v.size(1)))
            untran_z = Normal(qz_m, qz_v.sqrt()).sample()
            z = self.z_encoder.z_transformation(untran_z)

        gamma, beta, alpha, alpha_1, lambda_alpha = self._get_rates()

        outputs = {
            MODULE_KEYS.Z_KEY: z,
            MODULE_KEYS.QZM_KEY: qz_m,
            MODULE_KEYS.QZV_KEY: qz_v,
            "gamma": gamma,
            "beta": beta,
            "alpha": alpha,
            "alpha_1": alpha_1,
            "lambda_alpha": lambda_alpha,
        }
        return outputs

    def _prepare_neighbor_inputs(
        self,
        neighbor_index,
        neighbor_weight,
        device,
        dtype,
    ):
        if neighbor_index is None:
            return None, None
        if not torch.is_tensor(neighbor_index):
            neighbor_index = torch.as_tensor(neighbor_index, dtype=torch.long, device=device)
        else:
            neighbor_index = neighbor_index.to(device)
        if neighbor_index.shape[1] == 0:
            return neighbor_index, None
        if neighbor_weight is None:
            neighbor_weight = torch.full(
                (neighbor_index.shape[0], neighbor_index.shape[1]),
                fill_value=1.0 / float(neighbor_index.shape[1]),
                dtype=dtype,
                device=device,
            )
        else:
            if not torch.is_tensor(neighbor_weight):
                neighbor_weight = torch.as_tensor(neighbor_weight, dtype=dtype, device=device)
            else:
                neighbor_weight = neighbor_weight.to(device, dtype=dtype)
        return neighbor_index, neighbor_weight

    def _project_neighbor_summary(
        self,
        neighbor_index: torch.Tensor,
        neighbor_weight: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        neighbor_features = self.z_encoder._gather_neighbors(neighbor_index, device)
        weighted = neighbor_features * neighbor_weight.unsqueeze(-1)
        summary = torch.sum(weighted, dim=1)
        return self.neighbor_projection(summary)

    def loss(
        self,
        tensors,
        inference_outputs,
        generative_outputs,
        kl_weight: float = 1.0,
        n_obs: float = 1.0,
    ):
        loss_output = super().loss(tensors, inference_outputs, generative_outputs, kl_weight, n_obs)
        if self.transformer_neighbor_reg_weight <= 0:
            return loss_output
        neighbor_index = tensors.get(VELOVI_REGISTRY_KEYS.NEIGHBOR_INDEX_KEY)
        if neighbor_index is None:
            return loss_output
        z = inference_outputs.get(MODULE_KEYS.Z_KEY)
        if z is None or z.shape[0] == 0:
            return loss_output
        neighbor_weight = tensors.get(VELOVI_REGISTRY_KEYS.NEIGHBOR_WEIGHT_KEY)
        device = z.device
        dtype = z.dtype
        neighbor_index, neighbor_weight = self._prepare_neighbor_inputs(
            neighbor_index,
            neighbor_weight,
            device,
            dtype,
        )
        neighbor_index, neighbor_weight = self._clip_neighbors(neighbor_index, neighbor_weight)
        if neighbor_index is None or neighbor_weight is None or neighbor_index.shape[1] == 0:
            return loss_output
        projected_neighbors = self._project_neighbor_summary(neighbor_index, neighbor_weight, device)
        penalty = F.mse_loss(z, projected_neighbors)
        loss_output.loss = loss_output.loss + self.transformer_neighbor_reg_weight * penalty
        metrics = loss_output.extra_metrics if loss_output.extra_metrics is not None else {}
        metrics["transformer_neighbor_penalty"] = penalty.detach()
        loss_output.extra_metrics = metrics
        return loss_output
