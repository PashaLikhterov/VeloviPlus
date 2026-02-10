from __future__ import annotations

import warnings

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Dirichlet, Normal
from torch.distributions import kl_divergence as kl

from scvi._constants import REGISTRY_KEYS
from scvi.nn import Encoder

from ._module import VELOVAE
from ._constants import VELOVI_REGISTRY_KEYS
from scvi.module._constants import MODULE_KEYS
from scvi.module.base import LossOutput


class NeighborMessageAggregator(nn.Module):
    """Aggregate neighbor information to augment encoder inputs."""

    def _ensure_module_device(self, module, device):
        if module is None:
            return module
        first_param = next(module.parameters(), None)
        if first_param is not None and first_param.device != device:
            module = module.to(device)
        return module

    def __init__(
        self,
        spliced_full: np.ndarray,
        unspliced_full: np.ndarray,
        hidden_dim: int,
        log_variational: bool,
        dropout_rate: float,
        use_attention: bool = False,
        use_gate: bool = False,
        use_residual: bool = False,
        use_differences: bool = False,
    ):
        super().__init__()
        spliced_tensor = torch.as_tensor(spliced_full, dtype=torch.float32)
        unspliced_tensor = torch.as_tensor(unspliced_full, dtype=torch.float32)
        self.register_buffer("spliced_full", spliced_tensor)
        self.register_buffer("unspliced_full", unspliced_tensor)
        self.log_variational = log_variational
        input_dim = spliced_tensor.shape[1] + unspliced_tensor.shape[1]
        self.use_attention = use_attention
        self.use_gate = use_gate
        self.use_residual = use_residual
        self.use_differences = use_differences
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        if self.use_differences:
            self.difference_projection = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
            )
        else:
            self.difference_projection = None
        if self.use_attention:
            self.attention = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1),
            )
        else:
            self.attention = None
        if self.use_gate:
            self.layer_norm = nn.LayerNorm(hidden_dim)
            self.message_gate = nn.Parameter(torch.zeros(hidden_dim))
        else:
            self.layer_norm = None
            self.message_gate = None
        self.cross_graph_gate = nn.Parameter(torch.zeros(hidden_dim))
        if self.use_residual:
            self.residual_projection = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
            )
        else:
            self.residual_projection = None

    def forward(
        self,
        encoder_input: torch.Tensor,
        neighbor_index: torch.Tensor,
        neighbor_weight: torch.Tensor | None = None,
        neighbor_index2: torch.Tensor | None = None,
        neighbor_weight2: torch.Tensor | None = None,
    ) -> torch.Tensor:
        device = encoder_input.device
        self.projection = self._ensure_module_device(self.projection, device)
        if self.attention is not None:
            self.attention = self._ensure_module_device(self.attention, device)
        if self.layer_norm is not None:
            self.layer_norm = self._ensure_module_device(self.layer_norm, device)
        if self.difference_projection is not None:
            self.difference_projection = self._ensure_module_device(self.difference_projection, device)
        if self.residual_projection is not None:
            self.residual_projection = self._ensure_module_device(self.residual_projection, device)
        if self.message_gate is not None and self.message_gate.device != device:
            self.message_gate.data = self.message_gate.data.to(device)
        if self.cross_graph_gate.device != device:
            self.cross_graph_gate.data = self.cross_graph_gate.data.to(device)

        neighbor_index = neighbor_index.long().to(self.spliced_full.device)
        batch_size, k_neighbors = neighbor_index.shape
        neighbor_index_flat = neighbor_index.reshape(-1)

        spliced_neighbors = torch.index_select(self.spliced_full, 0, neighbor_index_flat)
        unspliced_neighbors = torch.index_select(self.unspliced_full, 0, neighbor_index_flat)

        if self.log_variational:
            spliced_neighbors = torch.log(0.01 + spliced_neighbors)
            unspliced_neighbors = torch.log(0.01 + unspliced_neighbors)

        neighbor_features = torch.cat([spliced_neighbors, unspliced_neighbors], dim=-1)
        neighbor_features = neighbor_features.view(batch_size, k_neighbors, -1).to(device)

        if neighbor_weight is None:
            neighbor_weight = torch.full(
                (batch_size, k_neighbors),
                fill_value=1.0 / k_neighbors,
                dtype=encoder_input.dtype,
                device=device,
            )
        else:
            neighbor_weight = neighbor_weight.to(device)

        if self.use_attention:
            att_scores = self.attention(neighbor_features).squeeze(-1)
            att_scores = torch.softmax(att_scores, dim=1)
            neighbor_weight = neighbor_weight * att_scores

        neighbor_weight = neighbor_weight / (neighbor_weight.sum(dim=1, keepdim=True) + 1e-12)
        neighbor_weight = neighbor_weight.unsqueeze(-1)
        aggregated = torch.sum(neighbor_weight * neighbor_features, dim=1)
        projected = self.projection(aggregated)

        # Optional second neighbor source (e.g., latent-graph neighbors)
        if neighbor_index2 is not None:
            neighbor_index2 = neighbor_index2.long().to(self.spliced_full.device)
            b2, k2 = neighbor_index2.shape
            ni2_flat = neighbor_index2.reshape(-1)
            sp2 = torch.index_select(self.spliced_full, 0, ni2_flat)
            un2 = torch.index_select(self.unspliced_full, 0, ni2_flat)
            if self.log_variational:
                sp2 = torch.log(0.01 + sp2)
                un2 = torch.log(0.01 + un2)
            nf2 = torch.cat([sp2, un2], dim=-1).view(b2, k2, -1).to(device)
            if neighbor_weight2 is None:
                nw2 = torch.full(
                    (b2, k2),
                    fill_value=1.0 / k2,
                    dtype=encoder_input.dtype,
                    device=device,
                )
            else:
                nw2 = neighbor_weight2.to(device)
            if self.use_attention:
                att2 = self.attention(nf2).squeeze(-1)
                att2 = torch.softmax(att2, dim=1)
                nw2 = nw2 * att2
            nw2 = nw2 / (nw2.sum(dim=1, keepdim=True) + 1e-12)
            nf2_sum = torch.sum(nw2.unsqueeze(-1) * nf2, dim=1)
            proj2 = self.projection(nf2_sum)
            gate = torch.sigmoid(self.cross_graph_gate)
            projected = gate * projected + (1.0 - gate) * proj2
        if self.use_differences:
            difference_features = neighbor_features - encoder_input.unsqueeze(1)
            difference_summary = torch.sum(neighbor_weight * difference_features, dim=1)
            projected = projected + self.difference_projection(difference_summary)
        if self.use_residual:
            projected = projected + self.residual_projection(encoder_input)
        if self.use_gate:
            projected = self.layer_norm(projected)
            gate = torch.sigmoid(self.message_gate)
            projected = gate * projected

        return torch.cat([encoder_input, projected], dim=-1)


class GraphAugmentedEncoder(nn.Module):
    """Wraps a standard encoder with message passing augmentation."""

    def __init__(self, base_encoder: Encoder, aggregator: NeighborMessageAggregator):
        super().__init__()
        self.base_encoder = base_encoder
        self.aggregator = aggregator

    @property
    def distribution(self) -> str:
        return self.base_encoder.distribution

    @property
    def var_activation(self):
        return self.base_encoder.var_activation

    def forward(
        self,
        encoder_input: torch.Tensor,
        neighbor_index: torch.Tensor,
        neighbor_weight: torch.Tensor | None = None,
        neighbor_index2: torch.Tensor | None = None,
        neighbor_weight2: torch.Tensor | None = None,
    ):
        enriched_input = self.aggregator(
            encoder_input=encoder_input,
            neighbor_index=neighbor_index,
            neighbor_weight=neighbor_weight,
            neighbor_index2=neighbor_index2,
            neighbor_weight2=neighbor_weight2,
        )
        target_device = self.base_encoder.mean_encoder.weight.device
        if enriched_input.device != target_device:
            enriched_input = enriched_input.to(target_device)
        return self.base_encoder(enriched_input)

    def z_transformation(self, untran_z: torch.Tensor) -> torch.Tensor:
        return self.base_encoder.z_transformation(untran_z)


def build_graph_encoder(
    n_latent: int,
    n_layers: int,
    n_hidden: int,
    dropout_rate: float,
    latent_distribution: str,
    use_batch_norm: bool,
    use_layer_norm: bool,
    var_activation,
    aggregator: NeighborMessageAggregator,
    encoder_input_dim: int,
) -> GraphAugmentedEncoder:
    """Factory helper to assemble graph-augmented encoder."""
    base_encoder = Encoder(
        n_input=encoder_input_dim,
        n_output=n_latent,
        n_layers=n_layers,
        n_hidden=n_hidden,
        dropout_rate=dropout_rate,
        distribution=latent_distribution,
        use_batch_norm=use_batch_norm,
        use_layer_norm=use_layer_norm,
        var_activation=var_activation,
    )
    return GraphAugmentedEncoder(base_encoder=base_encoder, aggregator=aggregator)


class GraphVELOVAE(VELOVAE):
    """VELOVAE extension with graph-based message passing in the encoder."""

    def __init__(
        self,
        *,
        spliced_full: np.ndarray,
        unspliced_full: np.ndarray,
        gnn_hidden_dim: int = 128,
        gnn_dropout_rate: float = 0.1,
        latent_distribution: str = "normal",
        use_batch_norm: str = "both",
        use_layer_norm: str = "both",
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
        var_activation=None,
        **super_kwargs,
    ):
        self._gnn_hidden_dim = gnn_hidden_dim
        self._gnn_dropout_rate = gnn_dropout_rate
        self._gnn_use_batch_norm = use_batch_norm
        self._gnn_use_layer_norm = use_layer_norm
        self._gnn_n_layers = n_layers
        self._gnn_n_hidden = n_hidden
        self._gnn_dropout_rate_encoder = dropout_rate
        self._gnn_latent_distribution = latent_distribution
        self._gnn_var_activation = var_activation
        self.velocity_laplacian_weight = float(super_kwargs.pop("velocity_laplacian_weight", 0.0))
        self.velocity_angle_weight = float(super_kwargs.pop("velocity_angle_weight", 0.0))
        self.velocity_angle_eps = float(super_kwargs.pop("velocity_angle_eps", 1e-6))
        self.velocity_penalty_mode = super_kwargs.pop("velocity_penalty_mode", "spliced")
        self._velocity_regularization_enabled = (
            self.velocity_laplacian_weight > 0.0 or self.velocity_angle_weight > 0.0
        )
        self.continuity_weight = float(super_kwargs.pop("gnn_continuity_weight", 0.0))
        self.continuity_horizon = float(super_kwargs.pop("gnn_continuity_horizon", 1.0))
        self._continuity_enabled = self.continuity_weight > 0.0

        gnn_use_attention = super_kwargs.pop("gnn_use_attention", False)
        gnn_use_gate = super_kwargs.pop("gnn_use_gate", False)
        gnn_use_residual = super_kwargs.pop("gnn_use_residual", False)
        gnn_use_differences = super_kwargs.pop("gnn_use_differences", False)

        super().__init__(
            n_hidden=n_hidden,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
            latent_distribution=latent_distribution,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
            var_activation=var_activation,
            **super_kwargs,
        )

        aggregator = NeighborMessageAggregator(
            spliced_full=spliced_full,
            unspliced_full=unspliced_full,
            hidden_dim=gnn_hidden_dim,
            log_variational=self.log_variational,
            dropout_rate=gnn_dropout_rate,
            use_attention=gnn_use_attention,
            use_gate=gnn_use_gate,
            use_residual=gnn_use_residual,
            use_differences=gnn_use_differences,
        )
        use_batch_norm_encoder = use_batch_norm in ("encoder", "both")
        use_layer_norm_encoder = use_layer_norm in ("encoder", "both")
        encoder_input_dim = (self.n_input * 2) + gnn_hidden_dim
        var_activation_fn = var_activation if var_activation is not None else torch.nn.Softplus()

        self.z_encoder = build_graph_encoder(
            n_latent=self.n_latent,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            latent_distribution=latent_distribution,
            use_batch_norm=use_batch_norm_encoder,
            use_layer_norm=use_layer_norm_encoder,
            var_activation=var_activation_fn,
            aggregator=aggregator,
            encoder_input_dim=encoder_input_dim,
        )
        self.register_buffer(
            "_continuity_spliced",
            torch.as_tensor(spliced_full, dtype=torch.float32),
            persistent=False,
        )

    def _get_inference_input(self, tensors):
        input_dict = super()._get_inference_input(tensors)
        input_dict[VELOVI_REGISTRY_KEYS.NEIGHBOR_INDEX_KEY] = tensors[
            VELOVI_REGISTRY_KEYS.NEIGHBOR_INDEX_KEY
        ]
        input_dict[VELOVI_REGISTRY_KEYS.NEIGHBOR_WEIGHT_KEY] = tensors.get(
            VELOVI_REGISTRY_KEYS.NEIGHBOR_WEIGHT_KEY
        )
        # Optional latent-graph neighbor inputs for dual-graph aggregation
        if VELOVI_REGISTRY_KEYS.NEIGHBOR_INDEX_KEY_LATENT in tensors:
            input_dict[VELOVI_REGISTRY_KEYS.NEIGHBOR_INDEX_KEY_LATENT] = tensors[
                VELOVI_REGISTRY_KEYS.NEIGHBOR_INDEX_KEY_LATENT
            ]
        if VELOVI_REGISTRY_KEYS.NEIGHBOR_WEIGHT_KEY_LATENT in tensors:
            input_dict[VELOVI_REGISTRY_KEYS.NEIGHBOR_WEIGHT_KEY_LATENT] = tensors.get(
                VELOVI_REGISTRY_KEYS.NEIGHBOR_WEIGHT_KEY_LATENT
            )
        if VELOVI_REGISTRY_KEYS.FUTURE_INDEX_KEY in tensors:
            input_dict[VELOVI_REGISTRY_KEYS.FUTURE_INDEX_KEY] = tensors[
                VELOVI_REGISTRY_KEYS.FUTURE_INDEX_KEY
            ]
        if VELOVI_REGISTRY_KEYS.FUTURE_WEIGHT_KEY in tensors:
            input_dict[VELOVI_REGISTRY_KEYS.FUTURE_WEIGHT_KEY] = tensors.get(
                VELOVI_REGISTRY_KEYS.FUTURE_WEIGHT_KEY
            )
        return input_dict

    def inference(
        self,
        spliced,
        unspliced,
        neighbor_index,
        neighbor_weight=None,
        neighbor_index_latent=None,
        neighbor_weight_latent=None,
        future_index=None,
        future_weight=None,
        n_samples: int = 1,
    ):
        spliced_ = spliced
        unspliced_ = unspliced
        if self.log_variational:
            spliced_ = torch.log(0.01 + spliced)
            unspliced_ = torch.log(0.01 + unspliced)

        encoder_input = torch.cat((spliced_, unspliced_), dim=-1)

        if not torch.is_tensor(neighbor_index):
            neighbor_index = torch.as_tensor(neighbor_index, dtype=torch.long, device=encoder_input.device)
        else:
            neighbor_index = neighbor_index.to(encoder_input.device)

        if neighbor_weight is not None:
            if not torch.is_tensor(neighbor_weight):
                neighbor_weight = torch.as_tensor(
                    neighbor_weight, dtype=encoder_input.dtype, device=encoder_input.device
                )
            else:
                neighbor_weight = neighbor_weight.to(encoder_input.device)
        if neighbor_index_latent is not None:
            if not torch.is_tensor(neighbor_index_latent):
                neighbor_index_latent = torch.as_tensor(
                    neighbor_index_latent, dtype=torch.long, device=encoder_input.device
                )
            else:
                neighbor_index_latent = neighbor_index_latent.to(encoder_input.device)
        if neighbor_weight_latent is not None:
            if not torch.is_tensor(neighbor_weight_latent):
                neighbor_weight_latent = torch.as_tensor(
                    neighbor_weight_latent, dtype=encoder_input.dtype, device=encoder_input.device
                )
            else:
                neighbor_weight_latent = neighbor_weight_latent.to(encoder_input.device)

        qz_m, qz_v, z = self.z_encoder(
            encoder_input,
            neighbor_index,
            neighbor_weight,
            neighbor_index2=neighbor_index_latent,
            neighbor_weight2=neighbor_weight_latent,
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

    def loss(
        self,
        tensors,
        inference_outputs,
        generative_outputs,
        kl_weight: float = 1.0,
        n_obs: float = 1.0,
    ):
        spliced = tensors[VELOVI_REGISTRY_KEYS.X_KEY]
        unspliced = tensors[VELOVI_REGISTRY_KEYS.U_KEY]

        qz_m = inference_outputs[MODULE_KEYS.QZM_KEY]
        qz_v = inference_outputs[MODULE_KEYS.QZV_KEY]

        px_pi = generative_outputs["px_pi"]
        px_pi_alpha = generative_outputs["px_pi_alpha"]

        end_penalty = generative_outputs["end_penalty"]
        mixture_dist_s = generative_outputs["mixture_dist_s"]
        mixture_dist_u = generative_outputs["mixture_dist_u"]

        kl_divergence_z = kl(Normal(qz_m, torch.sqrt(qz_v)), Normal(0, 1)).sum(dim=1)

        reconst_loss_s = -mixture_dist_s.log_prob(spliced)
        reconst_loss_u = -mixture_dist_u.log_prob(unspliced)

        reconst_loss = reconst_loss_u.sum(dim=-1) + reconst_loss_s.sum(dim=-1)

        kl_pi = kl(
            Dirichlet(px_pi_alpha),
            Dirichlet(self.dirichlet_concentration * torch.ones_like(px_pi)),
        ).sum(dim=-1)

        kl_local = kl_divergence_z + kl_pi
        weighted_kl_local = kl_weight * kl_divergence_z + kl_pi

        local_loss = torch.mean(reconst_loss + weighted_kl_local)

        loss = local_loss + self.penalty_scale * (1 - kl_weight) * end_penalty

        extra_metrics: dict[str, torch.Tensor] = {}

        if self._velocity_regularization_enabled:
            velocity_field = self._compute_batch_velocity(
                inference_outputs,
                generative_outputs,
                mode=self.velocity_penalty_mode,
            )
            neighbor_index_primary = tensors.get(VELOVI_REGISTRY_KEYS.NEIGHBOR_INDEX_KEY)
            neighbor_weight_primary = tensors.get(VELOVI_REGISTRY_KEYS.NEIGHBOR_WEIGHT_KEY)
            neighbor_index_latent = tensors.get(VELOVI_REGISTRY_KEYS.NEIGHBOR_INDEX_KEY_LATENT)
            neighbor_weight_latent = tensors.get(VELOVI_REGISTRY_KEYS.NEIGHBOR_WEIGHT_KEY_LATENT)
            cell_indices = tensors.get(REGISTRY_KEYS.INDICES_KEY)
            future_index = tensors.get(VELOVI_REGISTRY_KEYS.FUTURE_INDEX_KEY)
            future_weight = tensors.get(VELOVI_REGISTRY_KEYS.FUTURE_WEIGHT_KEY)

            penalties = self._compute_velocity_penalties(
                velocity_field,
                cell_indices,
                neighbor_index_primary,
                neighbor_weight_primary,
                neighbor_index_latent,
                neighbor_weight_latent,
            )

            if self.velocity_laplacian_weight > 0.0 and penalties["laplacian"] is not None:
                laplacian_penalty = penalties["laplacian"]
                loss = loss + self.velocity_laplacian_weight * laplacian_penalty
                extra_metrics["velocity_laplacian"] = laplacian_penalty.detach()
            if self.velocity_angle_weight > 0.0 and penalties["angular"] is not None:
                angular_penalty = penalties["angular"]
                loss = loss + self.velocity_angle_weight * angular_penalty
                extra_metrics["velocity_angular"] = angular_penalty.detach()
            if self._continuity_enabled:
                continuity_penalty = self._compute_continuity_penalty(
                    velocity_field,
                    spliced,
                    cell_indices,
                    future_index,
                    future_weight,
                )
                if continuity_penalty is not None:
                    loss = loss + self.continuity_weight * continuity_penalty
                    extra_metrics["gnn_continuity"] = continuity_penalty.detach()

        loss_recoder = LossOutput(
            loss=loss,
            reconstruction_loss=reconst_loss,
            kl_local=kl_local,
            extra_metrics=extra_metrics,
        )
        return loss_recoder

    def load_state_dict(self, state_dict, strict: bool = True):
        """Load weights while tolerating optional GNN components.

        Older checkpoints may lack the auxiliary aggregator modules (gate/attention/residual),
        and newer checkpoints may include them even if the current configuration disables them.
        We intersect the checkpoint with the current module structure before delegating to the
        parent loader to avoid spurious incompatibility errors.
        """
        current_state = super().state_dict()
        expected_keys = set(current_state.keys())
        filtered_state = {k: v for k, v in state_dict.items() if k in expected_keys}
        unexpected_keys = [k for k in state_dict if k not in expected_keys]

        if unexpected_keys:
            warnings.warn(
                (
                    "Ignoring unexpected parameters when loading GraphVELOVAE checkpoint: "
                    f"{', '.join(unexpected_keys[:10])}"
                    + ("..." if len(unexpected_keys) > 10 else "")
                ),
                RuntimeWarning,
            )

        load_result = super().load_state_dict(filtered_state, strict=False)
        load_result.unexpected_keys = list(unexpected_keys)

        if strict and load_result.missing_keys:
            warnings.warn(
                (
                    "Checkpoint is missing parameters required by the current GraphVELOVAE "
                    f"configuration: {', '.join(load_result.missing_keys[:10])}"
                    + ("..." if len(load_result.missing_keys) > 10 else "")
                ),
                RuntimeWarning,
            )
        return load_result

    def _compute_batch_velocity(
        self,
        inference_outputs: dict[str, torch.Tensor],
        generative_outputs: dict[str, torch.Tensor],
        mode: str = "spliced",
    ) -> torch.Tensor:
        pi = generative_outputs["px_pi"]
        px_tau = generative_outputs["px_tau"]
        px_rho = generative_outputs["px_rho"]

        gamma = inference_outputs["gamma"]
        beta = inference_outputs["beta"]
        alpha = inference_outputs["alpha"]
        alpha_1 = inference_outputs["alpha_1"]
        lambda_alpha = inference_outputs["lambda_alpha"]

        switch_time = torch.clamp(F.softplus(self.switch_time_unconstr), 0, self.t_max)

        ind_prob = pi[..., 0]
        steady_prob = pi[..., 1]
        rep_prob = pi[..., 2]

        ind_time = switch_time * px_rho
        rep_time = (self.t_max - switch_time) * px_tau

        mean_u_ind, mean_s_ind = self._get_induction_unspliced_spliced(
            alpha, alpha_1, lambda_alpha, beta, gamma, ind_time
        )
        mean_u_rep, mean_s_rep = self._get_repression_unspliced_spliced(
            mean_u_ind,
            mean_s_ind,
            beta,
            gamma,
            rep_time,
        )

        if mode == "spliced":
            velo_ind = beta * mean_u_ind - gamma * mean_s_ind
            velo_rep = beta * mean_u_rep - gamma * mean_s_rep
            velo_steady = torch.zeros_like(velo_ind)
        else:
            transcription_rate = alpha_1 - (alpha_1 - alpha) * torch.exp(-lambda_alpha * ind_time)
            velo_ind = transcription_rate - beta * mean_u_ind
            velo_rep = -beta * mean_u_rep
            velo_steady = torch.zeros_like(velo_ind)

        velocity = ind_prob * velo_ind + rep_prob * velo_rep + steady_prob * velo_steady
        return velocity

    def _compute_velocity_penalties(
        self,
        velocities: torch.Tensor,
        cell_indices: torch.Tensor | None,
        neighbor_index_primary: torch.Tensor | None,
        neighbor_weight_primary: torch.Tensor | None,
        neighbor_index_latent: torch.Tensor | None,
        neighbor_weight_latent: torch.Tensor | None,
    ) -> dict[str, torch.Tensor | None]:
        device = velocities.device
        dtype = velocities.dtype

        zero = velocities.new_zeros(())
        if cell_indices is None:
            return {"laplacian": zero, "angular": zero if self.velocity_angle_weight > 0.0 else None}

        index_array = cell_indices.view(-1).detach().cpu().numpy()
        index_lookup = {int(idx): pos for pos, idx in enumerate(index_array)}

        laplacian = velocities.new_tensor(0.0)
        laplacian_weight_sum = velocities.new_tensor(0.0)
        angular = velocities.new_tensor(0.0)
        angular_weight_sum = velocities.new_tensor(0.0)

        def process_graph(graph_index, graph_weight):
            nonlocal laplacian, laplacian_weight_sum, angular, angular_weight_sum
            if graph_index is None:
                return

            graph_index_np = graph_index.detach().cpu().numpy()
            if graph_weight is not None:
                graph_weight_np = graph_weight.detach().cpu().numpy()
            else:
                graph_weight_np = None

            for i, neighbor_row in enumerate(graph_index_np):
                valid_positions: list[int] = []
                valid_weights: list[float] = []
                for j, neighbor_idx in enumerate(neighbor_row):
                    pos = index_lookup.get(int(neighbor_idx))
                    if pos is None:
                        continue
                    valid_positions.append(pos)
                    if graph_weight_np is not None:
                        valid_weights.append(float(graph_weight_np[i, j]))
                if not valid_positions:
                    continue
                v_i = velocities[i]
                v_neighbors = velocities[valid_positions]
                if graph_weight_np is not None:
                    weights_tensor = torch.tensor(valid_weights, device=device, dtype=dtype)
                else:
                    weights_tensor = torch.full(
                        (len(valid_positions),),
                        1.0 / len(valid_positions),
                        device=device,
                        dtype=dtype,
                    )
                diff = v_i.unsqueeze(0) - v_neighbors
                laplacian = laplacian + (weights_tensor * diff.pow(2).sum(dim=1)).sum()
                laplacian_weight_sum = laplacian_weight_sum + weights_tensor.sum()
                if self.velocity_angle_weight > 0.0:
                    vi_norm = torch.norm(v_i) + self.velocity_angle_eps
                    vj_norm = torch.norm(v_neighbors, dim=1) + self.velocity_angle_eps
                    dot = torch.sum(v_neighbors * v_i.unsqueeze(0), dim=1)
                    cos = dot / (vi_norm * vj_norm)
                    cos = torch.clamp(cos, -1.0, 1.0)
                    angular = angular + (weights_tensor * (1.0 - cos)).sum()
                    angular_weight_sum = angular_weight_sum + weights_tensor.sum()

        process_graph(neighbor_index_primary, neighbor_weight_primary)
        process_graph(neighbor_index_latent, neighbor_weight_latent)

        if laplacian_weight_sum.item() > 0:
            laplacian = laplacian / laplacian_weight_sum
        else:
            laplacian = zero

        if self.velocity_angle_weight > 0.0:
            if angular_weight_sum.item() > 0:
                angular = angular / angular_weight_sum
            else:
                angular = zero
        else:
            angular = zero

        return {"laplacian": laplacian, "angular": angular}

    def _compute_continuity_penalty(
        self,
        velocities: torch.Tensor,
        spliced_batch: torch.Tensor,
        cell_indices: torch.Tensor | None,
        future_index: torch.Tensor | None,
        future_weight: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if (
            not self._continuity_enabled
            or cell_indices is None
            or future_index is None
            or self._continuity_spliced is None
        ):
            return None
        device = velocities.device
        dtype = velocities.dtype
        future_index = future_index.to(device=device, dtype=torch.long)
        if future_weight is not None:
            future_weight = future_weight.to(device=device, dtype=dtype)
        bsz, k_neighbors = future_index.shape
        ref = self._continuity_spliced.to(device=device)
        future_states = torch.index_select(ref, 0, future_index.view(-1)).view(bsz, k_neighbors, -1)
        predicted = spliced_batch.to(device=device) + self.continuity_horizon * velocities
        diff = predicted.unsqueeze(1) - future_states
        sq = diff.pow(2).sum(dim=-1)
        if future_weight is None:
            weights = torch.full((bsz, k_neighbors), 1.0 / max(1, k_neighbors), device=device, dtype=dtype)
        else:
            weights = future_weight
        weight_sum = weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        per_cell = (weights * sq).sum(dim=1) / weight_sum.squeeze(1)
        return per_cell.mean()
