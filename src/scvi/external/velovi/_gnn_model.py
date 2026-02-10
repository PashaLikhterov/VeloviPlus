from __future__ import annotations

import numpy as np

from scvi.data import AnnDataManager
from scvi.data.fields import LayerField, ObsmField
from scvi.model.base import BaseModelClass

from ._constants import VELOVI_REGISTRY_KEYS
from ._gnn_module import GraphVELOVAE
from ._model import VELOVI, _softplus_inverse


class VELOVIWithGNN(VELOVI):
    """VELOVI variant integrating graph neural network message passing in the encoder."""

    def __init__(
        self,
        adata,
        n_hidden: int = 256,
        n_latent: int = 10,
        n_layers: int = 1,
        dropout_rate: float = 0.1,
        gamma_init_data: bool = False,
        linear_decoder: bool = False,
        gnn_hidden_dim: int = 128,
        gnn_dropout_rate: float = 0.1,
        **model_kwargs,
    ):
        BaseModelClass.__init__(self, adata)
        self.n_latent = n_latent

        adata_manager = self.adata_manager
        spliced = adata_manager.get_from_registry(VELOVI_REGISTRY_KEYS.X_KEY)
        unspliced = adata_manager.get_from_registry(VELOVI_REGISTRY_KEYS.U_KEY)

        sorted_unspliced = np.argsort(unspliced, axis=0)
        ind = int(adata.n_obs * 0.99)
        us_upper_ind = sorted_unspliced[ind:, :]

        us_upper = []
        ms_upper = []
        for i in range(len(us_upper_ind)):
            row = us_upper_ind[i]
            us_upper += [unspliced[row, np.arange(adata.n_vars)][np.newaxis, :]]
            ms_upper += [spliced[row, np.arange(adata.n_vars)][np.newaxis, :]]
        us_upper = np.median(np.concatenate(us_upper, axis=0), axis=0)
        ms_upper = np.median(np.concatenate(ms_upper, axis=0), axis=0)

        alpha_unconstr = _softplus_inverse(us_upper)
        alpha_unconstr = np.asarray(alpha_unconstr).ravel()

        alpha_1_unconstr = np.zeros(us_upper.shape).ravel()
        lambda_alpha_unconstr = np.zeros(us_upper.shape).ravel()

        if gamma_init_data:
            gamma_unconstr = np.clip(_softplus_inverse(us_upper / ms_upper), None, 10)
        else:
            gamma_unconstr = None

        velocity_laplacian_weight = model_kwargs.pop("velocity_laplacian_weight", 0.0)
        velocity_angle_weight = model_kwargs.pop("velocity_angle_weight", 0.0)
        velocity_angle_eps = model_kwargs.pop("velocity_angle_eps", 1e-6)
        velocity_penalty_mode = model_kwargs.pop("velocity_penalty_mode", "spliced")

        gnn_kwargs = dict(
            spliced_full=spliced,
            unspliced_full=unspliced,
            gnn_hidden_dim=gnn_hidden_dim,
            gnn_dropout_rate=gnn_dropout_rate,
            gnn_use_attention=model_kwargs.pop("gnn_use_attention", False),
            gnn_use_gate=model_kwargs.pop("gnn_use_gate", False),
            gnn_use_residual=model_kwargs.pop("gnn_use_residual", False),
            gnn_use_differences=model_kwargs.pop("gnn_use_differences", False),
            velocity_laplacian_weight=velocity_laplacian_weight,
            velocity_angle_weight=velocity_angle_weight,
            velocity_angle_eps=velocity_angle_eps,
            velocity_penalty_mode=velocity_penalty_mode,
        )

        self.module = GraphVELOVAE(
            n_input=self.summary_stats["n_vars"],
            n_hidden=n_hidden,
            n_latent=n_latent,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
            gamma_unconstr_init=gamma_unconstr,
            alpha_unconstr_init=alpha_unconstr,
            alpha_1_unconstr_init=alpha_1_unconstr,
            lambda_alpha_unconstr_init=lambda_alpha_unconstr,
            switch_spliced=ms_upper,
            switch_unspliced=us_upper,
            linear_decoder=linear_decoder,
            **gnn_kwargs,
            **model_kwargs,
        )
        self._model_summary_string = (
            f"VELOVI GNN Model with params: \n"
            f"n_hidden: {n_hidden}, n_latent: {n_latent}, "
            f"n_layers: {n_layers}, dropout_rate: {dropout_rate}, "
            f"gnn_hidden_dim: {gnn_hidden_dim}"
        )
        self.init_params_ = self._get_init_params(locals())

    @classmethod
    def setup_anndata(
        cls,
        adata,
        spliced_layer: str,
        unspliced_layer: str,
        neighbor_index_key: str,
        neighbor_weight_key: str | None = None,
        neighbor_index_key_latent: str | None = None,
        neighbor_weight_key_latent: str | None = None,
        future_index_key: str | None = None,
        future_weight_key: str | None = None,
        **kwargs,
    ):
        """Register AnnData fields including neighbor information for GNN message passing."""
        setup_method_args = cls._get_setup_method_args(**locals())
        anndata_fields = [
            LayerField(VELOVI_REGISTRY_KEYS.X_KEY, spliced_layer, is_count_data=False),
            LayerField(VELOVI_REGISTRY_KEYS.U_KEY, unspliced_layer, is_count_data=False),
            ObsmField(VELOVI_REGISTRY_KEYS.NEIGHBOR_INDEX_KEY, neighbor_index_key),
        ]
        if neighbor_weight_key is not None:
            anndata_fields.append(
                ObsmField(VELOVI_REGISTRY_KEYS.NEIGHBOR_WEIGHT_KEY, neighbor_weight_key)
            )
        if neighbor_index_key_latent is not None:
            anndata_fields.append(
                ObsmField(
                    VELOVI_REGISTRY_KEYS.NEIGHBOR_INDEX_KEY_LATENT,
                    neighbor_index_key_latent,
                )
            )
        if neighbor_weight_key_latent is not None:
            anndata_fields.append(
                ObsmField(
                    VELOVI_REGISTRY_KEYS.NEIGHBOR_WEIGHT_KEY_LATENT,
                    neighbor_weight_key_latent,
                )
            )
        if future_index_key is not None:
            anndata_fields.append(
                ObsmField(VELOVI_REGISTRY_KEYS.FUTURE_INDEX_KEY, future_index_key)
            )
        if future_weight_key is not None:
            anndata_fields.append(
                ObsmField(VELOVI_REGISTRY_KEYS.FUTURE_WEIGHT_KEY, future_weight_key)
            )
        adata_manager = AnnDataManager(fields=anndata_fields, setup_method_args=setup_method_args)
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)
