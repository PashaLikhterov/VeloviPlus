from __future__ import annotations

import numpy as np

from scvi.data import AnnDataManager
from scvi.data.fields import LayerField, ObsmField
from scvi.model.base import BaseModelClass

from ._constants import VELOVI_REGISTRY_KEYS
from ._model import VELOVI, _softplus_inverse
from ._transformer_module import TransformerVELOVAE


class VELOVITransformerEncoder(VELOVI):
    """VELOVI variant whose encoder is a neighborhood-aware transformer."""

    def __init__(
        self,
        adata,
        n_hidden: int = 256,
        n_latent: int = 10,
        n_layers: int = 1,
        dropout_rate: float = 0.1,
        gamma_init_data: bool = False,
        linear_decoder: bool = False,
        transformer_hidden_dim: int = 256,
        transformer_heads: int = 4,
        transformer_layers: int = 2,
        transformer_dropout: float = 0.1,
        transformer_max_neighbors: int | None = None,
        transformer_neighbor_reg_weight: float = 0.0,
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

        transformer_kwargs = dict(
            spliced_full=spliced,
            unspliced_full=unspliced,
            transformer_hidden_dim=transformer_hidden_dim,
            transformer_heads=transformer_heads,
            transformer_layers=transformer_layers,
            transformer_dropout=transformer_dropout,
            transformer_max_neighbors=transformer_max_neighbors,
            transformer_neighbor_reg_weight=transformer_neighbor_reg_weight,
        )

        self.module = TransformerVELOVAE(
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
            **transformer_kwargs,
            **model_kwargs,
        )
        self._model_summary_string = (
            f"VELOVI Transformer Encoder with params: \n"
            f"n_hidden: {n_hidden}, n_latent: {n_latent}, n_layers: {n_layers}, dropout_rate: {dropout_rate}, "
            f"transformer_hidden_dim: {transformer_hidden_dim}, transformer_layers: {transformer_layers}, "
            f"transformer_heads: {transformer_heads}"
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
        **kwargs,
    ):
        """Register AnnData fields including neighbor information for transformer encoder."""
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
        adata_manager = AnnDataManager(fields=anndata_fields, setup_method_args=setup_method_args)
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)
