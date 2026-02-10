from __future__ import annotations

import numpy as np
import torch
from anndata import AnnData


def add_velovi_outputs_to_adata(adata: AnnData, vae, velocity_key: str = "velocity") -> AnnData:
    """Return an AnnData copy populated with VELOVI-derived layers for scVelo compatibility."""
    adata_copy = adata.copy()
    latent_time = vae.get_latent_time(n_samples=25)
    velocities = vae.get_velocity(n_samples=25, velo_statistic="mean")

    t = latent_time
    scaling = 20.0 / np.maximum(t.max(0), 1e-12)

    adata_copy.layers[velocity_key] = velocities / scaling
    adata_copy.layers["latent_time_velovi"] = latent_time
    adata_copy.obs["latent_time"] = latent_time.values

    rates = vae.get_rates()
    adata_copy.var["fit_alpha"] = rates["alpha"] / scaling
    adata_copy.var["fit_beta"] = rates["beta"] / scaling
    adata_copy.var["fit_gamma"] = rates["gamma"] / scaling
    adata_copy.var["fit_t_"] = (
        torch.nn.functional.softplus(vae.module.switch_time_unconstr).detach().cpu().numpy()
        * scaling
    )
    adata_copy.layers["fit_t"] = latent_time.values * scaling[np.newaxis, :]
    adata_copy.var["fit_scaling"] = 1.0
    return adata_copy
