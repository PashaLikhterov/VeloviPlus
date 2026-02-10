from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.linalg import norm
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix


@dataclass
class LatentGraphConfig:
    """Configuration for latent embedding based similarity graph construction."""

    metric: Literal["cosine", "rbf"] = "cosine"
    sigma: float = 1.0
    temperature: float = 1.0
    normalize: bool = True
    sparsify_percentile: float | None = 99.0
    self_loops: bool = False


class LatentEmbeddingGraphBuilder:
    """Construct a cell-to-cell similarity graph from latent embeddings.

    Parameters
    ----------
    config
        Parameters controlling similarity computation and post-processing.
    """

    def __init__(self, config: LatentGraphConfig | None = None):
        self.config = config or LatentGraphConfig()

    def build(self, latent: np.ndarray) -> csr_matrix:
        """Compute a sparse row-stochastic similarity graph.

        Parameters
        ----------
        latent
            Array with shape ``(n_cells, latent_dim)`` holding latent embeddings from the encoder.

        Returns
        -------
        csr_matrix
            Row stochastic sparse matrix capturing transition probabilities between cells.
        """
        latent = np.asarray(latent, dtype=np.float32)
        n_cells = latent.shape[0]

        if self.config.metric == "cosine":
            normed = latent / (norm(latent, axis=1, keepdims=True) + 1e-8)
            sim = np.clip(normed @ normed.T, -1.0, 1.0)
            # ensure positivity for transition interpretation
            sim = (sim + 1.0) / 2.0
        elif self.config.metric == "rbf":
            distances = cdist(latent, latent, metric="euclidean")
            sim = np.exp(-(distances**2) / (2.0 * self.config.sigma**2 + 1e-12))
        else:
            raise ValueError(f"Unsupported metric `{self.config.metric}`.")

        if not self.config.self_loops:
            np.fill_diagonal(sim, 0.0)

        if self.config.temperature != 1.0:
            sim = sim / (self.config.temperature + 1e-12)
            sim = np.exp(sim)

        if self.config.sparsify_percentile is not None:
            threshold = np.percentile(sim, self.config.sparsify_percentile)
            mask = sim >= threshold
            sim = sim * mask

        if self.config.normalize:
            row_sum = sim.sum(axis=1, keepdims=True)
            zero_mask = row_sum.squeeze() == 0
            sim = sim / (row_sum + 1e-12)
            if np.any(zero_mask):
                sim[zero_mask] = 1.0 / n_cells

        sim_sparse = csr_matrix(sim)
        return sim_sparse


def smooth_velocities_with_graph(
    velocities: np.ndarray,
    transition: csr_matrix,
) -> np.ndarray:
    """Left-multiply velocities with transition probabilities for smoothing.

    Parameters
    ----------
    velocities
        Array with shape ``(n_cells, n_genes)`` representing raw velocity estimates.
    transition
        Row stochastic transition matrix produced from latent embeddings.
    """
    if not isinstance(transition, csr_matrix):
        raise TypeError("`transition` must be a `csr_matrix`.")
    velocities = np.asarray(velocities, dtype=np.float32)
    return transition @ velocities


def latent_smoothing_pipeline(
    latent: np.ndarray,
    velocities: np.ndarray,
    config: LatentGraphConfig | None = None,
) -> np.ndarray:
    """Convenience wrapper returning smoothed velocities directly."""
    graph = LatentEmbeddingGraphBuilder(config).build(latent=latent)
    return smooth_velocities_with_graph(velocities=velocities, transition=graph)
