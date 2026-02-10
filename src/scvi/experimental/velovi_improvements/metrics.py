from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


@dataclass
class VelocityMetricResult:
    gene_likelihood_mean: float
    velocity_norm_mean: float
    velocity_local_smoothness: float
    velocity_cosine_alignment: Optional[float] = None


def compute_velocity_norm(velocities: np.ndarray) -> float:
    return float(np.linalg.norm(velocities, axis=1).mean())


def compute_local_smoothness(
    velocities: np.ndarray,
    neighbor_indices: np.ndarray,
) -> float:
    diffs = velocities[:, None, :] - velocities[neighbor_indices]
    return float(np.linalg.norm(diffs, axis=2).mean())


def compute_cosine_alignment(
    velocities: np.ndarray,
    target_vectors: np.ndarray,
) -> float:
    dot = (velocities * target_vectors).sum(axis=1)
    denom = np.linalg.norm(velocities, axis=1) * np.linalg.norm(target_vectors, axis=1) + 1e-12
    cos = dot / denom
    return float(np.clip(cos, -1.0, 1.0).mean())


def summarize_velocity_metrics(
    velocities: np.ndarray,
    neighbor_indices: np.ndarray,
    gene_likelihood_mean: float,
    alignment_vectors: np.ndarray | None = None,
) -> VelocityMetricResult:
    result = VelocityMetricResult(
        gene_likelihood_mean=gene_likelihood_mean,
        velocity_norm_mean=compute_velocity_norm(velocities),
        velocity_local_smoothness=compute_local_smoothness(velocities, neighbor_indices),
    )
    if alignment_vectors is not None:
        result.velocity_cosine_alignment = compute_cosine_alignment(
            velocities=velocities, target_vectors=alignment_vectors
        )
    return result


def result_to_dict(result: VelocityMetricResult) -> Dict[str, float]:
    data = {
        "gene_likelihood_mean": result.gene_likelihood_mean,
        "velocity_norm_mean": result.velocity_norm_mean,
        "velocity_local_smoothness": result.velocity_local_smoothness,
    }
    if result.velocity_cosine_alignment is not None:
        data["velocity_cosine_alignment"] = result.velocity_cosine_alignment
    return data
