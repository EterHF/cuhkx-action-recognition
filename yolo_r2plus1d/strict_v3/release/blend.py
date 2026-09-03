"""Strict-v3 confidence-weighted branch fusion."""

from collections.abc import Mapping

import numpy as np


def confidence(logits: np.ndarray, temperature: float) -> np.ndarray:
    scaled = logits / max(float(temperature), 1e-4)
    scaled -= scaled.max(axis=1, keepdims=True)
    probability = np.exp(scaled)
    probability /= probability.sum(axis=1, keepdims=True)
    return probability.max(axis=1)


def apply_gate(
    branches: Mapping[str, np.ndarray],
    temperatures: Mapping[str, float],
    base_weights: Mapping[str, float],
    quality_floor: float = 0.25,
    quality_gamma: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    names = ("fusion", "visual", "temporal")
    rows = len(next(iter(branches.values())))
    if not 0.0 <= quality_floor <= 1.0:
        raise ValueError("quality_floor must be in [0, 1]")
    if not np.isfinite(quality_gamma) or quality_gamma <= 0.0:
        raise ValueError("quality_gamma must be finite and positive")
    quality = np.stack(
        [confidence(branches[name], temperatures[name]) for name in names], axis=1
    )
    chance = 1.0 / 40.0
    quality = np.clip((quality - chance) / (1.0 - chance), 0.0, 1.0)
    quality = quality_floor + (1.0 - quality_floor) * quality**quality_gamma
    weights = np.asarray(
        [float(base_weights[name]) for name in names], dtype=np.float64
    )[None, :]
    weights = np.broadcast_to(weights, (rows, len(names))).copy() * quality
    weights[:, 1] = np.maximum(weights[:, 1], 1e-8)
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
    fused = sum(
        weights[:, index, None] * branches[name] / max(float(temperatures[name]), 1e-4)
        for index, name in enumerate(names)
    )
    return fused, weights
