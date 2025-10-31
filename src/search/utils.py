"""Shared helpers for search implementations."""

from __future__ import annotations

from typing import Tuple

import numpy as np


def deduplicate_keep_best(
    ids: np.ndarray, distances: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Deduplicate candidate vectors keeping the smallest distance per id."""
    seen: dict[int, float] = {}
    for vid, dist in zip(ids, distances):
        best = seen.get(int(vid))
        if best is None or dist < best:
            seen[int(vid)] = float(dist)

    if not seen:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32)

    unique_ids = np.fromiter(seen.keys(), dtype=np.int32)
    unique_dists = np.fromiter(seen.values(), dtype=np.float32)
    ordering = np.argsort(unique_dists)
    return unique_ids[ordering], unique_dists[ordering]
