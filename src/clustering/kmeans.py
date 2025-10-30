"""K-means clustering utilities for coarse quantizer training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import faiss  # type: ignore
except ImportError:  # pragma: no cover - faiss not always installed in CI
    faiss = None

from sklearn.cluster import KMeans


@dataclass
class KMeansResult:
    centroids: np.ndarray
    assignments: np.ndarray


def train_kmeans(
    vectors: np.ndarray,
    *,
    n_clusters: int,
    n_init: int = 10,
    max_iter: int = 100,
    seed: Optional[int] = None,
) -> KMeansResult:
    """Train k-means using Faiss if available, otherwise scikit-learn."""
    if vectors.ndim != 2:
        raise ValueError("vectors must be a 2D array")
    if n_clusters <= 0:
        raise ValueError("n_clusters must be positive")
    if vectors.shape[0] < n_clusters:
        raise ValueError("Number of vectors must be at least n_clusters")

    if faiss is not None:
        return _train_faiss_kmeans(
            vectors,
            n_clusters=n_clusters,
            n_init=n_init,
            max_iter=max_iter,
            seed=seed,
        )

    return _train_sklearn_kmeans(
        vectors,
        n_clusters=n_clusters,
        n_init=n_init,
        max_iter=max_iter,
        seed=seed,
    )


def _train_faiss_kmeans(
    vectors: np.ndarray,
    *,
    n_clusters: int,
    n_init: int,
    max_iter: int,
    seed: Optional[int],
) -> KMeansResult:
    dim = vectors.shape[1]
    kmeans = faiss.Kmeans(
        d=dim,
        k=n_clusters,
        niter=max_iter,
        nredo=n_init,
        verbose=False,
        seed=seed or 0,
    )
    kmeans.train(vectors)
    distances, assignments = kmeans.index.search(vectors, 1)
    # Faiss returns assignments as int64 column vector.
    assignments = assignments[:, 0].astype(np.int32)
    centroids = np.array(kmeans.centroids, dtype=np.float32)
    return KMeansResult(centroids=centroids, assignments=assignments)


def _train_sklearn_kmeans(
    vectors: np.ndarray,
    *,
    n_clusters: int,
    n_init: int,
    max_iter: int,
    seed: Optional[int],
) -> KMeansResult:
    kmeans = KMeans(
        n_clusters=n_clusters,
        init="k-means++",
        n_init=n_init,
        max_iter=max_iter,
        random_state=seed,
        verbose=0,
    )
    assignments = kmeans.fit_predict(vectors)
    centroids = kmeans.cluster_centers_.astype(np.float32)
    return KMeansResult(centroids=centroids, assignments=assignments.astype(np.int32))
