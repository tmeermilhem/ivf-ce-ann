"""Standard IVF index implemented with NumPy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np

from src.clustering import train_kmeans
from src.index.base_index import BaseIndex
from src.utils import pairwise_squared_l2


@dataclass
class InvertedList:
    ids: np.ndarray  # shape: (n_members,)


class IVFIndex(BaseIndex):
    """Inverted File Index without cross-cluster links."""

    def __init__(
        self,
        dimension: int,
        *,
        n_clusters: int,
        n_init: int = 10,
        max_iter: int = 100,
        seed: int | None = None,
    ) -> None:
        super().__init__(dimension=dimension)
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.seed = seed

        self.centroids: np.ndarray | None = None
        self.database: np.ndarray | None = None
        self.inverted_lists: Dict[int, InvertedList] = {}
        self.assignments: np.ndarray | None = None

    def build(self, vectors: np.ndarray) -> None:
        self._validate_input(vectors)
        result = train_kmeans(
            vectors,
            n_clusters=self.n_clusters,
            n_init=self.n_init,
            max_iter=self.max_iter,
            seed=self.seed,
        )

        self.centroids = result.centroids
        self.database = vectors.astype(np.float32, copy=True)
        self.assignments = result.assignments.astype(np.int32)
        self._build_inverted_lists(result.assignments)
        self.is_built = True

    def assign(self, vectors: np.ndarray) -> np.ndarray:
        self._ensure_ready()
        self._validate_input(vectors)

        distances = pairwise_squared_l2(vectors, self.centroids)
        return np.argmin(distances, axis=1).astype(np.int32)

    def search_cluster(
        self, cluster_id: int, query: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return vector ids and squared distances for a query within a cluster."""
        self._ensure_ready()

        member_ids = self.get_cluster_member_ids(cluster_id)
        if member_ids.size == 0:
            return member_ids, np.empty(0, dtype=np.float32)

        vectors = self.database[member_ids]
        diffs = vectors - query
        distances = np.einsum("ij,ij->i", diffs, diffs, dtype=np.float32)
        return member_ids, distances

    def get_cluster_member_ids(self, cluster_id: int) -> np.ndarray:
        self._ensure_ready()
        inverted_list = self.inverted_lists.get(cluster_id)
        if inverted_list is None:
            return np.empty(0, dtype=np.int32)
        return inverted_list.ids

    def nearest_centroids(
        self, query: np.ndarray, nprobe: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return the closest ``nprobe`` centroids to the query."""
        self._ensure_ready()
        if query.shape[0] != self.dimension:
            raise ValueError("query dimension does not match index dimension")
        if nprobe <= 0:
            raise ValueError("nprobe must be positive")

        diffs = self.centroids - query
        distances = np.einsum("ij,ij->i", diffs, diffs, dtype=np.float32)
        nprobe = min(nprobe, self.centroids.shape[0])
        idx = np.argpartition(distances, kth=nprobe - 1)[:nprobe]
        ordering = np.argsort(distances[idx])
        return idx[ordering], distances[idx][ordering]

    def _build_inverted_lists(self, assignments: np.ndarray) -> None:
        self.inverted_lists.clear()
        for cluster_id in range(self.n_clusters):
            member_ids = np.where(assignments == cluster_id)[0].astype(np.int32)
            self.inverted_lists[cluster_id] = InvertedList(ids=member_ids)

    def _ensure_ready(self) -> None:
        if (
            not self.is_built
            or self.centroids is None
            or self.database is None
            or getattr(self, "assignments", None) is None
        ):
            raise RuntimeError("Index has not been built yet.")
