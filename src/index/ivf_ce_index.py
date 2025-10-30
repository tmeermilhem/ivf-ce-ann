"""IVF-CE index that augments IVF with cross-cluster links."""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from src.index.cross_links import CrossLink, CrossLinkBuilder
from src.index.ivf_index import IVFIndex


class IVFCEIndex(IVFIndex):
    def __init__(
        self,
        dimension: int,
        *,
        n_clusters: int,
        k1: int,
        m_max: int,
        p_index: int,
        n_init: int = 10,
        max_iter: int = 100,
        seed: int | None = None,
    ) -> None:
        super().__init__(
            dimension=dimension,
            n_clusters=n_clusters,
            n_init=n_init,
            max_iter=max_iter,
            seed=seed,
        )
        self.k1 = k1
        self.m_max = m_max
        self.p_index = p_index
        self.cross_links: Dict[int, List[CrossLink]] = {}

    def build(self, vectors: np.ndarray) -> None:
        super().build(vectors)
        self.cross_links = self._build_cross_links()

    def _build_cross_links(self) -> Dict[int, List[CrossLink]]:
        builder = CrossLinkBuilder(
            self,
            k1=self.k1,
            m_max=self.m_max,
            p_index=self.p_index,
        )
        cross_links: Dict[int, List[CrossLink]] = {}
        for vector_id in range(self.database.shape[0]):
            cross_links[vector_id] = builder.build_for_vector(vector_id)
        return cross_links
