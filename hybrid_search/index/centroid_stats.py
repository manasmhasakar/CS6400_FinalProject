import os
import pickle
from typing import Dict, List

import numpy as np
from pyroaring import BitMap


class CentroidStats:
    """
    Stores per-centroid statistics for filter-aware search.
    Maps centroid_id -> {filter_key: count}.
    """
    
    def __init__(self, nlist: int):
        self.nlist = nlist
        # centroid_id -> BitMap of doc IDs
        self.centroid_docs: List[BitMap] = [BitMap() for _ in range(nlist)]
    
    def add_assignments(self, doc_ids: np.ndarray, centroid_assignments: np.ndarray) -> None:
        """
        Record which documents belong to which centroids.
        doc_ids: array of document IDs (int)
        centroid_assignments: array of centroid IDs (int), same length as doc_ids
        """
        for doc_id, centroid_id in zip(doc_ids, centroid_assignments):
            self.centroid_docs[int(centroid_id)].add(int(doc_id))
    
    def get_eligible_counts(self, eligible_ids: BitMap) -> np.ndarray:
        """
        For each centroid, count how many docs are in the eligible set.
        Returns: array of shape (nlist,) with counts.
        """
        counts = np.zeros(self.nlist, dtype=np.int32)
        for i, centroid_bm in enumerate(self.centroid_docs):
            counts[i] = len(centroid_bm & eligible_ids)
        return counts
    
    def rank_centroids_by_density(self, eligible_ids: BitMap) -> np.ndarray:
        """
        Return centroid IDs sorted by descending eligible count.
        Returns: array of centroid IDs.
        """
        counts = self.get_eligible_counts(eligible_ids)
        return np.argsort(-counts)
    
    def save(self, path: str) -> None:
        """Save centroid stats to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load(path: str) -> "CentroidStats":
        """Load centroid stats from disk."""
        with open(path, "rb") as f:
            return pickle.load(f)


def build_centroid_stats(
    index,  # IVFFlatIndex or IVFPQIndex
    vectors: np.ndarray,
    doc_ids: np.ndarray,
) -> CentroidStats:
    """
    Build centroid statistics given an index and vectors.
    """
    nlist = index.nlist
    stats = CentroidStats(nlist)
    
    assignments = index.get_list_assignments(vectors)
    stats.add_assignments(doc_ids, assignments)
    
    return stats

