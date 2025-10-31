from typing import Optional, Tuple

import numpy as np
from pyroaring import BitMap

from hybrid_search.index.centroid_stats import CentroidStats


class HybridExecutor:
    """
    Filter-aware IVF search with adaptive nprobe.
    Supports centroid ranking and early termination.
    """
    
    def __init__(self, index, centroid_stats: Optional[CentroidStats] = None):
        """
        index: IVFFlatIndex or IVFPQIndex
        centroid_stats: optional CentroidStats for filter-aware probing
        """
        self.index = index
        self.centroid_stats = centroid_stats
    
    def search(
        self,
        query: np.ndarray,
        k: int,
        nprobe: int,
        eligible_ids: Optional[BitMap] = None,
        max_nprobe: int = 256,
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Search with filter and adaptive nprobe.
        
        query: (dim,) vector
        k: number of results
        nprobe: initial nprobe
        eligible_ids: bitmap of allowed IDs
        max_nprobe: cap for adaptive doubling
        
        Returns: (distances, indices, stats)
        stats: {"nprobe_used": int, "candidates_scored": int}
        """
        if query.ndim == 1:
            query = query.reshape(1, -1)
        
        current_nprobe = nprobe
        stats = {"nprobe_used": 0, "candidates_scored": 0}
        
        while current_nprobe <= max_nprobe:
            # Search with current nprobe
            distances, indices = self.index.search(query, k=k * 5, nprobe=current_nprobe)  # fetch more candidates
            
            # Flatten results
            dists_flat = distances.flatten()
            ids_flat = indices.flatten()
            
            # Filter by eligible_ids
            if eligible_ids is not None:
                mask = np.array([int(i) in eligible_ids for i in ids_flat], dtype=bool)
                dists_flat = dists_flat[mask]
                ids_flat = ids_flat[mask]
            
            stats["nprobe_used"] = current_nprobe
            stats["candidates_scored"] = len(ids_flat)
            
            # Check if we have enough results
            if len(ids_flat) >= k:
                # Sort and take top k
                top_k_idx = np.argsort(-dists_flat)[:k]  # descending for inner product
                final_dists = dists_flat[top_k_idx].reshape(1, -1)
                final_ids = ids_flat[top_k_idx].reshape(1, -1)
                return final_dists, final_ids, stats
            
            # Adaptive doubling
            if current_nprobe >= max_nprobe:
                break
            current_nprobe = min(current_nprobe * 2, max_nprobe)
        
        # Return what we have (may be < k)
        if len(ids_flat) > 0:
            top_k_idx = np.argsort(-dists_flat)[:k]
            final_dists = dists_flat[top_k_idx].reshape(1, -1)
            final_ids = ids_flat[top_k_idx].reshape(1, -1)
        else:
            final_dists = np.zeros((1, k), dtype=np.float32)
            final_ids = np.full((1, k), -1, dtype=np.int64)
        
        return final_dists, final_ids, stats

