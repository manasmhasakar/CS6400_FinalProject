from typing import Optional, Tuple

import numpy as np
from pyroaring import BitMap


class PostfilterANNBaseline:
    """
    Baseline: run ANN on full dataset, then apply filter.
    """
    
    def __init__(self, index):
        """
        index: IVFFlatIndex or IVFPQIndex
        """
        self.index = index
    
    def search(
        self,
        query: np.ndarray,
        k: int,
        nprobe: int,
        eligible_ids: Optional[BitMap] = None,
        max_nprobe: int = 256,
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Run ANN, filter, return top-k eligible.
        
        Returns: (distances, indices, stats)
        stats: {"nprobe_used": int, "candidates_scored": int, "candidates_before_filter": int}
        """
        if query.ndim == 1:
            query = query.reshape(1, -1)
        
        current_nprobe = nprobe
        stats = {"nprobe_used": 0, "candidates_scored": 0, "candidates_before_filter": 0}
        
        while current_nprobe <= max_nprobe:
            # Fetch many candidates
            distances, indices = self.index.search(query, k=k * 10, nprobe=current_nprobe)
            
            dists_flat = distances.flatten()
            ids_flat = indices.flatten()
            
            stats["candidates_before_filter"] = len(ids_flat)
            
            # Apply filter
            if eligible_ids is not None:
                mask = np.array([int(i) in eligible_ids for i in ids_flat], dtype=bool)
                dists_flat = dists_flat[mask]
                ids_flat = ids_flat[mask]
            
            stats["nprobe_used"] = current_nprobe
            stats["candidates_scored"] = len(ids_flat)
            
            if len(ids_flat) >= k:
                top_k_idx = np.argsort(-dists_flat)[:k]
                final_dists = dists_flat[top_k_idx].reshape(1, -1)
                final_ids = ids_flat[top_k_idx].reshape(1, -1)
                return final_dists, final_ids, stats
            
            # Adaptive doubling
            if current_nprobe >= max_nprobe:
                break
            current_nprobe = min(current_nprobe * 2, max_nprobe)
        
        # Return what we have
        if len(ids_flat) > 0:
            top_k_idx = np.argsort(-dists_flat)[:k]
            final_dists = dists_flat[top_k_idx].reshape(1, -1)
            final_ids = ids_flat[top_k_idx].reshape(1, -1)
        else:
            final_dists = np.zeros((1, k), dtype=np.float32)
            final_ids = np.full((1, k), -1, dtype=np.int64)
        
        return final_dists, final_ids, stats

