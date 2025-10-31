from typing import Tuple

import numpy as np
from pyroaring import BitMap


class PrefilterExactBaseline:
    """
    Baseline: apply filter first, then exact brute-force search.
    """
    
    def __init__(self, vectors: np.ndarray):
        """
        vectors: (N, dim) array of all vectors
        """
        self.vectors = vectors
    
    def search(self, query: np.ndarray, k: int, eligible_ids: BitMap) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Apply filter, gather vectors, compute exact top-k.
        
        Returns: (distances, indices, stats)
        stats: {"candidates_scored": int}
        """
        if query.ndim == 1:
            query = query.reshape(1, -1)
        
        # Gather eligible vectors
        eligible_list = list(eligible_ids)
        if len(eligible_list) == 0:
            # No eligible items
            dists = np.zeros((1, k), dtype=np.float32)
            ids = np.full((1, k), -1, dtype=np.int64)
            return dists, ids, {"candidates_scored": 0}
        
        eligible_vecs = self.vectors[eligible_list]
        
        # Compute inner product (cosine similarity on normalized vectors)
        scores = np.dot(eligible_vecs, query.T).flatten()  # (n_eligible,)
        
        # Top-k
        if len(scores) < k:
            top_k_idx = np.argsort(-scores)
        else:
            top_k_idx = np.argpartition(-scores, k)[:k]
            top_k_idx = top_k_idx[np.argsort(-scores[top_k_idx])]
        
        top_dists = scores[top_k_idx]
        top_ids = np.array(eligible_list)[top_k_idx]
        
        # Pad if necessary
        if len(top_ids) < k:
            pad = k - len(top_ids)
            top_dists = np.pad(top_dists, (0, pad), constant_values=0.0)
            top_ids = np.pad(top_ids, (0, pad), constant_values=-1)
        
        return top_dists.reshape(1, -1), top_ids.reshape(1, -1), {"candidates_scored": len(eligible_list)}

