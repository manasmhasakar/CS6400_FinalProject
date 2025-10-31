import numpy as np
from pyroaring import BitMap


def compute_ground_truth(query_vec: np.ndarray, vectors: np.ndarray, eligible_ids: BitMap, k: int) -> np.ndarray:
    """
    Compute exact top-k on filtered subset.
    
    query_vec: (dim,)
    vectors: (N, dim)
    eligible_ids: bitmap of allowed IDs
    k: number of results
    
    Returns: array of top-k IDs (may be < k if not enough eligible)
    """
    eligible_list = list(eligible_ids)
    if len(eligible_list) == 0:
        return np.array([], dtype=np.int64)
    
    eligible_vecs = vectors[eligible_list]
    scores = np.dot(eligible_vecs, query_vec)
    
    if len(scores) < k:
        top_k_idx = np.argsort(-scores)
    else:
        top_k_idx = np.argpartition(-scores, k)[:k]
        top_k_idx = top_k_idx[np.argsort(-scores[top_k_idx])]
    
    top_ids = np.array(eligible_list)[top_k_idx]
    return top_ids

