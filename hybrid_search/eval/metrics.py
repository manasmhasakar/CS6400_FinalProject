import numpy as np


def recall_at_k(retrieved: np.ndarray, ground_truth: np.ndarray, k: int) -> float:
    """
    Compute recall@k.
    
    retrieved: array of retrieved IDs
    ground_truth: array of ground truth IDs
    k: top-k
    
    Returns: recall in [0, 1]
    """
    if len(ground_truth) == 0:
        return 0.0
    
    retrieved_set = set(retrieved[:k])
    gt_set = set(ground_truth[:k])
    
    intersection = len(retrieved_set & gt_set)
    return intersection / min(k, len(gt_set))


def compute_percentiles(values: list, percentiles: list = [50, 90, 95]) -> dict:
    """
    Compute percentiles of a list of values.
    
    Returns: {p: value}
    """
    if not values:
        return {p: 0.0 for p in percentiles}
    return {p: np.percentile(values, p) for p in percentiles}

