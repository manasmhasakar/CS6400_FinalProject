import random
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from pyroaring import BitMap

from hybrid_search.filters.predicate import PredicateEvaluator


def generate_query_predicates(
    meta_df: pd.DataFrame,
    bins_config: Dict,
    queries_per_bin: int,
    seed: int = 42,
) -> List[Tuple[str, Dict, float]]:
    """
    Generate query predicates across selectivity bins.
    
    Returns: list of (bin_name, predicate_dict, target_selectivity)
    """
    random.seed(seed)
    np.random.seed(seed)
    
    queries = []
    
    # Available categories
    categories = meta_df["main_category"].dropna().unique().tolist()
    
    for bin_name, bin_cfg in bins_config.items():
        pass_rate = bin_cfg.get("pass_rate", [0.1, 0.3])
        lo, hi = pass_rate
        
        for _ in range(queries_per_bin):
            target_sel = random.uniform(lo, hi)
            
            # Simple heuristic: use category alone for low, category + rating for med/high
            if bin_name == "low":
                cat = random.choice(categories)
                pred = {"op": "eq", "field": "main_category", "value": cat}
            elif bin_name == "med":
                cat = random.choice(categories)
                rating_lo = random.uniform(3.5, 4.5)
                pred = {
                    "op": "and",
                    "children": [
                        {"op": "eq", "field": "main_category", "value": cat},
                        {"op": "ge", "field": "average_rating", "value": rating_lo},
                    ],
                }
            else:  # high
                cat = random.choice(categories)
                rating_lo = random.uniform(4.0, 4.8)
                price_lo = random.uniform(10, 50)
                price_hi = random.uniform(50, 200)
                pred = {
                    "op": "and",
                    "children": [
                        {"op": "eq", "field": "main_category", "value": cat},
                        {"op": "ge", "field": "average_rating", "value": rating_lo},
                        {"op": "range", "field": "price", "lo": price_lo, "hi": price_hi},
                    ],
                }
            
            queries.append((bin_name, pred, target_sel))
    
    return queries


def generate_query_vectors(
    vectors: np.ndarray,
    num_queries: int,
    seed: int = 42,
) -> np.ndarray:
    """
    Sample query vectors from the dataset (simulate user queries).
    
    Returns: (num_queries, dim) array
    """
    np.random.seed(seed)
    indices = np.random.choice(len(vectors), size=num_queries, replace=False)
    return vectors[indices]

