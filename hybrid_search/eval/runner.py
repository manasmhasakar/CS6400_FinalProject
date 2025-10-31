import time
from typing import Dict, List

import numpy as np
import pandas as pd
from pyroaring import BitMap
from tqdm import tqdm

from hybrid_search.eval.ground_truth import compute_ground_truth
from hybrid_search.eval.metrics import recall_at_k
from hybrid_search.filters.predicate import PredicateEvaluator


class EvalRunner:
    """
    Orchestrates evaluation across methods and queries.
    """
    
    def __init__(
        self,
        vectors: np.ndarray,
        meta_df: pd.DataFrame,
        predicate_evaluator: PredicateEvaluator,
        k: int = 10,
    ):
        self.vectors = vectors
        self.meta_df = meta_df
        self.predicate_evaluator = predicate_evaluator
        self.k = k
    
    def run_method(
        self,
        method_name: str,
        executor,
        queries: List[Dict],
    ) -> pd.DataFrame:
        """
        Run a method on all queries.
        
        queries: list of {
            "query_vec": np.ndarray,
            "predicate": dict,
            "bin": str,
            "ground_truth": np.ndarray,
        }
        
        Returns: DataFrame with per-query results.
        """
        results = []
        
        for q in tqdm(queries, desc=f"Running {method_name}"):
            query_vec = q["query_vec"]
            predicate = q["predicate"]
            bin_name = q["bin"]
            ground_truth = q["ground_truth"]
            
            # Evaluate predicate
            eligible_ids = self.predicate_evaluator.evaluate(predicate)
            
            # Run search
            start = time.perf_counter()
            
            if method_name == "prefilter_exact":
                dists, ids, stats = executor.search(query_vec, self.k, eligible_ids)
            else:
                # Hybrid or postfilter
                nprobe = q.get("nprobe", 32)
                dists, ids, stats = executor.search(query_vec, self.k, nprobe, eligible_ids)
            
            latency_ms = (time.perf_counter() - start) * 1000
            
            # Compute recall
            retrieved = ids.flatten()
            recall = recall_at_k(retrieved, ground_truth, self.k)
            
            results.append({
                "method": method_name,
                "bin": bin_name,
                "latency_ms": latency_ms,
                "recall": recall,
                "nprobe_used": stats.get("nprobe_used", 0),
                "candidates_scored": stats.get("candidates_scored", 0),
                "eligible_count": len(eligible_ids),
            })
        
        return pd.DataFrame(results)
    
    def prepare_queries(
        self,
        query_vecs: np.ndarray,
        predicates: List[Dict],
        bins: List[str],
    ) -> List[Dict]:
        """
        Prepare queries with ground truth.
        """
        queries = []
        
        for query_vec, predicate, bin_name in zip(query_vecs, predicates, bins):
            eligible_ids = self.predicate_evaluator.evaluate(predicate)
            ground_truth = compute_ground_truth(query_vec, self.vectors, eligible_ids, self.k)
            
            queries.append({
                "query_vec": query_vec,
                "predicate": predicate,
                "bin": bin_name,
                "ground_truth": ground_truth,
            })
        
        return queries

