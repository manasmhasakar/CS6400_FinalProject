import os
import tempfile

import numpy as np
import pandas as pd
from pyroaring import BitMap

from hybrid_search.filters.bitmap_index import build_and_save_bitmaps
from hybrid_search.filters.bitslice_index import build_bsi_indexes
from hybrid_search.filters.predicate import PredicateEvaluator


def test_predicate_evaluator():
    """Test predicate evaluation with bitmaps and BSI."""
    # Create temp directory
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create sample data
        data = {
            "id": [0, 1, 2, 3, 4],
            "main_category": ["Electronics", "Electronics", "Books", "Books", "Electronics"],
            "price": [10.0, 25.0, 50.0, 75.0, 100.0],
            "average_rating": [3.0, 4.0, 4.5, 4.8, 5.0],
        }
        df = pd.DataFrame(data)
        
        meta_path = os.path.join(tmpdir, "meta.parquet")
        df.to_parquet(meta_path)
        
        # Build bitmaps
        filters_cfg = {"buckets": {}}
        bitmap_dir = os.path.join(tmpdir, "bitmaps")
        build_and_save_bitmaps(meta_path, filters_cfg, bitmap_dir)
        
        # Build BSI
        bsi_cfg = {
            "price": {"scale": 10, "bits": 12},
            "average_rating": {"scale": 10, "bits": 10},
        }
        bsi_dir = os.path.join(tmpdir, "bsi")
        build_bsi_indexes(meta_path, bsi_cfg, bsi_dir)
        
        # Test predicates
        evaluator = PredicateEvaluator(bitmap_dir, bsi_dir)
        
        # Test equality
        pred_eq = {"op": "eq", "field": "main_category", "value": "Electronics"}
        result = evaluator.evaluate(pred_eq)
        assert result == BitMap([0, 1, 4]), f"Expected [0,1,4], got {list(result)}"
        
        # Test >= on price
        pred_ge = {"op": "ge", "field": "price", "value": 50.0}
        result = evaluator.evaluate(pred_ge)
        assert result == BitMap([2, 3, 4]), f"Expected [2,3,4], got {list(result)}"
        
        # Test AND
        pred_and = {
            "op": "and",
            "children": [
                {"op": "eq", "field": "main_category", "value": "Electronics"},
                {"op": "ge", "field": "price", "value": 50.0},
            ],
        }
        result = evaluator.evaluate(pred_and)
        assert result == BitMap([4]), f"Expected [4], got {list(result)}"
        
        # Test range
        pred_range = {"op": "range", "field": "price", "lo": 20.0, "hi": 80.0}
        result = evaluator.evaluate(pred_range)
        assert result == BitMap([1, 2, 3]), f"Expected [1,2,3], got {list(result)}"
        
        print("✓ All predicate tests passed")


if __name__ == "__main__":
    test_predicate_evaluator()

