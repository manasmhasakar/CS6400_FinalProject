import numpy as np
from pyroaring import BitMap

from hybrid_search.filters.bitslice_index import BitSlicedIndex


def test_bsi_ge():
    """Test BSI >= operation."""
    values = np.array([10.5, 20.3, 30.1, 40.0, 50.0])
    ids = np.array([0, 1, 2, 3, 4])
    
    bsi = BitSlicedIndex(values, ids, scale=10, max_bits=10)
    
    result = bsi.ge(25.0)
    expected = BitMap([2, 3, 4])
    
    assert result == expected, f"Expected {expected}, got {result}"


def test_bsi_le():
    """Test BSI <= operation."""
    values = np.array([10.5, 20.3, 30.1, 40.0, 50.0])
    ids = np.array([0, 1, 2, 3, 4])
    
    bsi = BitSlicedIndex(values, ids, scale=10, max_bits=10)
    
    result = bsi.le(35.0)
    expected = BitMap([0, 1, 2])
    
    assert result == expected, f"Expected {expected}, got {result}"


def test_bsi_range():
    """Test BSI range operation."""
    values = np.array([10.5, 20.3, 30.1, 40.0, 50.0])
    ids = np.array([0, 1, 2, 3, 4])
    
    bsi = BitSlicedIndex(values, ids, scale=10, max_bits=10)
    
    result = bsi.range(20.0, 40.0)
    expected = BitMap([1, 2, 3])
    
    assert result == expected, f"Expected {expected}, got {result}"


def test_bsi_missing():
    """Test BSI with missing values."""
    values = np.array([10.0, np.nan, 30.0, np.nan, 50.0])
    ids = np.array([0, 1, 2, 3, 4])
    
    bsi = BitSlicedIndex(values, ids, scale=10, max_bits=10)
    
    result = bsi.ge(25.0)
    expected = BitMap([2, 4])
    
    assert result == expected, f"Expected {expected}, got {result}"
    assert bsi.missing_ids == BitMap([1, 3])


if __name__ == "__main__":
    test_bsi_ge()
    test_bsi_le()
    test_bsi_range()
    test_bsi_missing()
    print("✓ All BSI tests passed")

