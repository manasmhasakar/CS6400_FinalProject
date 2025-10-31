#!/bin/bash
set -e

echo "=== Running Tests ==="
echo ""

echo "[1/2] Testing BSI..."
python tests/test_bsi.py
echo ""

echo "[2/2] Testing Predicates..."
python tests/test_predicates.py
echo ""

echo "✓ All tests passed!"

