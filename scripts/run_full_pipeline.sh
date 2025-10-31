#!/bin/bash
set -e

echo "=== Hybrid Vector Search: Full Pipeline ==="
echo ""

echo "[1/4] Preprocessing: loading metadata + generating embeddings..."
python -m hybrid_search.cli preprocess
echo ""

echo "[2/4] Building filters: bitmaps + BSI..."
python -m hybrid_search.cli build-filters
echo ""

echo "[3/4] Building IVFFlat index..."
python -m hybrid_search.cli build-index --variant ivfflat
echo ""

echo "[4/4] Evaluating: hybrid vs baselines..."
python -m hybrid_search.cli evaluate --variant ivfflat
echo ""

echo "✓ Pipeline complete! Check hybrid_search/data/results/ for outputs."

