#!/bin/bash
set -e

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  Hybrid Vector Search: Full-Scale Execution (300k items)      ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  Virtual environment not activated. Activating..."
    source .venv/bin/activate
fi

# Clean previous data
echo "🗑️  Cleaning previous data..."
rm -rf hybrid_search/data
echo "✓ Data directory cleaned"
echo ""

# Step 1: Preprocessing
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  STEP 1/5: Data Preprocessing                                  ║"
echo "║  - Downloading 3 categories from UCSD                          ║"
echo "║  - Generating 300k embeddings (384-d)                          ║"
echo "║  - Estimated time: 30-40 minutes                               ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
START_TIME=$(date +%s)

python -m hybrid_search.cli preprocess

PREP_TIME=$(($(date +%s) - START_TIME))
echo ""
echo "✓ Preprocessing completed in ${PREP_TIME}s"
echo ""

# Verify data
echo "📊 Data verification:"
python -c "
import pandas as pd
import numpy as np
df = pd.read_parquet('hybrid_search/data/processed/meta.parquet')
vecs = np.load('hybrid_search/data/processed/vectors.npy')
print(f'  Items: {len(df):,}')
print(f'  Vectors: {vecs.shape}')
print(f'  Categories: {df.main_category.value_counts().to_dict()}')
print(f'  Stores: {df.store.nunique():,} unique')
print(f'  Price: {df.price.notna().sum():,} with price ({df.price.notna().sum()*100/len(df):.1f}%)')
"
echo ""

# Step 2: Build Filters
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  STEP 2/5: Building Filters                                    ║"
echo "║  - Categorical bitmaps (categories, stores)                    ║"
echo "║  - Bucketed bitmaps (price, rating)                            ║"
echo "║  - BSI indexes for numeric ranges                              ║"
echo "║  - Estimated time: 2-3 minutes                                 ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
START_TIME=$(date +%s)

python -m hybrid_search.cli build-filters

FILTER_TIME=$(($(date +%s) - START_TIME))
echo ""
echo "✓ Filters built in ${FILTER_TIME}s"
echo ""

# Step 3: Build Index (manual to avoid segfault)
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  STEP 3/5: Building IVFFlat Index                              ║"
echo "║  - Training IVF with nlist=2000                                ║"
echo "║  - Adding 300k vectors                                         ║"
echo "║  - Computing centroid statistics                               ║"
echo "║  - Estimated time: 3-5 minutes                                 ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
START_TIME=$(date +%s)

python -c "
import os
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from hybrid_search.index.faiss_ivfflat import build_ivfflat_index
from hybrid_search.index.centroid_stats import build_centroid_stats, CentroidStats

# Load config
cfg = OmegaConf.load('configs/index_ivfflat.yaml')

# Build index
print('Training IVF index...')
index = build_ivfflat_index(
    vectors_path='hybrid_search/data/processed/vectors.npy',
    output_path='hybrid_search/data/indexes/ivfflat.faiss',
    nlist=int(cfg.nlist),
    metric=cfg.metric,
    seed=int(cfg.seed),
)
print(f'✓ Index trained and saved: {index.ntotal:,} vectors in {index.nlist} clusters')

# Build centroid stats
print('Building centroid statistics...')
vectors = np.load('hybrid_search/data/processed/vectors.npy')
ids_df = pd.read_parquet('hybrid_search/data/processed/ids.parquet')
doc_ids = ids_df['id'].values

stats = build_centroid_stats(index, vectors, doc_ids)
stats.save('hybrid_search/data/indexes/ivfflat_centroid_stats.pkl')
print(f'✓ Centroid stats saved')
"

INDEX_TIME=$(($(date +%s) - START_TIME))
echo ""
echo "✓ Index built in ${INDEX_TIME}s"
echo ""

# Step 4: Evaluation
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  STEP 4/5: Running Evaluation                                  ║"
echo "║  - Generating 300 queries (100 per selectivity bin)            ║"
echo "║  - Computing ground truth                                      ║"
echo "║  - Running 3 methods: hybrid, prefilter, postfilter            ║"
echo "║  - Estimated time: 10-15 minutes                               ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
START_TIME=$(date +%s)

python -m hybrid_search.cli evaluate --variant ivfflat

EVAL_TIME=$(($(date +%s) - START_TIME))
echo ""
echo "✓ Evaluation completed in ${EVAL_TIME}s"
echo ""

# Step 5: Summary
TOTAL_TIME=$((PREP_TIME + FILTER_TIME + INDEX_TIME + EVAL_TIME))
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  EXECUTION COMPLETE ✅                                         ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "⏱️  Timing Summary:"
echo "  Preprocessing:  ${PREP_TIME}s ($((PREP_TIME/60))m $((PREP_TIME%60))s)"
echo "  Filters:        ${FILTER_TIME}s"
echo "  Index:          ${INDEX_TIME}s"
echo "  Evaluation:     ${EVAL_TIME}s ($((EVAL_TIME/60))m $((EVAL_TIME%60))s)"
echo "  ─────────────────────────"
echo "  TOTAL:          ${TOTAL_TIME}s ($((TOTAL_TIME/60))m $((TOTAL_TIME%60))s)"
echo ""
echo "📁 Output Files:"
echo "  Results CSV:    hybrid_search/data/results/ivfflat_eval_results.csv"
echo "  Plots:          hybrid_search/data/results/plots/"
echo "  Index:          hybrid_search/data/indexes/ivfflat.faiss"
echo ""
echo "📊 View Results:"
echo "  python -c \"import pandas as pd; df=pd.read_csv('hybrid_search/data/results/ivfflat_eval_results.csv'); print(df.groupby('method').agg({'latency_ms':['mean','median'],'recall':'mean','candidates_scored':'mean'}))\""
echo ""

