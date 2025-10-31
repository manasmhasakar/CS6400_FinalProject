# Full-Scale Execution Guide

## Quick Start

```bash
cd /Users/manasmhasakar/Desktop/Project
source .venv/bin/activate
./scripts/run_full_scale.sh
```

**Total Time**: ~50-70 minutes  
**Disk Space**: ~5 GB

---

## Execution Plan Overview

### Phase 1: Preprocessing (~30-40 min)
**What happens:**
- Downloads 3 category files from UCSD (All Beauty, Electronics, Toys and Games)
- Parses ~300k item metadata
- Generates 384-d embeddings using sentence-transformers
- Normalizes vectors for cosine similarity

**Progress indicators:**
```
✓ Wrote meta to: hybrid_search/data/processed/meta.parquet
✓ Wrote ids to: hybrid_search/data/processed/ids.parquet
Batches: 100%|██████████| 1172/1172 [XX:XX<00:00, X.XXit/s]
✓ Wrote vectors to: hybrid_search/data/processed/vectors.npy
```

**Expected output:**
- ~300k items across 3 categories
- 300k × 384-d float32 vectors (~460 MB)
- Metadata with price, ratings, store info

---

### Phase 2: Filter Building (~2-3 min)
**What happens:**
- Builds categorical bitmap indexes (main_category, store)
- Creates bucketed bitmaps for price/rating ranges
- Constructs BSI (bit-sliced indexes) for numeric queries

**Progress indicators:**
```
✓ Bitmap: main_category -> ...
✓ Bitmap: store -> ...
✓ Bitmap: price_bucketed -> ...
✓ BSI: price -> ...
✓ BSI: average_rating -> ...
✓ BSI: rating_number -> ...
```

**Expected output:**
- 5 bitmap index files
- 3 BSI index files
- Total size: ~50-100 MB

---

### Phase 3: Index Building (~3-5 min)
**What happens:**
- Trains FAISS IVFFlat with nlist=2000 clusters
- Adds 300k vectors to index
- Computes per-centroid statistics for filter-aware search

**Progress indicators:**
```
Training IVF index...
✓ Index trained and saved: 300,000 vectors in 2000 clusters
Building centroid statistics...
✓ Centroid stats saved
```

**Expected output:**
- `ivfflat.faiss` (~460 MB)
- `ivfflat_centroid_stats.pkl` (~500 KB)

---

### Phase 4: Evaluation (~10-15 min)
**What happens:**
- Generates 300 queries (100 per selectivity bin)
- Computes exact ground truth for recall calculation
- Runs 3 search methods on all queries
- Generates plots and CSV results

**Progress indicators:**
```
Generating queries...
Computing ground truth...
Running hybrid: 100%|██████████| 100/100 [XX:XX<00:00]
Running prefilter_exact: 100%|██████████| 100/100 [XX:XX<00:00]
Running postfilter_ann: 100%|██████████| 100/100 [XX:XX<00:00]
✓ Results saved: ...
✓ Plots saved to: ...
```

**Expected output:**
- CSV with 900 rows (300 queries × 3 methods)
- 3 PNG plots (latency vs recall, selectivity curves, candidates)

---

## Monitoring Progress

### Check Current Step
```bash
# In another terminal
tail -f /tmp/hybrid_search.log  # if logging enabled
# or
ps aux | grep python  # check if running
```

### Estimate Time Remaining
- **Preprocessing**: Watch batch progress (e.g., "Batches: 25% [500/1172]")
  - ~1172 batches at 256 items/batch
  - ~2-3 batches/second on CPU, ~10/sec on GPU
- **Evaluation**: Watch query progress
  - ~300 queries, ~2-5 seconds per query

### Check Disk Usage
```bash
du -sh hybrid_search/data/
```

Expected sizes:
- After preprocessing: ~500 MB
- After filters: ~600 MB
- After indexing: ~1.1 GB
- After evaluation: ~1.1 GB

---

## Handling Issues

### Out of Memory
**Symptoms**: Process killed, "Killed: 9" message

**Solutions:**
1. Reduce `max_items` in `configs/dataset.yaml` (try 200k)
2. Close other applications
3. Use IVFPQ instead: `--variant ivfpq` (4-8× memory reduction)

### Slow Embedding
**Symptoms**: <1 batch/second

**Solutions:**
1. Check if GPU available: `python -c "import torch; print(torch.cuda.is_available())"`
2. Reduce `batch_size` in `configs/model.yaml` if GPU OOM
3. Expected: ~2-3 min/10k items on CPU, ~30s/10k on GPU

### Download Errors
**Symptoms**: 404 errors, connection timeouts

**Solutions:**
1. Check internet connection
2. Verify category names match dataset (use spaces not underscores)
3. Try again later (UCSD server may be busy)

### Low Recall in Results
**Not necessarily an error!** This can happen if:
- Filters are very restrictive (expected for "high" selectivity)
- Check `eligible_count` in CSV - if very low (<100), filters too tight
- Review query generation in `eval/query_generator.py`

---

## After Execution

### View Summary
```bash
python -c "
import pandas as pd
df = pd.read_csv('hybrid_search/data/results/ivfflat_eval_results.csv')
print(df.groupby('method').agg({
    'latency_ms': ['mean', 'median', lambda x: x.quantile(0.95)],
    'recall': 'mean',
    'candidates_scored': 'mean',
}).round(3))
"
```

### View Plots
```bash
open hybrid_search/data/results/plots/latency_vs_recall.png
open hybrid_search/data/results/plots/selectivity_vs_latency.png
open hybrid_search/data/results/plots/candidates_scored.png
```

### Generate Report
```bash
python -c "
import pandas as pd
df = pd.read_csv('hybrid_search/data/results/ivfflat_eval_results.csv')

print('=== Performance Summary ===')
print(df.groupby(['method', 'bin']).agg({
    'latency_ms': 'median',
    'recall': 'mean',
}).round(3))

print('\n=== Speedup Analysis ===')
pivot = df.groupby('method')['latency_ms'].median()
print(f'Hybrid vs Prefilter: {pivot[\"prefilter_exact\"] / pivot[\"hybrid\"]:.2f}x faster')
print(f'Hybrid vs Postfilter: {pivot[\"postfilter_ann\"] / pivot[\"hybrid\"]:.2f}x faster')
"
```

---

## Expected Results (300k scale)

### Successful Execution Indicators
✅ **Recall > 0.90** for all methods (except very restrictive filters)  
✅ **Hybrid latency** 5-15ms median  
✅ **Prefilter latency** 50-200ms median  
✅ **Postfilter latency** 10-30ms median  
✅ **Speedup** 2-10× hybrid vs baselines

### Interpretation
- **Hybrid wins** when filters are moderately restrictive (med/high selectivity)
- **Prefilter may win** when filters are very loose (low selectivity, many eligible items)
- **Recall trade-off** Hybrid ~95%+, Prefilter 100% (exact)

---

## Troubleshooting Checklist

Before running:
- [ ] Virtual environment activated
- [ ] ~5 GB disk space available
- [ ] Configs updated (300k, 3 categories, nlist=2000)
- [ ] Previous data cleaned (`rm -rf hybrid_search/data`)

During execution:
- [ ] Monitor progress (batch counts, query counts)
- [ ] Check memory usage (`top` or Activity Monitor)
- [ ] Watch for errors in terminal output

After execution:
- [ ] Verify CSV created with 900 rows
- [ ] Check plots generated (3 PNG files)
- [ ] Review recall values (>0.90 expected)
- [ ] Compare methods (hybrid should outperform on med/high)

---

## Next Steps After Successful Run

1. **Document Results** in final report
2. **Include Plots** in presentation
3. **Highlight Innovations**:
   - BSI for numeric filtering
   - IVFPQ memory efficiency
   - Filter-aware centroid probing
4. **Compare to Baselines** - show 2-10× speedup
5. **Discuss Trade-offs** - recall vs latency

---

## Emergency: Quick 100k Run

If time-constrained, run with 100k items (~15 min total):

```bash
# Edit configs/dataset.yaml: max_items: 100000
# Edit configs/index_ivfflat.yaml: nlist: 1000
# Edit configs/eval.yaml: queries_per_bin: 50

./scripts/run_full_scale.sh
```

This still satisfies ≥100k requirement and runs much faster!

