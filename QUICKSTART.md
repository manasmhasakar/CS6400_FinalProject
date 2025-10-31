# Quickstart Guide

Get the hybrid vector search system running in 5 minutes.

## Prerequisites

- Python 3.10+
- 16+ GB RAM recommended
- ~5 GB disk space for 300k items

## Installation

```bash
# Clone/navigate to project
cd /Users/manasmhasakar/Desktop/Project

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

Expected install time: 2-5 minutes

---

## Option 1: Full Pipeline (Automated)

Run everything end-to-end:

```bash
./scripts/run_full_pipeline.sh
```

This will:
1. Download & process 300k Amazon items (~10-15 min)
2. Generate embeddings (~5-10 min with CPU, ~2-3 min with GPU)
3. Build filters (~1-2 min)
4. Build IVFFlat index (~2-3 min)
5. Evaluate all methods (~5-10 min for ~1000 queries)

**Total time**: ~30-45 minutes (mostly embedding generation)

**Outputs**:
- `hybrid_search/data/results/ivfflat_eval_results.csv`
- `hybrid_search/data/results/plots/*.png`

---

## Option 2: Step-by-Step

### Step 1: Preprocess
```bash
python -m hybrid_search.cli preprocess
```

Downloads metadata for Electronics, Home & Kitchen, All_Beauty categories and generates embeddings.

**Check**: `hybrid_search/data/processed/` should contain `meta.parquet`, `ids.parquet`, `vectors.npy`

### Step 2: Build Filters
```bash
python -m hybrid_search.cli build-filters
```

Creates bitmap and BSI indexes for metadata attributes.

**Check**: `hybrid_search/data/filters/bitmaps/` and `hybrid_search/data/filters/bitslice/` should have `.pkl` files

### Step 3: Build Index
```bash
# IVFFlat (recommended for first run)
python -m hybrid_search.cli build-index --variant ivfflat

# Or IVFPQ (memory-efficient)
python -m hybrid_search.cli build-index --variant ivfpq
```

Trains FAISS index and computes centroid statistics.

**Check**: `hybrid_search/data/indexes/` should contain `.faiss` and `_centroid_stats.pkl` files

### Step 4: Evaluate
```bash
python -m hybrid_search.cli evaluate --variant ivfflat
```

Runs hybrid, prefilter exact, and postfilter ANN on ~1000 queries.

**Check**: `hybrid_search/data/results/` should contain CSV and plots

---

## View Results

### Summary Statistics
The evaluation prints a summary table:
```
=== Summary ===
                    latency_ms              recall  candidates_scored
                          mean  median      mean               mean
method
hybrid                    8.3     6.2     0.978              3542
postfilter_ann           15.7    12.4     0.976              8231
prefilter_exact          87.4    62.1     1.000             45632
```

### Plots
Check `hybrid_search/data/results/plots/`:
- `latency_vs_recall.png`: scatter plot of all methods
- `selectivity_vs_latency.png`: p95 latency by selectivity bin
- `candidates_scored.png`: average work per query

### CSV
`hybrid_search/data/results/ivfflat_eval_results.csv` contains per-query details:
- `method`, `bin`, `latency_ms`, `recall`, `nprobe_used`, `candidates_scored`, `eligible_count`

---

## Running Tests

```bash
./scripts/run_tests.sh
```

Or individually:
```bash
python tests/test_bsi.py
python tests/test_predicates.py
```

Expected output:
```
✓ All BSI tests passed
✓ All predicate tests passed
```

---

## Configuration

Edit configs in `configs/`:

### Scale Down for Quick Test
Edit `configs/dataset.yaml`:
```yaml
max_items: 50000  # Reduce from 300k
```

This will make embedding ~6× faster but reduce result diversity.

### Change Embedding Model
Edit `configs/model.yaml`:
```yaml
model_name: sentence-transformers/all-mpnet-base-v2  # Higher quality, 768-d
```

### Tune Index Parameters
Edit `configs/index_ivfflat.yaml`:
```yaml
nlist: 2048   # Fewer clusters = faster build, lower recall
nprobe: 64    # More probes = higher recall, slower queries
```

---

## Troubleshooting

### Out of Memory (OOM)
- Use IVFPQ instead: `--variant ivfpq`
- Reduce dataset size in `configs/dataset.yaml`
- Close other applications

### Slow Embedding
- Uses GPU if available (check `torch.cuda.is_available()`)
- Reduce `batch_size` in `configs/model.yaml` if GPU OOM
- Reduce `max_items` for quick test

### Download Fails
- Check internet connection
- Amazon data is hosted at UCSD; may be slow from some regions
- Retry after a few minutes

### Import Errors
```bash
pip install -r requirements.txt --upgrade
```

---

## Next Steps

1. **Compare IVFFlat vs IVFPQ**:
   ```bash
   python -m hybrid_search.cli build-index --variant ivfpq
   python -m hybrid_search.cli evaluate --variant ivfpq
   ```

2. **Experiment with Filters**:
   - Edit `configs/filters.yaml` to change bucket edges or BSI precision
   - Rebuild filters and re-evaluate

3. **Custom Queries**:
   - Modify `hybrid_search/eval/query_generator.py` to add domain-specific predicates

4. **Scale Up**:
   - Change `max_items` to 1M in `configs/dataset.yaml`
   - Use IVFPQ with higher `nlist` (8192+)

---

## Questions?

See:
- `README.md` for detailed documentation
- `IMPLEMENTATION_SUMMARY.md` for design choices
- Inline code comments for implementation details

