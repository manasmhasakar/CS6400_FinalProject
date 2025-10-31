# Hybrid Vector Search: ANN + Structured Filters

Production-grade hybrid search over 300k Amazon product items combining:
- **IVFFlat & IVFPQ** FAISS indexes with optional OPQ
- **Categorical bitmaps** (roaring) + **numeric bit-sliced indexes (BSI)** for efficient range queries
- **Filter-aware centroid probing** with adaptive nprobe
- **Baselines**: pre-filter exact and post-filter ANN

Dataset: [Amazon Reviews 2023 (McAuley Lab)](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023)

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Workflow

### 1. Preprocess: Load metadata + Generate embeddings
```bash
python -m hybrid_search.cli preprocess
```
- Downloads/parses Amazon metadata (Electronics, Home & Kitchen, All_Beauty)
- Generates 384-d sentence-transformers embeddings (MiniLM)
- Outputs: `hybrid_search/data/processed/{meta.parquet, ids.parquet, vectors.npy}`

### 2. Build Filters: Bitmaps + BSI
```bash
python -m hybrid_search.cli build-filters
```
- Categorical bitmaps for `main_category`, `store`
- Bucketed bitmaps for `price`, `average_rating`, `rating_number`
- Bit-sliced indexes (BSI) for numeric range queries on the same fields
- Outputs: `hybrid_search/data/filters/{bitmaps/*, bitslice/*}`

### 3. Build Index: IVFFlat or IVFPQ
```bash
# IVFFlat (faster, more memory)
python -m hybrid_search.cli build-index --variant ivfflat

# IVFPQ (compressed, slower queries)
python -m hybrid_search.cli build-index --variant ivfpq
```
- Trains and persists FAISS index
- Computes centroid statistics for filter-aware search
- Outputs: `hybrid_search/data/indexes/{ivfflat.faiss, ivfflat_centroid_stats.pkl}`

### 4. Evaluate: Hybrid vs Baselines
```bash
python -m hybrid_search.cli evaluate --variant ivfflat
```
- Generates ~1000 queries across selectivity bins (low 10-30%, med 1-10%, high 0.1-1%)
- Runs three methods:
  1. **Hybrid**: filter-aware IVF with adaptive nprobe
  2. **Prefilter exact**: apply filter → brute-force top-k
  3. **Postfilter ANN**: full ANN → filter → top-k
- Outputs:
  - `hybrid_search/data/results/{variant}_eval_results.csv`
  - `hybrid_search/data/results/plots/{latency_vs_recall.png, selectivity_vs_latency.png, candidates_scored.png}`

---

## Configuration

All configs in `configs/`:
- `dataset.yaml`: categories, max items
- `model.yaml`: sentence-transformer model, batch size
- `index_ivfflat.yaml`: nlist, nprobe, metric, seed
- `index_ivfpq.yaml`: nlist, m, nbits, use_opq, seed
- `filters.yaml`: bucket edges, BSI scale/bits
- `eval.yaml`: k, queries per bin, selectivity targets

---

## Key Design Choices

### Numeric Filtering Strategy
We implement **bit-sliced indexes (BSI)** for `price`, `average_rating`, `rating_number`:
- Stores each numeric value as a sequence of bit-level bitmaps
- Enables O(num_bits) range queries: `price >= 25`, `rating <= 4.5`, `price ∈ [25, 100]`
- Compare against coarse bucketed bitmaps in ablations

### IVFFlat vs IVFPQ
- **IVFFlat**: exact distances within clusters; higher memory (~460 MB for 300k×384 float32)
- **IVFPQ**: product quantization (m=32, nbits=8) compresses vectors 4-8×; slight recall drop but scalable

### Adaptive Nprobe
If `<k` eligible results found, automatically double `nprobe` up to max (256) and retry.

---

## Evaluation Metrics

- **Latency (ms)**: per-query wall time (p50/p90/p95)
- **Recall@k**: `|retrieved ∩ ground_truth| / k` vs exact filtered top-k
- **Candidates scored**: avg #vectors distance-computed per query
- **Index build time & memory**: training + add time; resident set size

---

## Expected Results

| Method            | p95 Latency (ms) | Recall@10 | Candidates Scored |
|-------------------|------------------|-----------|-------------------|
| Hybrid (IVFFlat)  | ~5-15            | >0.95     | ~2000-5000        |
| Postfilter ANN    | ~10-25           | >0.95     | ~5000-15000       |
| Prefilter Exact   | ~50-200          | 1.00      | ~10000-100000     |

*(Actual values depend on selectivity and nprobe settings)*

---

## Project Structure

```
hybrid_search/
├── cli.py                      # Typer CLI entry point
├── io/
│   ├── dataset_loader.py       # Amazon metadata download/parse
│   ├── embedding.py            # Sentence-transformers encoding
│   └── serialization.py        # Parquet/npy helpers
├── filters/
│   ├── bitmap_index.py         # Roaring bitmaps (categorical + bucketed)
│   ├── bitslice_index.py       # Bit-sliced indexes for numeric ranges
│   └── predicate.py            # Predicate evaluator (AND/OR/NOT)
├── index/
│   ├── faiss_ivfflat.py        # IVFFlat wrapper
│   ├── faiss_ivfpq.py          # IVFPQ + OPQ wrapper
│   └── centroid_stats.py       # Per-centroid eligible counts
├── executors/
│   ├── hybrid_executor.py      # Filter-aware IVF with adaptive nprobe
│   ├── baseline_prefilter_exact.py
│   └── baseline_postfilter_ann.py
└── eval/
    ├── query_generator.py      # Generate queries per selectivity bin
    ├── ground_truth.py         # Exact top-k on filtered subset
    ├── runner.py               # Orchestrate eval across methods
    ├── metrics.py              # Recall@k, percentiles
    └── plots.py                # Matplotlib/seaborn plots
```

---

## References

- **Dataset**: [Amazon Reviews 2023 (McAuley Lab)](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023)
- **FAISS**: [Facebook AI Similarity Search](https://github.com/facebookresearch/faiss)
- **Roaring Bitmaps**: [pyroaring](https://github.com/Ezibenroc/PyRoaringBitMap)
- **Sentence Transformers**: [sbert.net](https://www.sbert.net/)

---

## Notes

- **Scale**: Tested at 300k items (Electronics, Home & Kitchen, All_Beauty). Easily scales to 1M+ with IVFPQ.
- **BSI vs Bucketed**: Ablations show BSI provides finer-grained filtering with similar latency; trade-off is memory (one bitmap per bit).
- **Single-threaded**: Default evaluations run single-thread for fair comparison. Multi-thread can be enabled via `OMP_NUM_THREADS`.
