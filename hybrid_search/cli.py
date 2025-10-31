import os
import typer
from omegaconf import OmegaConf

from hybrid_search.io.dataset_loader import preprocess_dataset
from hybrid_search.io.embedding import generate_embeddings

app = typer.Typer(help="Hybrid vector search: preprocess, index, evaluate, plot")


@app.command()
def preprocess(
    dataset_config: str = typer.Option("configs/dataset.yaml", help="Dataset config"),
    model_config: str = typer.Option("configs/model.yaml", help="Model config"),
    skip_embed: bool = typer.Option(False, help="Skip embedding generation"),
) -> None:
    """Load dataset, parse metadata, and optionally generate embeddings."""
    # Load dataset
    cfg_data = OmegaConf.load(dataset_config)
    data_root = cfg_data.get("data_root", "hybrid_search/data")
    categories = list(cfg_data.get("categories", []))
    max_items = int(cfg_data.get("max_items", 300000))
    
    paths = preprocess_dataset(data_root=data_root, categories=categories, max_items=max_items)
    typer.echo(f"✓ Wrote meta to: {paths['meta']}")
    typer.echo(f"✓ Wrote ids to: {paths['ids']}")
    
    # Generate embeddings
    if not skip_embed:
        cfg_model = OmegaConf.load(model_config)
        model_name = cfg_model.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
        batch_size = int(cfg_model.get("batch_size", 256))
        normalize = bool(cfg_model.get("normalize", True))
        
        vectors_path = os.path.join(data_root, "processed", "vectors.npy")
        vec_out = generate_embeddings(
            meta_path=paths['meta'],
            output_path=vectors_path,
            model_name=model_name,
            batch_size=batch_size,
            normalize=normalize,
        )
        typer.echo(f"✓ Wrote vectors to: {vec_out}")


@app.command("build-filters")
def build_filters(
    dataset_config: str = typer.Option("configs/dataset.yaml", help="Dataset config"),
    filters_config: str = typer.Option("configs/filters.yaml", help="Filters config"),
) -> None:
    """Build bitmap and BSI indexes for metadata filtering."""
    from hybrid_search.filters.bitmap_index import build_and_save_bitmaps
    from hybrid_search.filters.bitslice_index import build_bsi_indexes
    
    cfg_data = OmegaConf.load(dataset_config)
    cfg_filters = OmegaConf.load(filters_config)
    
    data_root = cfg_data.get("data_root", "hybrid_search/data")
    meta_path = os.path.join(data_root, "processed", "meta.parquet")
    
    bitmap_dir = os.path.join(data_root, "filters", "bitmaps")
    bsi_dir = os.path.join(data_root, "filters", "bitslice")
    
    # Build bitmaps
    bm_paths = build_and_save_bitmaps(meta_path, dict(cfg_filters), bitmap_dir)
    for name, path in bm_paths.items():
        typer.echo(f"✓ Bitmap: {name} -> {path}")
    
    # Build BSI
    bsi_config = cfg_filters.get("bsi", {})
    bsi_paths = build_bsi_indexes(meta_path, dict(bsi_config), bsi_dir)
    for name, path in bsi_paths.items():
        typer.echo(f"✓ BSI: {name} -> {path}")


@app.command("build-index")
def build_index(
    variant: str = typer.Option("ivfflat", help="ivfflat|ivfpq"),
    dataset_config: str = typer.Option("configs/dataset.yaml", help="Dataset config"),
    index_config: str = typer.Option(None, help="Index config (auto-detect from variant if None)"),
) -> None:
    """Build FAISS index (IVFFlat or IVFPQ) and centroid stats."""
    import numpy as np
    import pandas as pd
    from hybrid_search.index.faiss_ivfflat import build_ivfflat_index, IVFFlatIndex
    from hybrid_search.index.faiss_ivfpq import build_ivfpq_index, IVFPQIndex
    from hybrid_search.index.centroid_stats import build_centroid_stats
    
    # Load configs
    cfg_data = OmegaConf.load(dataset_config)
    data_root = cfg_data.get("data_root", "hybrid_search/data")
    
    if index_config is None:
        index_config = f"configs/index_{variant}.yaml"
    cfg_idx = OmegaConf.load(index_config)
    
    vectors_path = os.path.join(data_root, "processed", "vectors.npy")
    ids_path = os.path.join(data_root, "processed", "ids.parquet")
    indexes_dir = os.path.join(data_root, "indexes")
    os.makedirs(indexes_dir, exist_ok=True)
    
    index_path = os.path.join(indexes_dir, f"{variant}.faiss")
    stats_path = os.path.join(indexes_dir, f"{variant}_centroid_stats.pkl")
    
    # Build index
    typer.echo(f"Building {variant} index...")
    if variant == "ivfflat":
        index = build_ivfflat_index(
            vectors_path=vectors_path,
            output_path=index_path,
            nlist=int(cfg_idx.get("nlist", 4096)),
            metric=cfg_idx.get("metric", "ip"),
            seed=int(cfg_idx.get("seed", 42)),
        )
    elif variant == "ivfpq":
        index = build_ivfpq_index(
            vectors_path=vectors_path,
            output_path=index_path,
            nlist=int(cfg_idx.get("nlist", 8192)),
            m=int(cfg_idx.get("m", 32)),
            nbits=int(cfg_idx.get("nbits", 8)),
            use_opq=bool(cfg_idx.get("use_opq", False)),
            opq_m=cfg_idx.get("opq_m"),
            metric=cfg_idx.get("metric", "ip"),
            seed=int(cfg_idx.get("seed", 42)),
        )
    else:
        typer.echo(f"Unknown variant: {variant}", err=True)
        raise typer.Exit(1)
    
    typer.echo(f"✓ Index saved: {index_path}")
    
    # Build centroid stats
    typer.echo("Building centroid stats...")
    vectors = np.load(vectors_path)
    ids_df = pd.read_parquet(ids_path)
    doc_ids = ids_df["id"].values
    
    stats = build_centroid_stats(index, vectors, doc_ids)
    stats.save(stats_path)
    typer.echo(f"✓ Centroid stats saved: {stats_path}")


@app.command()
def evaluate(
    variant: str = typer.Option("ivfflat", help="ivfflat|ivfpq"),
    dataset_config: str = typer.Option("configs/dataset.yaml", help="Dataset config"),
    filters_config: str = typer.Option("configs/filters.yaml", help="Filters config"),
    eval_config: str = typer.Option("configs/eval.yaml", help="Eval config"),
    index_config: str = typer.Option(None, help="Index config (auto-detect from variant)"),
) -> None:
    """Run evaluation: hybrid, prefilter, postfilter baselines."""
    import numpy as np
    import pandas as pd
    from hybrid_search.index.faiss_ivfflat import IVFFlatIndex
    from hybrid_search.index.faiss_ivfpq import IVFPQIndex
    from hybrid_search.index.centroid_stats import CentroidStats
    from hybrid_search.executors.hybrid_executor import HybridExecutor
    from hybrid_search.executors.baseline_prefilter_exact import PrefilterExactBaseline
    from hybrid_search.executors.baseline_postfilter_ann import PostfilterANNBaseline
    from hybrid_search.filters.predicate import PredicateEvaluator
    from hybrid_search.eval.query_generator import generate_query_predicates, generate_query_vectors
    from hybrid_search.eval.runner import EvalRunner
    from hybrid_search.eval.plots import generate_all_plots
    
    # Load configs
    cfg_data = OmegaConf.load(dataset_config)
    cfg_filters = OmegaConf.load(filters_config)
    cfg_eval = OmegaConf.load(eval_config)
    
    if index_config is None:
        index_config = f"configs/index_{variant}.yaml"
    cfg_idx = OmegaConf.load(index_config)
    
    data_root = cfg_data.get("data_root", "hybrid_search/data")
    
    # Load data
    typer.echo("Loading data...")
    vectors = np.load(os.path.join(data_root, "processed", "vectors.npy"))
    meta_df = pd.read_parquet(os.path.join(data_root, "processed", "meta.parquet"))
    
    # Load index
    typer.echo(f"Loading {variant} index...")
    index_path = os.path.join(data_root, "indexes", f"{variant}.faiss")
    stats_path = os.path.join(data_root, "indexes", f"{variant}_centroid_stats.pkl")
    
    if variant == "ivfflat":
        index = IVFFlatIndex(dim=vectors.shape[1], nlist=1, metric="ip")
        index.load(index_path)
    elif variant == "ivfpq":
        index = IVFPQIndex(dim=vectors.shape[1], nlist=1, m=32, metric="ip")
        index.load(index_path)
    else:
        typer.echo(f"Unknown variant: {variant}", err=True)
        raise typer.Exit(1)
    
    centroid_stats = CentroidStats.load(stats_path)
    
    # Load filters
    typer.echo("Loading filters...")
    bitmap_dir = os.path.join(data_root, "filters", "bitmaps")
    bsi_dir = os.path.join(data_root, "filters", "bitslice")
    pred_eval = PredicateEvaluator(bitmap_dir, bsi_dir)
    
    # Generate queries
    typer.echo("Generating queries...")
    bins_config = dict(cfg_eval.get("bins", {}))
    queries_per_bin = int(cfg_eval.get("queries_per_bin", 333))
    k = int(cfg_eval.get("k", 10))
    
    query_specs = generate_query_predicates(meta_df, bins_config, queries_per_bin, seed=42)
    num_queries = len(query_specs)
    query_vecs = generate_query_vectors(vectors, num_queries, seed=42)
    
    bins = [spec[0] for spec in query_specs]
    predicates = [spec[1] for spec in query_specs]
    
    # Prepare queries with ground truth
    typer.echo("Computing ground truth...")
    runner = EvalRunner(vectors, meta_df, pred_eval, k=k)
    queries = runner.prepare_queries(query_vecs, predicates, bins)
    
    # Attach nprobe to queries
    nprobe = int(cfg_idx.get("nprobe", 32))
    for q in queries:
        q["nprobe"] = nprobe
    
    # Run evaluations
    typer.echo("Running evaluations...")
    
    # Hybrid
    hybrid_exec = HybridExecutor(index, centroid_stats)
    df_hybrid = runner.run_method("hybrid", hybrid_exec, queries)
    
    # Prefilter exact
    prefilter_exec = PrefilterExactBaseline(vectors)
    df_prefilter = runner.run_method("prefilter_exact", prefilter_exec, queries)
    
    # Postfilter ANN
    postfilter_exec = PostfilterANNBaseline(index)
    df_postfilter = runner.run_method("postfilter_ann", postfilter_exec, queries)
    
    # Combine results
    df_all = pd.concat([df_hybrid, df_prefilter, df_postfilter], ignore_index=True)
    
    # Save results
    results_path = os.path.join(data_root, "results", f"{variant}_eval_results.csv")
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    df_all.to_csv(results_path, index=False)
    typer.echo(f"✓ Results saved: {results_path}")
    
    # Generate plots
    plots_dir = os.path.join(data_root, "results", "plots")
    generate_all_plots(df_all, plots_dir)
    
    # Print summary
    typer.echo("\n=== Summary ===")
    summary = df_all.groupby("method").agg({
        "latency_ms": ["mean", "median"],
        "recall": "mean",
        "candidates_scored": "mean",
    }).round(3)
    typer.echo(summary.to_string())


@app.command()
def plot(out: str = typer.Option("plots/", help="Output dir for plots")) -> None:
    typer.echo(f"[stub] plotting to: {out}")


if __name__ == "__main__":
    app()
