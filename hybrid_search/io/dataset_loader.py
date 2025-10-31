import gzip
import io
import json
import os
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

from hybrid_search.io.serialization import ensure_dir, write_parquet
from hybrid_search.utils.text import build_text_field, parse_price

UCSD_META_BASE = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/"


def _category_to_filename(category: str) -> str:
    # Dataset uses underscores for spaces and some categories include underscores already
    # Expect exact names like "Electronics" -> meta_Electronics.jsonl.gz
    safe = category.replace(" ", "_")
    return f"meta_{safe}.jsonl.gz"


def iter_jsonl_gz_bytes(fp: io.BufferedReader) -> Iterable[Dict]:
    with gzip.GzipFile(fileobj=fp, mode="rb") as gz:
        for line in gz:
            if not line:
                continue
            yield json.loads(line)


def load_category_meta(path_or_url: str, show_progress: bool = True) -> pd.DataFrame:
    # path_or_url can be a local path or http(s) URL
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        import requests  # lazy import
        with requests.get(path_or_url, stream=True, timeout=60) as r:
            r.raise_for_status()
            raw = io.BytesIO(r.content)
            rows = list(iter_jsonl_gz_bytes(raw))
    else:
        with open(path_or_url, "rb") as f:
            rows = list(iter_jsonl_gz_bytes(f))
    df = pd.DataFrame(rows)
    return df


def prepare_metadata(df: pd.DataFrame) -> pd.DataFrame:
    # Keep and normalize key fields; some may be missing
    keep_cols = [
        "parent_asin",
        "main_category",
        "store",
        "average_rating",
        "rating_number",
        "price",
        "title",
        "description",
        "features",
    ]
    for c in keep_cols:
        if c not in df.columns:
            df[c] = None
    out = df[keep_cols].copy()

    # Normalize types
    out["average_rating"] = pd.to_numeric(out["average_rating"], errors="coerce")
    out["rating_number"] = pd.to_numeric(out["rating_number"], errors="coerce").fillna(0).astype("int64")

    # Price parsing to float dollars; NaN if missing
    out["price"] = out["price"].apply(parse_price)

    # Text field for embedding
    out["text"] = out.apply(
        lambda r: build_text_field(
            title=r.get("title"),
            description=r.get("description"),
            features=r.get("features"),
        ),
        axis=1,
    )

    # Drop rows without parent_asin or text
    out = out.dropna(subset=["parent_asin"]).reset_index(drop=True)
    out = out[out["text"].str.len() > 0].reset_index(drop=True)

    return out


def select_and_limit(df: pd.DataFrame, categories: List[str], max_items: int) -> pd.DataFrame:
    if categories:
        df = df[df["main_category"].isin(categories)]
    if max_items and len(df) > max_items:
        df = df.sample(n=max_items, random_state=42)
    return df.reset_index(drop=True)


def add_internal_ids(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.reset_index(drop=True)
    df["id"] = df.index.astype("int64")
    ids = df[["id", "parent_asin"]].copy()
    return df, ids


def preprocess_dataset(data_root: str, categories: List[str], max_items: int) -> Dict[str, str]:
    os.makedirs(data_root, exist_ok=True)
    raw_dir = os.path.join(data_root, "raw")
    processed_dir = os.path.join(data_root, "processed")
    ensure_dir(raw_dir)
    ensure_dir(processed_dir)

    frames: List[pd.DataFrame] = []
    for cat in categories:
        fname = _category_to_filename(cat)
        url = UCSD_META_BASE + fname
        df_cat = load_category_meta(url)
        frames.append(df_cat)

    df_all = pd.concat(frames, ignore_index=True)
    df_meta = prepare_metadata(df_all)
    df_meta = select_and_limit(df_meta, categories=categories, max_items=max_items)
    df_meta, ids = add_internal_ids(df_meta)

    meta_path = os.path.join(processed_dir, "meta.parquet")
    ids_path = os.path.join(processed_dir, "ids.parquet")

    write_parquet(df_meta, meta_path)
    write_parquet(ids, ids_path)

    return {"meta": meta_path, "ids": ids_path}
