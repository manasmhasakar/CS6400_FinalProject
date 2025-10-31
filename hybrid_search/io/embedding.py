import os
from typing import Optional

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def embed_texts(
    texts: list[str],
    model_name: str,
    batch_size: int = 256,
    normalize: bool = True,
    device: Optional[str] = None,
) -> np.ndarray:
    """
    Encode texts using sentence-transformers model.
    Returns shape (N, dim) float32 array.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = SentenceTransformer(model_name, device=device)
    
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
    )
    
    return embeddings.astype(np.float32)


def generate_embeddings(
    meta_path: str,
    output_path: str,
    model_name: str,
    batch_size: int = 256,
    normalize: bool = True,
) -> str:
    """
    Load meta.parquet, embed the 'text' column, write vectors.npy aligned with IDs.
    Returns path to vectors.npy.
    """
    df = pd.read_parquet(meta_path)
    texts = df["text"].tolist()
    
    embeddings = embed_texts(
        texts=texts,
        model_name=model_name,
        batch_size=batch_size,
        normalize=normalize,
    )
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, embeddings)
    
    return output_path

