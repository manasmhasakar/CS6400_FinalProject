import os
from typing import Optional, Tuple

import faiss
import numpy as np


class IVFFlatIndex:
    """
    FAISS IndexIVFFlat wrapper with reproducible training.
    """
    
    def __init__(self, dim: int, nlist: int, metric: str = "ip", seed: int = 42):
        """
        dim: vector dimension
        nlist: number of coarse clusters
        metric: 'ip' (inner product) or 'l2'
        seed: random seed for kmeans
        """
        self.dim = dim
        self.nlist = nlist
        self.metric = metric
        self.seed = seed
        
        # Build quantizer and index
        if metric == "ip":
            quantizer = faiss.IndexFlatIP(dim)
            self.index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        elif metric == "l2":
            quantizer = faiss.IndexFlatL2(dim)
            self.index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_L2)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        self.is_trained = False
        self.ntotal = 0
    
    def train(self, vectors: np.ndarray) -> None:
        """Train the IVF index on vectors."""
        # Set seed for reproducible kmeans clustering
        clustering_params = faiss.ClusteringParameters()
        clustering_params.seed = self.seed
        clustering_params.niter = 25
        clustering_params.verbose = False
        
        # Access the quantizer's clustering params (if possible)
        # FAISS Python API doesn't directly expose per-train seed setting cleanly,
        # so we'll rely on global seed for now
        np.random.seed(self.seed)
        
        self.index.train(vectors)
        self.is_trained = True
    
    def add(self, vectors: np.ndarray) -> None:
        """Add vectors to the index."""
        if not self.is_trained:
            raise RuntimeError("Index must be trained before adding vectors")
        self.index.add(vectors)
        self.ntotal = self.index.ntotal
    
    def search(self, query: np.ndarray, k: int, nprobe: int = 1, ids_filter: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for k nearest neighbors.
        query: (n_queries, dim) or (dim,) for single query
        ids_filter: optional array of allowed IDs (not directly supported in base FAISS, returns all)
        Returns: distances (n_queries, k), indices (n_queries, k)
        """
        if query.ndim == 1:
            query = query.reshape(1, -1)
        
        self.index.nprobe = nprobe
        
        # Note: IDSelector support requires faiss >= 1.7.3 and specific APIs
        # For now, we'll do basic search; hybrid executor will handle filtering
        distances, indices = self.index.search(query, k)
        return distances, indices
    
    def save(self, path: str) -> None:
        """Save index to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        faiss.write_index(self.index, path)
    
    def load(self, path: str) -> None:
        """Load index from disk."""
        self.index = faiss.read_index(path)
        self.is_trained = self.index.is_trained
        self.ntotal = self.index.ntotal
    
    def get_list_assignments(self, vectors: np.ndarray) -> np.ndarray:
        """
        Return coarse cluster assignment for each vector.
        Returns: array of shape (n,) with cluster IDs.
        """
        quantizer = faiss.downcast_index(self.index.quantizer)
        _, assignments = quantizer.search(vectors, 1)
        return assignments.flatten()


def build_ivfflat_index(
    vectors_path: str,
    output_path: str,
    nlist: int,
    metric: str = "ip",
    seed: int = 42,
) -> IVFFlatIndex:
    """
    Build and save an IVFFlat index.
    """
    vectors = np.load(vectors_path)
    dim = vectors.shape[1]
    
    index = IVFFlatIndex(dim=dim, nlist=nlist, metric=metric, seed=seed)
    index.train(vectors)
    index.add(vectors)
    index.save(output_path)
    
    return index

