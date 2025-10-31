import os
from typing import Optional, Tuple

import faiss
import numpy as np


class IVFPQIndex:
    """
    FAISS IndexIVFPQ wrapper with optional OPQ preprocessing.
    """
    
    def __init__(
        self,
        dim: int,
        nlist: int,
        m: int,
        nbits: int = 8,
        metric: str = "ip",
        use_opq: bool = False,
        opq_m: Optional[int] = None,
        seed: int = 42,
    ):
        """
        dim: vector dimension
        nlist: number of coarse clusters
        m: number of PQ subquantizers (must divide dim)
        nbits: bits per subquantizer code
        metric: 'ip' or 'l2'
        use_opq: whether to use OPQ preprocessing
        opq_m: OPQ dimension (default=m if None)
        seed: random seed
        """
        self.dim = dim
        self.nlist = nlist
        self.m = m
        self.nbits = nbits
        self.metric = metric
        self.use_opq = use_opq
        self.opq_m = opq_m if opq_m is not None else m
        self.seed = seed
        
        # Build quantizer
        if metric == "ip":
            quantizer = faiss.IndexFlatIP(dim)
            metric_type = faiss.METRIC_INNER_PRODUCT
        elif metric == "l2":
            quantizer = faiss.IndexFlatL2(dim)
            metric_type = faiss.METRIC_L2
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        # Build PQ index
        self.index = faiss.IndexIVFPQ(quantizer, dim, nlist, m, nbits, metric_type)
        
        # Optional OPQ
        self.opq_matrix: Optional[faiss.LinearTransform] = None
        if use_opq:
            self.opq_matrix = faiss.OPQMatrix(dim, self.opq_m)
        
        self.is_trained = False
        self.ntotal = 0
    
    def train(self, vectors: np.ndarray) -> None:
        """Train the index (and OPQ if enabled)."""
        np.random.seed(self.seed)
        
        train_vecs = vectors
        
        # Train OPQ if enabled
        if self.opq_matrix is not None:
            self.opq_matrix.train(vectors)
            train_vecs = self.opq_matrix.apply_py(vectors)
        
        self.index.train(train_vecs)
        self.is_trained = True
    
    def add(self, vectors: np.ndarray) -> None:
        """Add vectors to the index (apply OPQ if enabled)."""
        if not self.is_trained:
            raise RuntimeError("Index must be trained before adding vectors")
        
        add_vecs = vectors
        if self.opq_matrix is not None:
            add_vecs = self.opq_matrix.apply_py(vectors)
        
        self.index.add(add_vecs)
        self.ntotal = self.index.ntotal
    
    def search(self, query: np.ndarray, k: int, nprobe: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for k nearest neighbors.
        query: (n_queries, dim) or (dim,)
        Returns: distances, indices
        """
        if query.ndim == 1:
            query = query.reshape(1, -1)
        
        search_query = query
        if self.opq_matrix is not None:
            search_query = self.opq_matrix.apply_py(query)
        
        self.index.nprobe = nprobe
        distances, indices = self.index.search(search_query, k)
        return distances, indices
    
    def save(self, path: str, opq_path: Optional[str] = None) -> None:
        """Save index (and OPQ matrix if used)."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        faiss.write_index(self.index, path)
        
        if self.opq_matrix is not None:
            if opq_path is None:
                opq_path = path.replace(".faiss", ".opq")
            faiss.write_VectorTransform(self.opq_matrix, opq_path)
    
    def load(self, path: str, opq_path: Optional[str] = None) -> None:
        """Load index (and OPQ matrix if present)."""
        self.index = faiss.read_index(path)
        self.is_trained = self.index.is_trained
        self.ntotal = self.index.ntotal
        
        if self.use_opq:
            if opq_path is None:
                opq_path = path.replace(".faiss", ".opq")
            if os.path.exists(opq_path):
                self.opq_matrix = faiss.read_VectorTransform(opq_path)
    
    def get_list_assignments(self, vectors: np.ndarray) -> np.ndarray:
        """Return coarse cluster assignment for each vector."""
        search_vecs = vectors
        if self.opq_matrix is not None:
            search_vecs = self.opq_matrix.apply_py(vectors)
        
        quantizer = faiss.downcast_index(self.index.quantizer)
        _, assignments = quantizer.search(search_vecs, 1)
        return assignments.flatten()


def build_ivfpq_index(
    vectors_path: str,
    output_path: str,
    nlist: int,
    m: int,
    nbits: int = 8,
    use_opq: bool = False,
    opq_m: Optional[int] = None,
    metric: str = "ip",
    seed: int = 42,
) -> IVFPQIndex:
    """Build and save an IVFPQ index."""
    vectors = np.load(vectors_path)
    dim = vectors.shape[1]
    
    index = IVFPQIndex(
        dim=dim,
        nlist=nlist,
        m=m,
        nbits=nbits,
        metric=metric,
        use_opq=use_opq,
        opq_m=opq_m,
        seed=seed,
    )
    
    index.train(vectors)
    index.add(vectors)
    
    opq_path = output_path.replace(".faiss", ".opq") if use_opq else None
    index.save(output_path, opq_path=opq_path)
    
    return index

