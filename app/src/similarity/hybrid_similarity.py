"""
Hybrid similarity search: bi-encoder for filtering, cross-encoder for re-ranking.
"""

from typing import List, Dict, Optional
import logging
import numpy as np

from .base import BaseSimilaritySearch
from .bi_encoder_similarity import BiEncoderSimilarity
from .cross_encoder_similarity import CrossEncoderSimilarity

logger = logging.getLogger(__name__)

class HybridSimilarity(BaseSimilaritySearch):
    """Hybrid similarity: bi-encoder for filtering, cross-encoder for re-ranking."""

    def __init__(
        self,
        bi_encoder_model: str = "all-MiniLM-L6-v2",
        cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        candidate_multiplier: int = 3,
        batch_size: int = 32
    ):
        super().__init__(bi_encoder_model)
        self.bi_encoder = BiEncoderSimilarity(bi_encoder_model)
        self.cross_encoder = CrossEncoderSimilarity(cross_encoder_model)
        self.candidate_multiplier = candidate_multiplier
        self.batch_size = batch_size
        logger.info(f"Initialized HybridSimilarity with bi-encoder: {bi_encoder_model}, cross-encoder: {cross_encoder_model}")

    def _load_model(self):
        # Models are loaded in their respective classes
        pass

    def compute_similarity_matrix_batch(
        self, 
        texts1: List[str], 
        texts2: List[str]
    ) -> np.ndarray:
        """
        Compute similarity matrix between two sets of texts using the bi-encoder.
        
        Args:
            texts1: First set of texts
            texts2: Second set of texts
            
        Returns:
            Similarity matrix of shape (len(texts1), len(texts2))
        """
        # Use the bi-encoder for batch similarity computation
        return self.bi_encoder.compute_similarity_matrix_batch(texts1, texts2)

    def compute_similarity_matrix(self, texts: List[str]) -> np.ndarray:
        """
        Compute similarity matrix between all pairs of texts using the bi-encoder.
        
        Args:
            texts: List of texts
            
        Returns:
            Similarity matrix of shape (len(texts), len(texts))
        """
        # Use the bi-encoder for similarity matrix computation
        return self.bi_encoder.compute_similarity_matrix(texts)

    def find_topk_hybrid_similarity_batch(
        self,
        queries: list,
        candidate_chunks: list,
        top_k_v1: int = 15,
        threshold: float = 5.0,
        batch_size: int = 32
    ):
        """
        For each query in queries, find top_k_v1 candidates by bi-encoder, then re-rank with cross-encoder.
        Returns a list of lists of result dicts (one per query).
        """
        threshold = 5
        if not queries or not candidate_chunks:
            return [[] for _ in queries]
        # Bi-encoder: compute similarity matrix (len(queries) x len(candidate_chunks))
        chunk_texts = [chunk["text"] for chunk in candidate_chunks]
        sim_matrix = self.compute_similarity_matrix_batch(queries, chunk_texts)
        results_per_query = []
        for q_idx, query in enumerate(queries):
            similarities = sim_matrix[q_idx]
            # Top-k bi-encoder candidates
            top_indices = np.argsort(similarities)[::-1][:top_k_v1]
            candidates = [candidate_chunks[idx] for idx in top_indices]
            for i, idx in enumerate(top_indices):
                candidates[i]["bi_encoder_score"] = float(similarities[idx])
            # Cross-encoder: batch pairs
            pairs = [[query, c["text"]] for c in candidates]
            cross_scores = self.cross_encoder.model.predict(pairs)
            # Collect results
            cross_results = []
            for c, score in zip(candidates, cross_scores):
                result = c.copy()
                result["score"] = float(score)
                cross_results.append(result)
            # Sort and filter
            cross_results.sort(key=lambda x: x["score"], reverse=True)
            filtered = [r for r in cross_results if r["score"] >= threshold]
            results_per_query.append(filtered)
        return results_per_query

    def search_documents(
        self,
        query: str,
        chunks: List[Dict],
        top_k: int = 5,           # For cross-encoder
        threshold: float = 5,  # For cross-encoder
        top_k_v1: int = 15        # For bi-encoder candidate selection
    ) -> List[Dict]:
        """
        Hybrid search: bi-encoder for filtering, cross-encoder for re-ranking.
        """
        if not chunks:
            return []
        # If query is a list, use batch hybrid search
        if isinstance(query, list):
            return self.find_topk_hybrid_similarity_batch(
                queries=query,
                candidate_chunks=chunks,
                top_k_v1=top_k_v1,
                top_k=top_k,
                threshold=threshold,
                batch_size=self.batch_size
            )
        # Single query logic (as before)
        chunk_texts = [chunk.get("text", "") for chunk in chunks]
        similarity_matrix = self.compute_similarity_matrix_batch([query], chunk_texts)
        similarities = similarity_matrix[0]
        candidate_count = top_k_v1
        top_indices = np.argsort(similarities)[::-1][:candidate_count]
        candidate_chunks = [chunks[idx] for idx in top_indices]
        for i, idx in enumerate(top_indices):
            candidate_chunks[i]["bi_encoder_score"] = float(similarities[idx])
        if not candidate_chunks:
            return []
        cross_results = self.cross_encoder.search_documents(
            query,
            candidate_chunks,
            top_k=top_k,
            batch_size=self.batch_size,
        )
        filtered_results = [r for r in cross_results if r["score"] >= threshold]
        logger.info(f"Hybrid search returned {len(filtered_results)} results (threshold: {threshold})")
        return filtered_results 