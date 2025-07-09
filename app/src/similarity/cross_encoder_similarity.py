"""
Cross-encoder similarity search implementation using sentence transformers.
"""

from sentence_transformers import CrossEncoder
from typing import List, Dict
import logging

from .base import BaseSimilaritySearch

logger = logging.getLogger(__name__)


class CrossEncoderSimilarity(BaseSimilaritySearch):
    """Cross-encoder similarity search using sentence transformers."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize cross-encoder similarity search.

        Args:
            model_name: Cross-encoder model name
        """
        super().__init__(model_name)
        self._load_model()

    def _load_model(self):
        """Load the cross-encoder model."""
        self.model = CrossEncoder(self.model_name)
        logger.info(f"Loaded cross-encoder model: {self.model_name}")

    def search_documents(
        self, 
        query: str, 
        chunks: List[Dict], 
        top_k: int = 5,
        batch_size: int = 32
    ) -> List[Dict]:
        """
        Search for documents using cross-encoder scoring.

        Args:
            query: Search query string
            chunks: List of chunk dictionaries to search in
            top_k: Number of results to return
            filter_arxiv_id: Optional filter by arxiv_id
            batch_size: Batch size for cross-encoder scoring

        Returns:
            List of search results with scores
        """
        if not chunks:
            return []


        # Extract chunk texts
        chunk_texts = [chunk.get("text", "") for chunk in chunks]

        # Create query-document pairs
        pairs = [[query, chunk_text] for chunk_text in chunk_texts]

        # Score pairs in batches
        scores = []
        for i in range(0, len(pairs), batch_size):
            batch_pairs = pairs[i:i + batch_size]
            batch_scores = self.model.predict(batch_pairs)
            scores.extend(batch_scores.tolist())

        # Create results with scores
        results = []
        for chunk, score in zip(chunks, scores):
            result = self._create_result_dict(chunk, score)
            results.append(result)

        # Sort by score and return top_k
        results.sort(key=lambda x: x["score"], reverse=True)
        results = results[:top_k]

        logger.info(f"Cross-encoder search returned {len(results)} results")
        return results

    def score_pairs(self, pairs: List[List[str]], batch_size: int = 32) -> List[float]:
        """Score a list of text pairs using cross-encoder."""
        scores = []
        for i in range(0, len(pairs), batch_size):
            batch_pairs = pairs[i:i + batch_size]
            batch_scores = self.model.predict(batch_pairs)
            scores.extend(batch_scores.tolist())
        return scores 