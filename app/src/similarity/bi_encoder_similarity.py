"""
Bi-encoder similarity search implementation using sentence transformers.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
import logging
from sklearn.metrics.pairwise import cosine_similarity

from .base import BaseSimilaritySearch

logger = logging.getLogger(__name__)


class BiEncoderSimilarity(BaseSimilaritySearch):
    """Bi-encoder similarity search using sentence transformers and cosine similarity."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize bi-encoder similarity search.

        Args:
            model_name: Sentence transformer model name
        """
        super().__init__(model_name)
        self._load_model()

    def _load_model(self):
        """Load the sentence transformer model."""
        self.model = SentenceTransformer(self.model_name)
        logger.info(f"Loaded bi-encoder model: {self.model_name}")

    def search_documents(
        self, 
        query: str, 
        chunks: List[Dict], 
        top_k: int = 5,
        filter_arxiv_id: Optional[str] = None
    ) -> List[Dict]:
        """
        Search for documents using bi-encoder cosine similarity.

        Args:
            query: Search query string
            chunks: List of chunk dictionaries to search in
            top_k: Number of results to return
            filter_arxiv_id: Optional filter by arxiv_id

        Returns:
            List of search results with cosine similarity scores
        """
        if not chunks:
            return []

        # Filter chunks if needed
        chunks = self._filter_chunks(chunks, filter_arxiv_id)
        if not chunks:
            return []

        # Extract chunk texts
        chunk_texts = [chunk.get("text", "") for chunk in chunks]

        # Encode query and chunks
        query_embedding = self.model.encode([query])
        chunk_embeddings = self.model.encode(chunk_texts)

        # Calculate cosine similarities
        similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]

        # Create results with scores
        results = []
        for chunk, similarity in zip(chunks, similarities):
            result = self._create_result_dict(chunk, similarity)
            results.append(result)

        # Sort by similarity and return top_k
        results.sort(key=lambda x: x["score"], reverse=True)
        results = results[:top_k]

        logger.info(f"Bi-encoder search returned {len(results)} results")
        return results

    def compute_similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity between two texts."""
        embeddings = self.model.encode([text1, text2])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return float(similarity)

    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get embeddings for a list of texts."""
        return self.model.encode(texts)

    def compute_similarity_matrix(self, texts: List[str]) -> np.ndarray:
        """Compute similarity matrix between all pairs of texts."""
        embeddings = self.get_embeddings(texts)
        return cosine_similarity(embeddings)

    def compute_similarity_matrix_batch(
        self, 
        texts1: List[str], 
        texts2: List[str]
    ) -> np.ndarray:
        """
        Compute similarity matrix between two sets of texts efficiently.
        
        Args:
            texts1: First set of texts
            texts2: Second set of texts
            
        Returns:
            Similarity matrix of shape (len(texts1), len(texts2))
        """
        # Encode all texts at once
        all_texts = texts1 + texts2
        all_embeddings = self.get_embeddings(all_texts)
        
        embeddings1 = all_embeddings[:len(texts1)]
        embeddings2 = all_embeddings[len(texts1):]
        
        return cosine_similarity(embeddings1, embeddings2) 