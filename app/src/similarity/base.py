"""
Base class for similarity search implementations.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class BaseSimilaritySearch(ABC):
    """Base class for similarity search implementations."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize similarity search.

        Args:
            model_name: Model name for the similarity search
        """
        self.model_name = model_name
        self.model = None
        logger.info(f"Initialized {self.__class__.__name__} with model: {model_name}")

    @abstractmethod
    def search_documents(
        self, 
        query: str, 
        chunks: List[Dict], 
        top_k: int = 5,
        filter_arxiv_id: Optional[str] = None
    ) -> List[Dict]:
        """
        Search for documents/chunks relevant to a query.

        Args:
            query: Search query string
            chunks: List of chunk dictionaries to search in
            top_k: Number of results to return
            filter_arxiv_id: Optional filter by arxiv_id

        Returns:
            List of search results with scores
        """
        pass

    def _filter_chunks(self, chunks: List[Dict], filter_arxiv_id: Optional[str] = None) -> List[Dict]:
        """Filter chunks by arxiv_id if specified."""
        if filter_arxiv_id:
            filtered_chunks = [chunk for chunk in chunks if chunk.get("arxiv_id") == filter_arxiv_id]
            logger.info(f"Filtered chunks: {len(chunks)} -> {len(filtered_chunks)}")
            return filtered_chunks
        return chunks

    def _create_result_dict(self, chunk: Dict, score: float) -> Dict:
        """Create a standardized result dictionary."""
        return {
            "score": float(score),
            "chunk_id": chunk.get("chunk_id"),
            "text": chunk.get("text"),
            "sentences": chunk.get("sentences"),
            "start_sentence_idx": chunk.get("start_sentence_idx"),
            "end_sentence_idx": chunk.get("end_sentence_idx"),
            "arxiv_id": chunk.get("arxiv_id"),
            "title": chunk.get("title"),
            "summary": chunk.get("summary"),
            "link": chunk.get("link"),
        } 