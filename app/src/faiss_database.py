"""
FAISS Vector Database for storing and querying semantic chunks in-memory.
"""

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
import logging
import os
import pickle

logger = logging.getLogger(__name__)


class FaissDatabase:
    """In-memory vector database using FAISS for semantic search."""

    def __init__(
        self, model_name: str = "all-MiniLM-L6-v2", index_path: Optional[str] = None
    ):
        """
        Initialize FAISS database.

        Args:
            model_name: Sentence transformer model name (e.g., 'all-MiniLM-L6-v2', 'allenai/scibert_scivocab_uncased')
            index_path: Optional path to save/load FAISS index
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.chunks: List[Dict] = []
        self.index_path = index_path
        if index_path and os.path.exists(index_path):
            self.load(index_path)
        logger.info(f"Initialized FAISS database (dim={self.embedding_dim}, model={model_name})")

    def set_model(self, model_name: str):
        """Set a new embedding model for the database."""
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.chunks = []
        logger.info(f"Switched FAISS model to {model_name} (dim={self.embedding_dim})")

    def add_chunks(self, chunks: List[Dict]):
        """
        Add chunks to the FAISS index.

        Args:
            chunks: List of chunk dictionaries
        """
        vectors = np.stack([np.array(chunk["embedding"]) for chunk in chunks]).astype(
            "float32"
        )
        self.index.add(vectors)
        self.chunks.extend(chunks)
        logger.info(f"Added {len(chunks)} chunks to FAISS index")

    def search(
        self, query: str, top_k: int = 5, filter_arxiv_id: Optional[str] = None
    ) -> List[Dict]:
        """
        Search for similar chunks. Optionally filter by arxiv_id.

        Args:
            query: Search query
            top_k: Number of results to return
            filter_arxiv_id: Optional filter by arxiv_id

        Returns:
            List of search results with metadata and similarity score
        """
        if len(self.chunks) == 0:
            logger.warning("FAISS index is empty. No results to return.")
            return []
        query_vec = self.model.encode(query).astype("float32")
        query_vec = np.expand_dims(query_vec, axis=0)
        scores, indices = self.index.search(query_vec, min(top_k * 3, len(self.chunks)))
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            chunk = self.chunks[idx]
            if filter_arxiv_id and chunk.get("arxiv_id") != filter_arxiv_id:
                continue
            result = {
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
            results.append(result)
            if len(results) >= top_k:
                break
        logger.info(f"FAISS search returned {len(results)} results for query: {query}")
        return results

    def save(self, path: str = "faiss_index"):
        """
        Save FAISS index and chunk metadata to disk. Defaults to "faiss_index" prefix.
        """
        faiss.write_index(self.index, path + ".index")
        with open(path + ".chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)
        logger.info(
            f"Saved FAISS index and chunks to {path}.index and {path}.chunks.pkl"
        )

    def load(self, path: str = "faiss_index"):
        """
        Load FAISS index and chunk metadata from disk. Defaults to "faiss_index" prefix.
        """
        self.index = faiss.read_index(path + ".index")
        with open(path + ".chunks.pkl", "rb") as f:
            self.chunks = pickle.load(f)
        logger.info(
            f"Loaded FAISS index and chunks from {path}.index and {path}.chunks.pkl"
        )

    def clear(self):
        """Clear the index and all stored chunks."""
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.chunks = []
        logger.info("Cleared FAISS index and chunks")

    def add_paper(
        self, arxiv_id: str, title: str, summary: str, link: str, chunks: List[Dict]
    ) -> bool:
        """
        Add a paper and its chunks to the FAISS database (for pipeline compatibility).
        Attaches paper metadata to each chunk and adds them to the index.
        """
        try:
            for chunk in chunks:
                chunk["arxiv_id"] = arxiv_id
                chunk["title"] = title
                chunk["summary"] = summary
                chunk["link"] = link
            self.add_chunks(chunks)
            return True
        except Exception as e:
            logger.error(f"Error adding paper {arxiv_id} to FAISS: {e}")
            return False

    def get_collection_info(self) -> dict:
        """
        Return collection info for compatibility with QdrantDatabase.
        """
        return {
            "name": "faiss_in_memory",
            "vectors_count": self.index.ntotal,
            "points_count": len(self.chunks),
            "status": "in_memory",
        }

    def delete_paper(self, arxiv_id: str) -> bool:
        """
        Remove all chunks for a specific paper and rebuild the FAISS index.
        """
        try:
            # Filter out chunks with the given arxiv_id
            self.chunks = [
                chunk for chunk in self.chunks if chunk.get("arxiv_id") != arxiv_id
            ]
            # Rebuild the index
            self.index = faiss.IndexFlatIP(self.embedding_dim)
            if self.chunks:
                vectors = np.stack(
                    [np.array(chunk["embedding"]) for chunk in self.chunks]
                ).astype("float32")
                self.index.add(vectors)
            return True
        except Exception as e:
            logger.error(f"Error deleting paper {arxiv_id} from FAISS: {e}")
            return False

    def clear_collection(self) -> bool:
        """
        Clear all data from the FAISS database (for compatibility).
        """
        try:
            self.clear()
            return True
        except Exception as e:
            logger.error(f"Error clearing FAISS collection: {e}")
            return False
