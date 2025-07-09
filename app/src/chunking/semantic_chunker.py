"""
Semantic chunker for processing text into meaningful chunks.
"""

from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer


class ChunkerBase:
    """Base class for chunkers."""
    def process_sentences(self, sentences: List[str], **kwargs) -> List[Dict]:
        raise NotImplementedError


class SemanticChunker(ChunkerBase):
    """Semantic chunker that groups sentences based on semantic similarity."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        
    def process_sentences(
        self,
        sentences: List[str],
        chunk_size: int = 5,
        overlap: int = 2,
        similarity_threshold: float = 0.85
    ) -> List[Dict]:
        if not sentences:
            return []
        chunks = []
        for i in range(0, len(sentences), chunk_size - overlap):
            chunk_sentences = sentences[i:i + chunk_size]
            if chunk_sentences:
                chunk_text = " ".join(chunk_sentences)
                embedding = self.model.encode(chunk_text)
                chunks.append({
                    "text": chunk_text,
                    "sentences": chunk_sentences,
                    "embedding": embedding,
                    "start_idx": i,
                    "end_idx": min(i + chunk_size, len(sentences)),
                    "chunk_id": len(chunks)  # Add unique chunk_id
                })
        return chunks 