import re
from typing import List, Dict
from .base import ChunkerBase
import logging

logger = logging.getLogger(__name__)

class SemanticChunker(ChunkerBase):
    """Creates semantic chunks from sentences using sentence transformers."""
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        super().__init__(model_name)

    def create_chunks(self, text: str, chunk_size: int = 5, overlap: int = 2) -> List[Dict]:
        # Split text into sentences (simple split, can be improved)
        sentences = re.split(r'(?<=[.!?]) +', text)
        chunks = []
        for i in range(0, len(sentences), chunk_size - overlap):
            chunk_sentences = sentences[i : i + chunk_size]
            if len(chunk_sentences) < 2:
                continue
            chunk_text = " ".join(chunk_sentences)
            chunk_embedding = self.get_embedding(chunk_text)
            chunk = {
                "text": chunk_text,
                "sentences": chunk_sentences,
                "embedding": chunk_embedding,
                "start_sentence_idx": i,
                "end_sentence_idx": min(i + chunk_size, len(sentences)),
                "chunk_id": len(chunks),
            }
            chunks.append(chunk)
        logger.info(f"Created {len(chunks)} semantic chunks from {len(sentences)} sentences")
        return chunks 