from typing import List, Dict, Optional
from .base import ChunkerBase
import logging

logger = logging.getLogger(__name__)

class TokenChunker(ChunkerBase):
    """Chunks text based on token count (approximate, using whitespace split)."""
    def __init__(self, model_name: Optional[str] = None):
        super().__init__(model_name)

    def create_chunks(self, text: str, tokens_per_chunk: int = 100, overlap: int = 20) -> List[Dict]:
        tokens = text.split()
        chunks = []
        for i in range(0, len(tokens), tokens_per_chunk - overlap):
            chunk_tokens = tokens[i : i + tokens_per_chunk]
            if len(chunk_tokens) < 10:
                continue
            chunk_text = " ".join(chunk_tokens)
            chunk_embedding = self.get_embedding(chunk_text) if self.model else None
            chunk = {
                "text": chunk_text,
                "tokens": chunk_tokens,
                "embedding": chunk_embedding,
                "start_token_idx": i,
                "end_token_idx": min(i + tokens_per_chunk, len(tokens)),
                "chunk_id": len(chunks),
            }
            chunks.append(chunk)
        logger.info(f"Created {len(chunks)} token-based chunks from {len(tokens)} tokens")
        return chunks 