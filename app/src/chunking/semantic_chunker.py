
"""
Semantic chunker for processing text into meaningful chunks.
"""

import logging
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from pdf_process import PDFProcessor

logger = logging.getLogger(__name__)


class ChunkerBase:
    """Base class for chunkers."""
    def process_sentences(self, sentences: List[str], **kwargs) -> List[Dict]:
        raise NotImplementedError


class SemanticChunker(ChunkerBase):
    """Semantic chunker that groups sentences based on semantic similarity."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.pdf_processor = PDFProcessor()
    
    def create_chunks(self, text: str, chunk_size: int = 5, overlap: int = 2) -> List[Dict]:
        logger.info(f"[SemanticChunker] create_chunks called. Text length: {len(text) if text else 0}")
        if not text:
            logger.warning("[SemanticChunker] Input text is empty.")
            return []
        sentences = self.pdf_processor.split_sentences(text)
        logger.info(f"[SemanticChunker] Number of sentences extracted: {len(sentences)}")
        if not sentences:
            logger.warning("[SemanticChunker] No sentences extracted from text.")
            return []
        chunks = self.process_sentences(sentences, chunk_size=chunk_size, overlap=overlap)
        logger.info(f"[SemanticChunker] Number of chunks created: {len(chunks)}")
        return chunks
    
    def process_sentences(
        self,
        sentences: List[str],
        chunk_size: int = 5,
        overlap: int = 2
    ) -> List[Dict]:
        logger.info(f"[SemanticChunker] process_sentences called. Sentences: {len(sentences)}, chunk_size: {chunk_size}, overlap: {overlap}")
        if not sentences:
            logger.warning("[SemanticChunker] No sentences provided to process_sentences.")
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
        logger.info(f"[SemanticChunker] Finished process_sentences. Chunks created: {len(chunks)}")
        return chunks