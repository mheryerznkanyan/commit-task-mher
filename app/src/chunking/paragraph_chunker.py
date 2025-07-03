from typing import List, Dict, Optional
from .base import ChunkerBase
import logging
import nltk

logger = logging.getLogger(__name__)

class ParagraphChunker(ChunkerBase):
    """
    Chunks text by paragraphs, optimized for arXiv papers.
    Handles common arXiv formatting quirks (e.g., excessive newlines, section headers).
    """
    def __init__(self, model_name: Optional[str] = None):
        super().__init__(model_name)
        # Ensure the punkt tokenizer is available
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')

    def _is_section_header(self, line: str) -> bool:
        # Simple heuristic for arXiv section headers (e.g., "1 Introduction", "2.1 Related Work")
        import re
        return bool(re.match(r"^\s*\d+(\.\d+)*\s+[A-Z][\w\s\-]+$", line.strip()))

    def create_chunks(self, text: str) -> List[Dict]:
        """
        Split arXiv paper text into paragraphs, handling common arXiv formatting.
        """
        from nltk.tokenize import sent_tokenize

        # Normalize line endings and remove excessive blank lines
        lines = [line.rstrip() for line in text.splitlines()]
        paragraphs = []
        current_paragraph = []

        for line in lines:
            if self._is_section_header(line):
                # Treat section headers as their own paragraph
                if current_paragraph:
                    paragraphs.append(" ".join(current_paragraph).strip())
                    current_paragraph = []
                paragraphs.append(line.strip())
            elif line.strip() == "":
                if current_paragraph:
                    paragraphs.append(" ".join(current_paragraph).strip())
                    current_paragraph = []
            else:
                current_paragraph.append(line.strip())
        if current_paragraph:
            paragraphs.append(" ".join(current_paragraph).strip())

        # Remove empty paragraphs and very short ones (common in arXiv metadata)
        paragraphs = [p for p in paragraphs if p and len(p.split()) > 5]

        # Optionally, merge very short paragraphs with the next one (arXiv abstracts, etc.)
        merged_paragraphs = []
        buffer = ""
        for para in paragraphs:
            if len(para.split()) < 20:
                buffer = buffer + " " + para if buffer else para
            else:
                if buffer:
                    merged_paragraphs.append(buffer.strip())
                    buffer = ""
                merged_paragraphs.append(para)
        if buffer:
            merged_paragraphs.append(buffer.strip())

        # Create chunks
        chunks = []
        for i, para in enumerate(merged_paragraphs):
            chunk_embedding = self.get_embedding(para) if self.model else None
            chunk = {
                "text": para,
                "embedding": chunk_embedding,
                "paragraph_idx": i,
                "chunk_id": i,
            }
            chunks.append(chunk)
        logger.info(f"Created {len(chunks)} paragraph-based chunks from {len(merged_paragraphs)} paragraphs (arXiv mode)")
        return chunks