"""
Paragraph chunker for processing text into paragraph-based chunks.
"""

from typing import List, Dict
from sentence_transformers import SentenceTransformer

class ParagraphChunker:
    """Chunker that creates chunks based on paragraph boundaries."""
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    def process_sentences(
        self,
        sentences: List[str],
        chunk_size: int = 5,
        overlap: int = 2,
        similarity_threshold: float = 0.85,
        min_sentence_length: int = 10,
        sentence_split_regex: str = r'(?<=[.!?])\s+(?=[A-Z])',
        **kwargs
    ) -> List[Dict]:
        if not sentences:
            return []
        # Filter out very short sentences
        filtered_sentences = [
            s.strip() for s in sentences 
            if len(s.strip()) >= min_sentence_length
        ]
        if not filtered_sentences:
            return []
        # Group sentences into paragraphs (simple approach)
        paragraphs = ParagraphChunker._group_into_paragraphs(filtered_sentences)
        chunks = []
        for i, paragraph in enumerate(paragraphs):
            if paragraph:
                chunk_text = " ".join(paragraph)
                embedding = self.model.encode(chunk_text)
                chunks.append({
                    "text": chunk_text,
                    "sentences": paragraph,
                    "embedding": embedding,
                    "paragraph_id": i,
                    "chunk_type": "paragraph",
                    "chunk_id": len(chunks)  # Add unique chunk_id
                })
        return chunks
    @staticmethod
    def _group_into_paragraphs(sentences: List[str]) -> List[List[str]]:
        paragraphs = []
        current_paragraph = []
        for sentence in sentences:
            if ParagraphChunker._is_paragraph_start(sentence) and current_paragraph:
                paragraphs.append(current_paragraph)
                current_paragraph = []
            current_paragraph.append(sentence)
        if current_paragraph:
            paragraphs.append(current_paragraph)
        return paragraphs
    @staticmethod
    def _is_paragraph_start(sentence: str) -> bool:
        starters = [
            "first", "second", "third", "finally", "in conclusion",
            "however", "moreover", "furthermore", "additionally",
            "on the other hand", "conversely", "meanwhile",
            "the", "this", "these", "those", "we", "our", "the authors"
        ]
        sentence_lower = sentence.lower().strip()
        return any(sentence_lower.startswith(starter) for starter in starters)