"""
Semantic Chunker for creating meaningful text chunks with embeddings.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class SemanticChunker:
    """Creates semantic chunks from sentences using sentence transformers."""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the semantic chunker.
        
        Args:
            model_name: Sentence transformer model to use
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        logger.info(f"Initialized SemanticChunker with model: {model_name}")
    
    def create_chunks(self, sentences: List[str], 
                     chunk_size: int = 5, 
                     overlap: int = 2) -> List[Dict]:
        """
        Create semantic chunks from sentences.
        
        Args:
            sentences: List of sentences
            chunk_size: Number of sentences per chunk
            overlap: Number of overlapping sentences between chunks
            
        Returns:
            List of chunk dictionaries
        """
        chunks = []
        
        for i in range(0, len(sentences), chunk_size - overlap):
            chunk_sentences = sentences[i:i + chunk_size]
            
            if len(chunk_sentences) < 2:
                continue
                
            chunk_text = ' '.join(chunk_sentences)
            chunk_embedding = self.model.encode(chunk_text)
            
            chunk = {
                'text': chunk_text,
                'sentences': chunk_sentences,
                'embedding': chunk_embedding,
                'start_sentence_idx': i,
                'end_sentence_idx': min(i + chunk_size, len(sentences)),
                'chunk_id': len(chunks)
            }
            
            chunks.append(chunk)
        
        logger.info(f"Created {len(chunks)} chunks from {len(sentences)} sentences")
        return chunks
    
    def merge_similar_chunks(self, chunks: List[Dict], 
                           similarity_threshold: float = 0.85) -> List[Dict]:
        """
        Merge chunks that are semantically similar.
        
        Args:
            chunks: List of chunks
            similarity_threshold: Threshold for merging
            
        Returns:
            List of merged chunks
        """
        if len(chunks) <= 1:
            return chunks
        
        merged_chunks = []
        used_indices = set()
        
        # Compute similarity matrix
        embeddings = np.array([chunk['embedding'] for chunk in chunks])
        similarity_matrix = cosine_similarity(embeddings)
        
        for i in range(len(chunks)):
            if i in used_indices:
                continue
                
            similar_indices = np.where(similarity_matrix[i] > similarity_threshold)[0]
            similar_indices = [idx for idx in similar_indices if idx not in used_indices]
            
            if len(similar_indices) > 1:
                similar_chunks = [chunks[idx] for idx in similar_indices]
                used_indices.update(similar_indices)
                
                # Merge chunks
                merged_text = ' '.join(chunk['text'] for chunk in similar_chunks)
                merged_sentences = [sent for chunk in similar_chunks for sent in chunk['sentences']]
                merged_embedding = self.model.encode(merged_text)
                
                merged_chunk = {
                    'text': merged_text,
                    'sentences': merged_sentences,
                    'embedding': merged_embedding,
                    'start_sentence_idx': min(c['start_sentence_idx'] for c in similar_chunks),
                    'end_sentence_idx': max(c['end_sentence_idx'] for c in similar_chunks),
                    'chunk_id': len(merged_chunks),
                    'merged_from': [c['chunk_id'] for c in similar_chunks]
                }
                
                merged_chunks.append(merged_chunk)
                
            elif i not in used_indices:
                chunks[i]['chunk_id'] = len(merged_chunks)
                merged_chunks.append(chunks[i])
                used_indices.add(i)
        
        logger.info(f"Merged {len(chunks)} chunks into {len(merged_chunks)} chunks")
        return merged_chunks
    
    def process_sentences(self, sentences: List[str], 
                         chunk_size: int = 5,
                         overlap: int = 2,
                         similarity_threshold: float = 0.85) -> List[Dict]:
        """
        Complete semantic chunking pipeline.
        
        Args:
            sentences: List of sentences
            chunk_size: Number of sentences per chunk
            overlap: Number of overlapping sentences
            similarity_threshold: Threshold for merging
            
        Returns:
            List of final chunks
        """
        # Create initial chunks
        chunks = self.create_chunks(sentences, chunk_size, overlap)
        
        # Merge similar chunks
        final_chunks = self.merge_similar_chunks(chunks, similarity_threshold)
        
        return final_chunks
    
    def get_embedding_dimension(self) -> int:
        """Get the embedding dimension of the model."""
        return self.model.get_sentence_embedding_dimension() 