"""
PDF Processor for text extraction and preprocessing.
"""

import PyPDF2
import re
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


class PDFProcessor:
    """Handles PDF text extraction and preprocessing."""
    
    def __init__(self):
        pass
    
    def extract_text(self, pdf_path: str) -> str:
        """
        Extract text from PDF file.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted text string
        """
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text
                
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {e}")
            return ""
    
    def preprocess_text(self, text: str) -> str:
        """
        Clean and normalize extracted text.
        
        Args:
            text: Raw text from PDF
            
        Returns:
            Cleaned and normalized text
        """
        # Remove special characters and normalize whitespace
        text = re.sub(r'[\n\r\t]+', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        
        # Fix common PDF artifacts
        text = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', text)
        text = re.sub(r'(?<=[.!?])(?=[a-zA-Z])', ' ', text)
        
        return text.strip()
    
    def split_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.
        
        Args:
            text: Input text
            
        Returns:
            List of sentences
        """
        # Split by sentence endings followed by space and capital letter
        raw_sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        
        # Filter out empty sentences and very short ones
        sentences = [s.strip() for s in raw_sentences if len(s.strip()) > 10]
        
        logger.info(f"Extracted {len(sentences)} sentences from text")
        return sentences
    
    def process_pdf(self, pdf_path: str) -> Dict:
        """
        Complete PDF processing pipeline.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Dictionary with extracted text and sentences
        """
        # Extract text
        raw_text = self.extract_text(pdf_path)
        if not raw_text:
            return {'text': '', 'sentences': []}
        
        # Preprocess text
        cleaned_text = self.preprocess_text(raw_text)
        
        # Split into sentences
        sentences = self.split_sentences(cleaned_text)
        
        return {
            'text': cleaned_text,
            'sentences': sentences,
            'text_length': len(cleaned_text),
            'num_sentences': len(sentences)
        } 