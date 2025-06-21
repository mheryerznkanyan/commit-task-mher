"""
Main Research Pipeline for processing ArXiv papers and building a vector database.
"""

import os
import json
from typing import List, Dict, Optional
import logging

from arxiv_client import ArXivClient
from pdf_processor import PDFProcessor
from semantic_chunker import SemanticChunker
from faiss_database import FaissDatabase

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ResearchPipeline:
    """Main pipeline for processing research papers and building a vector database."""
    
    def __init__(self, 
                 downloads_dir: str = "downloads",
                 chunks_dir: str = "chunks",
                    ):
        """
        Initialize the research pipeline.
        
        Args:
            downloads_dir: Directory for downloaded PDFs
            chunks_dir: Directory for saved chunks
            qdrant_host: Qdrant host
            qdrant_port: Qdrant port
        """
        self.downloads_dir = downloads_dir
        self.chunks_dir = chunks_dir
        
        # Create directories
        os.makedirs(downloads_dir, exist_ok=True)
        os.makedirs(chunks_dir, exist_ok=True)
        
        # Initialize components
        self.arxiv_client = ArXivClient()
        self.pdf_processor = PDFProcessor()
        self.chunker = SemanticChunker()
        self.database = FaissDatabase()
        
        logger.info("Research pipeline initialized")
    
    def search_and_download(self, query: str, max_results: int = 5) -> List[Dict]:
        """
        Search for papers and download them.
        
        Args:
            query: Search query
            max_results: Maximum number of papers to download
            
        Returns:
            List of paper information dictionaries
        """
        # Search for papers
        papers = self.arxiv_client.search(query, max_results)
        
        if not papers:
            logger.warning("No papers found for query")
            return []
        
        # Download PDFs
        arxiv_ids = [paper['arxiv_id'] for paper in papers]
        downloaded = self.arxiv_client.download_papers(arxiv_ids, self.downloads_dir)
        
        # Add PDF paths to paper info
        for paper in papers:
            paper['pdf_path'] = downloaded.get(paper['arxiv_id'])
        
        logger.info(f"Downloaded {len(downloaded)} papers")
        return papers
    
    def process_paper(self, pdf_path: str, arxiv_id: str) -> Optional[List[Dict]]:
        """
        Process a single paper: extract text, create chunks, and add to database.
        
        Args:
            pdf_path: Path to PDF file
            arxiv_id: ArXiv ID of the paper
            
        Returns:
            List of chunks or None if failed
        """
        try:
            # Process PDF
            pdf_data = self.pdf_processor.process_pdf(pdf_path)
            
            if not pdf_data['sentences']:
                logger.warning(f"No sentences extracted from {pdf_path}")
                return None
            
            # Create semantic chunks
            chunks = self.chunker.process_sentences(
                pdf_data['sentences'],
                chunk_size=5, # How many sentences per chunk
                overlap=2, # How many sentences to overlap
                similarity_threshold=0.85 # How similar the chunks should be
            )
            
            # Add arxiv_id to chunks
            for chunk in chunks:
                chunk['arxiv_id'] = arxiv_id
            
            logger.info(f"Processed paper {arxiv_id}: {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing paper {arxiv_id}: {e}")
            return None
    
    def save_chunks(self, chunks: List[Dict], arxiv_id: str) -> str:
        """
        Save chunks to JSON file.
        
        Args:
            chunks: List of chunks
            arxiv_id: ArXiv ID
            
        Returns:
            Path to saved file
        """
        file_path = os.path.join(self.chunks_dir, f"{arxiv_id}_chunks.json")
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_chunks = []
        for chunk in chunks:
            serializable_chunk = chunk.copy()
            serializable_chunk['embedding'] = chunk['embedding'].tolist()
            serializable_chunks.append(serializable_chunk)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_chunks, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved chunks to {file_path}")
        return file_path
    
    def add_paper_to_database(self, paper: Dict, chunks: List[Dict]) -> bool:
        """
        Add a paper and its chunks to the vector database.
        
        Args:
            paper: Paper information dictionary
            chunks: List of chunks
            
        Returns:
            True if successful
        """
        return self.database.add_paper(
            arxiv_id=paper['arxiv_id'],
            title=paper['title'],
            summary=paper['summary'],
            link=paper['link'],
            chunks=chunks
        )
    
    def process_papers(self, papers: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Process multiple papers.
        
        Args:
            papers: List of paper information dictionaries
            
        Returns:
            Dictionary mapping ArXiv IDs to chunks
        """
        all_chunks = {}
        
        for paper in papers:
            if not paper.get('pdf_path'):
                logger.warning(f"No PDF path for paper {paper['arxiv_id']}")
                continue
            
            chunks = self.process_paper(paper['pdf_path'], paper['arxiv_id'])
            if chunks:
                # Save chunks
                self.save_chunks(chunks, paper['arxiv_id'])
                
                # Add to database
                success = self.add_paper_to_database(paper, chunks)
                if success:
                    all_chunks[paper['arxiv_id']] = chunks
                else:
                    logger.error(f"Failed to add paper {paper['arxiv_id']} to database")
        
        logger.info(f"Processed {len(all_chunks)} papers successfully")
        return all_chunks
    
    def search_database(self, query: str, top_k: int = 10, 
                       filter_arxiv_id: Optional[str] = None) -> List[Dict]:
        """
        Search the vector database.
        
        Args:
            query: Search query
            top_k: Number of results to return
            filter_arxiv_id: Optional filter by ArXiv ID
            
        Returns:
            List of search results
        """
        return self.database.search(query, top_k, filter_arxiv_id)
    
    def get_database_stats(self) -> Dict:
        """Get database statistics."""
        return self.database.get_collection_info()
    
    def run_complete_pipeline(self, query: str, max_results: int = 5) -> Dict:
        """
        Run the complete pipeline: search, download, process, and index.
        
        Args:
            query: Search query
            max_results: Maximum number of papers to process
            
        Returns:
            Pipeline results summary
        """
        logger.info(f"Starting complete pipeline for query: {query}")
        
        # Step 1: Search and download
        papers = self.search_and_download(query, max_results)
        
        if not papers:
            return {'error': 'No papers found'}
        
        # Step 2: Process papers
        processed_chunks = self.process_papers(papers)
        
        # Automatically save FAISS database after processing
        if isinstance(self.database, FaissDatabase):
            logger.info("Saving FAISS database to disk...")
            os.makedirs("app/vector_db", exist_ok=True)
            self.database.save("app/vector_db/faiss_index")
            logger.info("FAISS database saved.")

        # Step 3: Get database stats
        db_stats = self.get_database_stats()
        
        # Summary
        summary = {
            'query': query,
            'papers_found': len(papers),
            'papers_processed': len(processed_chunks),
            'total_chunks': sum(len(chunks) for chunks in processed_chunks.values()),
            'database_stats': db_stats,
            'arxiv_ids': list(processed_chunks.keys())
        }
        
        logger.info(f"Pipeline completed: {summary}")
        return summary 