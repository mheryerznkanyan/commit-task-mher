"""
Example usage of the Research Pipeline system.
"""

import logging
from research_pipeline import ResearchPipeline
from arxiv_client import ArXivClient
from pdf_processor import PDFProcessor
from semantic_chunker import SemanticChunker
from qdrant_database import QdrantDatabase
from faiss_database import FaissDatabase

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_complete_pipeline():
    """Example of running the complete pipeline."""
    print("=== Complete Pipeline Example ===")
    
    pipeline = ResearchPipeline()
    
    # Run complete pipeline
    results = pipeline.run_complete_pipeline("machine learning", max_results=2)
    
    print(f"Processed {results['papers_processed']} papers")
    print(f"Created {results['total_chunks']} chunks")
    
    # Search the database
    search_results = pipeline.search_database("neural networks", top_k=3)
    print(f"Found {len(search_results)} search results")
    
    return results


def example_component_usage():
    """Example of using individual components."""
    print("\n=== Component Usage Example ===")
    
    # ArXiv client
    arxiv_client = ArXivClient()
    papers = arxiv_client.search("deep learning", max_results=2)
    print(f"Found {len(papers)} papers")
    
    if papers:
        # Download first paper
        paper = papers[0]
        pdf_path = arxiv_client.download_pdf(paper['arxiv_id'])
        
        if pdf_path:
            # Process PDF
            pdf_processor = PDFProcessor()
            pdf_data = pdf_processor.process_pdf(pdf_path)
            print(f"Extracted {pdf_data['num_sentences']} sentences")
            
            # Create chunks
            chunker = SemanticChunker()
            chunks = chunker.process_sentences(pdf_data['sentences'])
            print(f"Created {len(chunks)} chunks")
            
            # Add to database
            db = QdrantDatabase()
            success = db.add_paper(
                paper['arxiv_id'],
                paper['title'],
                paper['summary'],
                paper['link'],
                chunks
            )
            print(f"Added to database: {success}")
            
            # Search
            results = db.search("artificial intelligence", top_k=2)
            print(f"Search returned {len(results)} results")


def example_database_operations():
    """Example of database operations."""
    print("\n=== Database Operations Example ===")
    
    db = QdrantDatabase()
    
    # Get collection info
    info = db.get_collection_info()
    print(f"Collection: {info.get('name', 'N/A')}")
    print(f"Points: {info.get('points_count', 'N/A')}")
    
    # Search with filters
    results = db.search("transformer", top_k=5)
    print(f"Search results: {len(results)}")
    
    if results:
        # Filter by specific paper
        arxiv_id = results[0]['arxiv_id']
        filtered_results = db.search("transformer", top_k=3, filter_arxiv_id=arxiv_id)
        print(f"Filtered results: {len(filtered_results)}")


def example_custom_chunking():
    """Example of custom chunking parameters."""
    print("\n=== Custom Chunking Example ===")
    
    # Create chunker with custom parameters
    chunker = SemanticChunker(model_name='all-MiniLM-L6-v2')
    
    # Sample text (in practice, this would come from PDF)
    sample_sentences = [
        "Machine learning is a subset of artificial intelligence.",
        "It focuses on algorithms that can learn from data.",
        "Deep learning is a type of machine learning.",
        "Neural networks are the foundation of deep learning.",
        "Transformers have revolutionized natural language processing.",
        "Attention mechanisms are key to transformer architecture.",
        "BERT and GPT are popular transformer models.",
        "They have achieved state-of-the-art results in many tasks."
    ]
    
    # Create chunks with custom parameters
    chunks = chunker.process_sentences(
        sentences=sample_sentences,
        chunk_size=3,  # Smaller chunks
        overlap=1,     # Less overlap
        similarity_threshold=0.8  # Lower threshold
    )
    
    print(f"Created {len(chunks)} chunks from {len(sample_sentences)} sentences")
    
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i}: {len(chunk['sentences'])} sentences")


def main():
    """Run all examples."""
    try:
        # Example 1: Complete pipeline
        example_complete_pipeline()
        
        # Example 2: Component usage
        example_component_usage()
        
        # Example 3: Database operations
        example_database_operations()
        
        # Example 4: Custom chunking
        example_custom_chunking()
        
        print("\n=== All examples completed successfully ===")
        
    except Exception as e:
        logger.error(f"Example failed: {e}")
        raise


if __name__ == "__main__":
    main() 