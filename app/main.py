"""
Main script demonstrating the complete research pipeline.
"""

import logging
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
import research_pipeline
from faiss_database import FaissDatabase

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

os.makedirs("app/chunks", exist_ok=True)
os.makedirs("app/downloads", exist_ok=True)

def main():
    """Run the complete research pipeline."""
    
    # Initialize pipeline
    pipeline = research_pipeline.ResearchPipeline()
    
    # Example query
    query = "large language models"
    max_results = 15
    
    logger.info("Starting research pipeline")
    logger.info(f"Query: {query}")
    logger.info(f"Max results: {max_results}")
    
    # Run complete pipeline
    try:
        results = pipeline.run_complete_pipeline(query, max_results)
        logger.info(pipeline.get_database_stats())
        
        if 'error' in results:
            logger.error(f"Pipeline failed: {results['error']}")
            return
        
        # Print results
        logger.info("\n" + "="*60)
        logger.info("PIPELINE RESULTS")
        logger.info("="*60)
        logger.info(f"Query: {results['query']}")
        logger.info(f"Papers found: {results['papers_found']}")
        logger.info(f"Papers processed: {results['papers_processed']}")
        logger.info(f"Total chunks: {results['total_chunks']}")
        logger.info(f"ArXiv IDs: {', '.join(results['arxiv_ids'])}")
        
        # Database stats
        db_stats = results['database_stats']
        if db_stats:
            logger.info(f"\nDatabase Statistics:")
            logger.info(f"  Collection: {db_stats.get('name', 'N/A')}")
            logger.info(f"  Points: {db_stats.get('points_count', 'N/A')}")
            logger.info(f"  Status: {db_stats.get('status', 'N/A')}")
        
        # Example search
        logger.info("\n" + "="*60)
        logger.info("EXAMPLE SEARCH")
        logger.info("="*60)
        
        search_query = "transformer architecture"
        search_results = pipeline.search_database(search_query, top_k=3)
        
        logger.info(f"Search query: {search_query}")
        logger.info(f"Results found: {len(search_results)}")
        
        for i, result in enumerate(search_results, 1):
            logger.debug(f"\nResult {i}:")
            logger.debug(f"  Score: {result['score']:.3f}")
            logger.debug(f"  Paper: {result['title']}")
            logger.debug(f"  ArXiv ID: {result['arxiv_id']}")
            logger.debug(f"  Text preview: {result['text'][:200]}...")
        
        logger.info("Pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}")
        raise


def get_related_texts(query_text, top_k=5, db_path="app/vector_db/faiss_index"):
    """
    Load the FAISS database and return the most related texts and their scores to the given query_text.
    """
    db = FaissDatabase()
    db.load(db_path)
    results = db.search(query_text, top_k=top_k)
    return [(r['text'], r['score']) for r in results]


if __name__ == "__main__":
    # main()


    logger.info("="*60)
    logger.info("EXAMPLE SEARCH")
    logger.info("="*60)
    related = get_related_texts("transformer models for healthcare", top_k=10)
    for i, (text, score) in enumerate(related, 1):
        logger.info(f"Result {i} (Score: {score:.3f}):\n{text}\n")

    print("\nTo run the FastAPI server, use:")
    print("uvicorn app.src.api:app --reload --port 8000")
    print("(Run this command from the project root directory)") 

