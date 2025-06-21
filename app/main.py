"""
Main script demonstrating the complete research pipeline.
"""

import logging
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
import research_pipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Run the complete research pipeline."""
    
    # Initialize pipeline
    pipeline = research_pipeline.ResearchPipeline()
    
    # Example query
    query = "large language models 2024"
    max_results = 3
    
    logger.info("Starting research pipeline")
    logger.info(f"Query: {query}")
    logger.info(f"Max results: {max_results}")
    
    # Run complete pipeline
    try:
        results = pipeline.run_complete_pipeline(query, max_results)
        print(pipeline.get_database_stats())
        
        if 'error' in results:
            logger.error(f"Pipeline failed: {results['error']}")
            return
        
        # Print results
        print("\n" + "="*60)
        print("PIPELINE RESULTS")
        print("="*60)
        print(f"Query: {results['query']}")
        print(f"Papers found: {results['papers_found']}")
        print(f"Papers processed: {results['papers_processed']}")
        print(f"Total chunks: {results['total_chunks']}")
        print(f"ArXiv IDs: {', '.join(results['arxiv_ids'])}")
        
        # Database stats
        db_stats = results['database_stats']
        if db_stats:
            print(f"\nDatabase Statistics:")
            print(f"  Collection: {db_stats.get('name', 'N/A')}")
            print(f"  Points: {db_stats.get('points_count', 'N/A')}")
            print(f"  Status: {db_stats.get('status', 'N/A')}")
        
        # Example search
        print(f"\n" + "="*60)
        print("EXAMPLE SEARCH")
        print("="*60)
        
        search_query = "transformer architecture"
        search_results = pipeline.search_database(search_query, top_k=3)
        
        print(f"Search query: {search_query}")
        print(f"Results found: {len(search_results)}")
        
        for i, result in enumerate(search_results, 1):
            print(f"\nResult {i}:")
            print(f"  Score: {result['score']:.3f}")
            print(f"  Paper: {result['title']}")
            print(f"  ArXiv ID: {result['arxiv_id']}")
            print(f"  Text preview: {result['text'][:200]}...")
        
        logger.info("Pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}")
        raise


if __name__ == "__main__":
    main() 