"""
Semantic Search Interface for Research Vector Database (FAISS or Qdrant).

Run this script to perform searches over the existing vector database after it has been built.
"""

import argparse
import logging
from faiss_database import FaissDatabase
# from qdrant_database import QdrantDatabase  # Uncomment if you want to support Qdrant
import os
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Semantic Search Interface for Research Vector Database")
    parser.add_argument('--db', type=str, default='faiss', choices=['faiss'], help='Database type (faiss)')
    parser.add_argument('--faiss-index', type=str, default=None, help='Path prefix for FAISS index (if saved)')
    parser.add_argument('--top-k', type=int, default=5, help='Number of top results to show')
    args = parser.parse_args()

    # Load the vector database
    if args.db == 'faiss':
        db = FaissDatabase()
        db.load()
        if args.faiss_index:
            if os.path.exists(args.faiss_index + '.index') and os.path.exists(args.faiss_index + '.chunks.pkl'):
                db.load(args.faiss_index)
                logger.info(f"Loaded FAISS index from {args.faiss_index}")
            else:
                logger.warning(f"FAISS index files not found at {args.faiss_index}, using in-memory DB.")
        else:
            logger.info("Using in-memory FAISS DB (from current session)")
    # elif args.db == 'qdrant':
    #     db = QdrantDatabase()  # Add Qdrant support if needed
    else:
        logger.error("Unsupported database type")
        sys.exit(1)

    print("\nSemantic Search Interface (type 'exit' to quit)")
    print(f"Database: {args.db}")
    print(f"Top-K results: {args.top_k}")

    while True:
        try:
            query = input("\nEnter your search query: ").strip()
            if query.lower() in ('exit', 'quit'):
                print("Exiting search interface.")
                break
            if not query:
                continue
            results = db.search(query, top_k=args.top_k)
            if not results:
                print("No results found. The database may be empty or your query did not match any chunks.")
                continue
            print(f"\nResults for: '{query}' (top {args.top_k})")
            for i, result in enumerate(results, 1):
                print(f"\nResult {i}:")
                print(f"  Score: {result['score']:.3f}")
                print(f"  Paper: {result.get('title', 'N/A')}")
                print(f"  ArXiv ID: {result.get('arxiv_id', 'N/A')}")
                print(f"  Text preview: {result['text'][:200]}...")
        except KeyboardInterrupt:
            print("\nExiting search interface.")
            break
        except Exception as e:
            logger.error(f"Search failed: {e}", exc_info=True)

if __name__ == "__main__":
    main() 