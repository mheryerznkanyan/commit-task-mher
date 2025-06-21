"""
Qdrant Vector Database for storing and querying semantic chunks.
"""

from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, Distance, VectorParams
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class QdrantDatabase:
    """Vector database using Qdrant for semantic search."""
    
    def __init__(self, host: str = "localhost", port: int = 6333, 
                 collection_name: str = "arxiv_chunks",
                 model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize Qdrant database.
        
        Args:
            host: Qdrant host
            port: Qdrant port
            collection_name: Name of the collection
            model_name: Sentence transformer model name
        """
        self.client = QdrantClient(host, port=port)
        self.collection_name = collection_name
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        self._initialize_collection()
        logger.info(f"Initialized Qdrant database with collection: {collection_name}")
    
    def _initialize_collection(self):
        """Initialize the collection if it doesn't exist."""
        try:
            # Check if collection exists
            collections = self.client.get_collections()
            collection_names = [c.name for c in collections.collections]
            
            if self.collection_name not in collection_names:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_dim,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Created collection: {self.collection_name}")
            else:
                logger.info(f"Using existing collection: {self.collection_name}")
                
        except Exception as e:
            logger.error(f"Error initializing collection: {e}")
            raise
    
    def add_paper(self, arxiv_id: str, title: str, summary: str, 
                  link: str, chunks: List[Dict]) -> bool:
        """
        Add a paper and its chunks to the database.
        
        Args:
            arxiv_id: ArXiv ID
            title: Paper title
            summary: Paper summary
            link: Paper link
            chunks: List of chunk dictionaries
            
        Returns:
            True if successful
        """
        try:
            points = []
            
            for i, chunk in enumerate(chunks):
                point = PointStruct(
                    id=f"{arxiv_id}_{i}",
                    vector=chunk['embedding'].tolist(),
                    payload={
                        "arxiv_id": arxiv_id,
                        "title": title,
                        "summary": summary,
                        "link": link,
                        "chunk_id": chunk['chunk_id'],
                        "text": chunk['text'],
                        "sentences": chunk['sentences'],
                        "start_sentence_idx": chunk['start_sentence_idx'],
                        "end_sentence_idx": chunk['end_sentence_idx']
                    }
                )
                points.append(point)
            
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            
            logger.info(f"Added paper {arxiv_id} with {len(chunks)} chunks")
            return True
            
        except Exception as e:
            logger.error(f"Error adding paper {arxiv_id}: {e}")
            return False
    
    def search(self, query: str, top_k: int = 10, 
               filter_arxiv_id: Optional[str] = None) -> List[Dict]:
        """
        Search for similar chunks.
        
        Args:
            query: Search query
            top_k: Number of results to return
            filter_arxiv_id: Optional filter by ArXiv ID
            
        Returns:
            List of search results
        """
        try:
            # Encode query
            query_vector = self.model.encode(query).tolist()
            
            # Build search parameters
            search_params = {
                "collection_name": self.collection_name,
                "query_vector": query_vector,
                "limit": top_k
            }
            
            # Add filter if specified
            if filter_arxiv_id:
                search_params["query_filter"] = {
                    "must": [
                        {"key": "arxiv_id", "match": {"value": filter_arxiv_id}}
                    ]
                }
            
            # Perform search
            hits = self.client.search(**search_params)
            
            # Format results
            results = []
            for hit in hits:
                result = {
                    'score': hit.score,
                    'arxiv_id': hit.payload['arxiv_id'],
                    'title': hit.payload['title'],
                    'chunk_id': hit.payload['chunk_id'],
                    'text': hit.payload['text'],
                    'sentences': hit.payload['sentences'],
                    'start_sentence_idx': hit.payload['start_sentence_idx'],
                    'end_sentence_idx': hit.payload['end_sentence_idx']
                }
                results.append(result)
            
            logger.info(f"Search returned {len(results)} results for query: {query}")
            return results
            
        except Exception as e:
            logger.error(f"Error during search: {e}")
            return []
    
    def get_collection_info(self) -> Dict:
        """Get information about the collection."""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                'name': info.name,
                'vectors_count': info.vectors_count,
                'points_count': info.points_count,
                'status': info.status
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {}
    
    def delete_paper(self, arxiv_id: str) -> bool:
        """
        Delete all chunks for a specific paper.
        
        Args:
            arxiv_id: ArXiv ID to delete
            
        Returns:
            True if successful
        """
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector={
                    "filter": {
                        "must": [
                            {"key": "arxiv_id", "match": {"value": arxiv_id}}
                        ]
                    }
                }
            )
            logger.info(f"Deleted paper {arxiv_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting paper {arxiv_id}: {e}")
            return False
    
    def clear_collection(self) -> bool:
        """Clear all data from the collection."""
        try:
            self.client.delete_collection(self.collection_name)
            self._initialize_collection()
            logger.info("Cleared collection")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
            return False 