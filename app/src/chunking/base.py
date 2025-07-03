import logging
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class ChunkerBase:
    """Base class for all chunkers."""
    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name
        self.model = None
        if model_name:
            self.model = SentenceTransformer(model_name)
            logger.info(f"Initialized {self.__class__.__name__} with model: {model_name}")

    def create_chunks(self, text: str, **kwargs) -> List[Dict]:
        """Abstract method to create chunks from text."""
        raise NotImplementedError

    def get_embedding(self, text: str):
        if self.model:
            return self.model.encode(text)
        return None

    def get_embedding_dimension(self) -> Optional[int]:
        if self.model:
            return self.model.get_sentence_embedding_dimension()
        return None 