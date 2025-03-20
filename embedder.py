from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings
from timings import time_it, logger
import numpy as np

class Embedder(Embeddings):
    def __init__(self):
        try:
            self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')
            logger.info("Embedding model loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading embedder: {str(e)}")
            raise RuntimeError("Failed to load embedding model.")

    @time_it
    def embed_documents(self, texts):
        try:
            embeddings = self.model.encode(texts, batch_size=32, convert_to_numpy=True)
            logger.info(f"Embedded {len(texts)} documents successfully.")
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Error embedding documents: {str(e)}")
            raise

    @time_it
    def embed_query(self, text):
        try:
            embedding = self.model.encode([text], convert_to_numpy=True)[0]
            logger.info("Query embedded successfully.")
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error embedding query: {str(e)}")
            raise
