# RAG_pipeline/embedder.py
from sentence_transformers import SentenceTransformer
import numpy as np
import logging

class Embedder:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        logging.basicConfig(filename='assets/chatbot.log', level=logging.INFO)
        
    def embed_text(self, text):
        """Convert text to embeddings"""
        try:
            logging.info("Generating embeddings for text")
            if isinstance(text, str):
                return self.model.encode([text])[0]
            elif isinstance(text, list):
                return self.model.encode(text)
            else:
                raise ValueError("Input must be string or list of strings")
        except Exception as e:
            logging.error(f"Error in embedding text: {str(e)}")
            raise