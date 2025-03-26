# src/query_handler.py
from RAG_pipeline.embedder import Embedder
from RAG_pipeline.db_manager import DBManager
import logging

class QueryHandler:
    def __init__(self):
        self.embedder = Embedder()
        self.db_manager = DBManager()
        self.index, self.text_chunks = self.db_manager.load_index()
        logging.basicConfig(filename='assets/chatbot.log', level=logging.INFO)
        
    def process_query(self, query: str, k: int = 3) -> list:
        """Process user query and return relevant chunks"""
        try:
            logging.info(f"Processing query: {query}")
            query_embedding = self.embedder.embed_text(query)
            query_embedding = query_embedding.reshape(1, -1)
            
            if self.index is None:
                raise ValueError("FAISS index not loaded")
                
            results = self.db_manager.search(query_embedding, k=k)
            return results
        except Exception as e:
            logging.error(f"Error processing query: {str(e)}")
            raise