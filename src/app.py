# src/app.py
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from RAG_pipeline.embedder import Embedder
from RAG_pipeline.chunker import Chunker
from RAG_pipeline.db_manager import DBManager
import os
import logging

def initialize_system():
    """Initialize the RAG system, building index if needed"""
    logging.basicConfig(filename='assets/chatbot.log', level=logging.INFO)
    
    db_manager = DBManager()
    index, text_chunks = db_manager.load_index()
    
    # If index doesn't exist, create it
    if index is None:
        try:
            logging.info("No existing index found. Building new index.")
            
            # Read the text file
            with open('assets/crawler_results.txt', 'r', encoding='utf-8') as f:
                text = f.read()
                
            # Chunk the text
            chunker = Chunker()
            chunks = chunker.chunk_text(text)
            
            # Generate embeddings
            embedder = Embedder()
            embeddings = embedder.embed_text(chunks)
            
            # Build and save index
            db_manager.build_index(embeddings, chunks)
            logging.info("Index built successfully")
        except Exception as e:
            logging.error(f"Error during system initialization: {str(e)}")
            raise

if __name__ == "__main__":
    initialize_system()
    from src.ui import ChatUI
    ui = ChatUI()
    ui.run()