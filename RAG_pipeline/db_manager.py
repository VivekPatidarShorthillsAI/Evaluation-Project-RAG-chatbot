# RAG_pipeline/db_manager.py
import faiss
import numpy as np
import os
import pickle
import logging
from typing import List, Tuple

class DBManager:
    def __init__(self, index_path='assets/faiss_index_file', text_path='assets/crawler_results.txt'):
        self.index_path = index_path
        self.text_path = text_path
        self.index = None
        self.text_chunks = []
        logging.basicConfig(filename='assets/chatbot.log', level=logging.INFO)
        
    def build_index(self, embeddings: np.ndarray, text_chunks: List[str]) -> None:
        """Build and save FAISS index"""
        try:
            logging.info("Building FAISS index")
            dim = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dim)
            self.index.add(embeddings)
            self.text_chunks = text_chunks
            
            # Save index and text chunks
            faiss.write_index(self.index, self.index_path)
            with open(f"{self.index_path}_chunks.pkl", 'wb') as f:
                pickle.dump(text_chunks, f)
        except Exception as e:
            logging.error(f"Error building index: {str(e)}")
            raise
            
    def load_index(self) -> Tuple[faiss.Index, List[str]]:
        """Load existing FAISS index and text chunks"""
        try:
            if os.path.exists(self.index_path):
                logging.info("Loading existing FAISS index")
                self.index = faiss.read_index(self.index_path)
                with open(f"{self.index_path}_chunks.pkl", 'rb') as f:
                    self.text_chunks = pickle.load(f)
                return self.index, self.text_chunks
            return None, []
        except Exception as e:
            logging.error(f"Error loading index: {str(e)}")
            raise
            
    def search(self, query_embedding: np.ndarray, k: int = 3) -> List[Tuple[int, float, str]]:
        """Search the index for similar vectors"""
        try:
            if self.index is None:
                raise ValueError("Index not initialized")
            
            distances, indices = self.index.search(query_embedding, k)
            results = []
            for i in range(len(indices[0])):
                idx = indices[0][i]
                dist = distances[0][i]
                text = self.text_chunks[idx]
                results.append((idx, dist, text))
            return results
        except Exception as e:
            logging.error(f"Error searching index: {str(e)}")
            raise