# RAG_pipeline/chunker.py
import re
from typing import List
import logging

class Chunker:
    def __init__(self):
        logging.basicConfig(filename='assets/chatbot.log', level=logging.INFO)
        
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks"""
        try:
            logging.info("Chunking text document")
            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            
            words = text.split(' ')
            chunks = []
            current_chunk = []
            current_length = 0
            
            for word in words:
                if current_length + len(word) + 1 > chunk_size:
                    chunks.append(' '.join(current_chunk))
                    # Start new chunk with overlap
                    current_chunk = current_chunk[-overlap:] if overlap > 0 else []
                    current_length = sum(len(w) + 1 for w in current_chunk)
                current_chunk.append(word)
                current_length += len(word) + 1
            
            if current_chunk:
                chunks.append(' '.join(current_chunk))
            
            return chunks
        except Exception as e:
            logging.error(f"Error in chunking text: {str(e)}")
            raise