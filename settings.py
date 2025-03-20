import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # Use a folder for the FAISS index; here it's a folder.
    DB_FOLDER = os.getenv('DB_FOLDER', './faiss_index')
    
    # Folder containing your text files.
    TEXT_FILES_FOLDER = os.getenv('TEXT_FILES_FOLDER', './text_files')
    
    # Scraped file name for ingestion (added attribute)
    SCRAPED_FILE_NAME = os.getenv('SCRAPED_FILE_NAME', 'crawler_results.txt')
    
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', 500))
    CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', 100))

    @classmethod
    def check(cls):
        required_vars = ['GEMINI_API_KEY']
        for var in required_vars:
            if not getattr(cls, var):
                raise ValueError(f"Missing required environment variable: {var}")
