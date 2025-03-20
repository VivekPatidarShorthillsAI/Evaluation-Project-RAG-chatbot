from langchain_community.vectorstores import FAISS
from settings import Settings
from timings import logger, time_it
import faiss
import pickle
import numpy as np
import os

class DBManager:
    def __init__(self, embedder):
        self.embedder = embedder
        self.db_file = os.path.join(Settings.DB_FOLDER, "faiss_index.pkl")
        self.index_file = os.path.join(Settings.DB_FOLDER, "faiss_index.faiss")
        self.db = self._load_or_create_db()

    def _load_or_create_db(self):
        if os.path.exists(self.index_file):
            try:
                index = faiss.read_index(self.index_file)
                logger.info("Loaded existing FAISS index.")
                return index
            except Exception as e:
                logger.error(f"Error loading FAISS index: {str(e)}")

        logger.info("Creating new FAISS index.")
        return self._create_db()

    def _create_db(self):
        index = faiss.IndexFlatL2(384)  # Assuming 384-d embeddings
        self._save_db(index)
        return index

    def is_faiss_index_loaded(self):  # ✅ Fix for missing method
        return self.db is not None and self.db.ntotal > 0

    @time_it
    def add_docs(self, documents):
        if not documents:
            logger.warning("No documents to add to FAISS database.")
            return

        try:
            texts = [doc.page_content for doc in documents]
            metadatas = [doc.metadata for doc in documents]

            embeddings = self.embedder.embed_documents(texts)  # ✅ Fixed method call

            if len(embeddings) == 0:
                logger.warning("No valid embeddings generated.")
                return

            embeddings = np.array(embeddings, dtype=np.float32)  # ✅ Convert list to NumPy array

            if not self.db.is_trained:
                self.db.train(embeddings)  # Train if needed

            self.db.add(embeddings)  # ✅ Fixed: FAISS requires NumPy array
            self._save_db(self.db)

            logger.info(f"Added {len(documents)} documents to FAISS database.")
        except Exception as e:
            logger.error(f"Error adding documents to FAISS database: {str(e)}")
            raise

    def search(self, query, k=4):
        try:
            query_embedding = self.embedder.embed_query(query)  # ✅ Fixed method call
            query_embedding = np.array([query_embedding], dtype=np.float32)  # ✅ Convert to NumPy array
            distances, indices = self.db.search(query_embedding, k)

            if indices.shape[0] == 0:
                return []

            logger.info(f"Query: '{query}' - Retrieved {len(indices)} results")

            results = []
            for idx in indices[0]:
                if idx >= 0:
                    results.append(self.db.reconstruct(idx))  # Retrieve vector

            return results
        except Exception as e:
            logger.error(f"Error searching FAISS database: {str(e)}")
            return []

    def _save_db(self, db):
        os.makedirs(os.path.dirname(self.index_file), exist_ok=True)
        try:
            faiss.write_index(db, self.index_file)  # Save FAISS index
            logger.info("FAISS database saved successfully.")
        except Exception as e:
            logger.error(f"Error saving FAISS database: {str(e)}")
            raise
