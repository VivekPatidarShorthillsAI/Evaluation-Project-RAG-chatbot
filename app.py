from flask import Flask, render_template, request, jsonify
from waitress import serve
import logging
import os
from settings import Settings
from embedder import Embedder
from db_manager import DBManager
from query_handler import QueryHandler
from responder import Responder
from text_processor import TextProcessor

# Simple in-memory conversation context
class MemoryContext:
    def __init__(self, max_history=5):
        self.history = []
        self.max_history = max_history
        self.current_context = None

    def add(self, question, answer):
        self.history.append((question, answer))
        if len(self.history) > self.max_history:
            self.history.pop(0)

    def set_current_context(self, context):
        self.current_context = context

    def get_context(self):
        context = "\n\n".join([f"Q: {q}\nA: {a}" for q, a in self.history])
        if self.current_context:
            context += f"\n\nCurrent context: {self.current_context}"
        return context

# Initialize Flask app
app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize vector embedding components
embedder = Embedder()
db_manager = DBManager(embedder)

# Ensure `SCRAPED_FILE_NAME` exists in Settings
if hasattr(Settings, "SCRAPED_FILE_NAME"):
    file_path = os.path.join(Settings.TEXT_FILES_FOLDER, Settings.SCRAPED_FILE_NAME)

    # Check if FAISS is already populated before adding documents
    if not db_manager.is_faiss_index_loaded():
        logger.info("FAISS index is empty. Processing new data.")
        documents = TextProcessor.process_text(file_path)
        if documents:
            db_manager.add_docs(documents)
            logger.info("Documents added to FAISS successfully.")
        else:
            logger.warning("No documents found to process.")
    else:
        logger.info("FAISS database already loaded. Skipping ingestion.")
else:
    logger.error("SCRAPED_FILE_NAME is not defined in Settings. Please check the configuration.")

query_handler = QueryHandler(db_manager)
responder = Responder()
memory = MemoryContext()

@app.route('/')
def index():
    return render_template('chat.html')

@app.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.json
        question = data.get('question', '').strip()
        if not question:
            return jsonify({"error": "Empty question received."}), 400

        # Retrieve context from the vector store using the query
        context = query_handler.handle(question, memory)
        # Generate answer using the retrieved context
        answer = responder.respond(question, context, memory)

        # Update memory with new question and answer
        memory.add(question, answer)
        memory.set_current_context(f"{question} - {answer[:100]}...")

        return jsonify({'answer': answer})
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    logger.info("Starting server at http://127.0.0.1:5000 ...")
    serve(app, host='127.0.0.1', port=5000)
