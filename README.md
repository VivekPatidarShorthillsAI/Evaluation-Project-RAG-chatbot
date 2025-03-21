# Evaluation-Project-RAG-chatbot

## Directory Structure
```
Evaluation-Project-RAG-chatbot/
├── __pycache__/              # Compiled Python files
├── logs/                     # Log files generated during runtime
├── static/                   # Static files (e.g., CSS, JS)
│   ├── style.css             # Stylesheet for UI
├── templates/                # HTML templates for the chatbot UI
│   ├── chat.html             # Chatbot UI page
├── text_files/               # Text files for ingestion and processing
│   ├── crawler_results.txt   # Scraped data from the university website
│   ├── formatted_q_and_r.txt # Formatted questions and responses
│   ├── processed_files.txt   # Processed text data
│   ├── question and answer.txt # Additional QA data
├── vector_store/             # Vector store for embeddings
├── .env                      # Environment variables
├── .gitattributes            # Git attributes file
├── .gitignore                # Git ignore file
├── app.py                    # Flask application entry point
├── db_manager.py             # Database manager for storing and retrieving documents
├── dev.py                    # Development server entry point
├── embedder.py               # Embedding generation logic
├── Evaluation_Metrics.xlsx   # Evaluation metrics file
├── evaluation.py             # Evaluation script for comparing model outputs
├── generate_model_output.py  # Script to generate model outputs
├── Golden_set.json           # Golden set for evaluation
├── main.py                   # Main setup and execution logic
├── Model_Output_Set.json     # Model output for evaluation
├── query_handler.py          # Query handling logic
├── README.md                 # Project documentation
├── responder.py              # Response generation logic
├── scrape.py                 # Web scraping logic
├── settings.py               # Configuration settings
├── text_processor.py         # Text processing logic
├── timings.py                # Logging and timing utilities
├── txt_to_json.py            # Utility to convert text files to JSON
```

## Installation

Follow these steps to set up and run the project:

### 1. Clone the Repository
```bash
git clone https://github.com/your-repo/Evaluation-Project-RAG-chatbot.git
cd Evaluation-Project-RAG-chatbot
```

### 2. Set Up a Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create a `.env` file in the root directory and add the necessary environment variables. Example:
```
FLASK_APP=app.py
FLASK_ENV=development
```

### 5. Run the Development Server
```bash
python dev.py
```

### 6. Access the Chatbot
Open your browser and navigate to `http://127.0.0.1:5000`.

## Project Flow

### 1. Text Ingestion
- Text files are stored in the `text_files/` directory.
- `text_processor.py` processes these files into chunks for embedding.

### 2. Embedding Generation
- `embedder.py` generates embeddings for the processed text chunks.

### 3. Database Storage
- `db_manager.py` stores the embeddings and associated metadata in a FAISS vector store.

### 4. Query Handling
- `query_handler.py` processes user queries and retrieves relevant documents from the vector store.

### 5. Response Generation
- `responder.py` generates a response based on the retrieved documents and user query.

### 6. Evaluation
- `evaluation.py` compares chatbot responses with a golden set to measure accuracy.

## Key Features
- **RAG Architecture**: Combines retrieval and generation for accurate responses.
- **Text Ingestion**: Supports large text file ingestion.
- **Embeddings**: Uses pre-trained embeddings for semantic search.
- **Evaluation**: Built-in evaluation tools for performance measurement.

## Logs and Debugging
Logs are stored in the `logs/` directory. Configure logging settings in `timings.py`.

