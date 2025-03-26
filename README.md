# RAG-LLM Chatbot

## Project Description
The **RAG-LLM Chatbot** is a Retrieval-Augmented Generation (RAG) system built using **FAISS, SentenceTransformers, and Gemini AI**. This chatbot is designed to retrieve and process data from the **University of North Dakota website**, storing indexed text chunks and embeddings for efficient querying and response generation. It uses a **FAISS vector database** for similarity searches and a **Gemini LLM** for generating context-aware responses.

## Features
- **Web Scraping**: Extracts and stores university website data.
- **Chunking & Embeddings**: Preprocesses data and converts it into embeddings.
- **FAISS Vector Search**: Efficiently retrieves relevant information.
- **LLM Response Generation**: Provides accurate responses based on retrieved data.
- **Streamlit UI**: User-friendly chatbot interface.
- **Logging & Debugging**: Maintains logs for troubleshooting.

---

## Project Directory Structure
```
EVALUATION_PROJECT_CHATBOT_3/
│── assets/
│   ├── chatbot.log
│   ├── crawler_results.txt
│   ├── faiss_index_file
│   ├── faiss_index_file_chunks.pkl
│   ├── golden_set.json
│   ├── model_output.json
│   ├── evaluation_results.xlsx
│   ├── evaluation_ragas.log
│   ├── evaluation.log
│── RAG_pipeline/
│   ├── chunker.py         # Splits text into overlapping chunks
│   ├── db_manager.py      # Handles FAISS indexing and retrieval
│   ├── embedder.py        # Generates embeddings using SentenceTransformers
│   ├── responder.py       # Uses Gemini API for response generation
│── src/
│   ├── app.py             # Initializes chatbot system
│   ├── scraper.py         # Scrapes university website data
│   ├── query_handler.py   # Handles user queries & searches FAISS index
│   ├── ui.py              # Streamlit-based chatbot UI
│   ├── evaluation.py      # Evaluates chatbot responses
│── venv/                  # Virtual environment
│── .env                   # Environment variables (e.g., GEMINI_API_KEY)
│── .gitignore             # Git ignore file
│── README.md              # Project documentation
│── requirements.txt       # Required Python dependencies
```

---

## Setup Instructions

### 1. Clone the Repository
```sh
git clone <repo_link>
cd EVALUATION_PROJECT_CHATBOT_3
```

### 2. Create and Activate Virtual Environment
```sh
python -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate  # On Windows
```

### 3. Install Dependencies
```sh
pip install -r requirements.txt
```

### 4. Set Up Environment Variables
Create a `.env` file and add your **GEMINI API Key**:
```sh
GEMINI_API_KEY=your_api_key_here
```

### 5. Initialize FAISS Index
```sh
python src/app.py
```
This will:
- Load and chunk `crawler_results.txt`
- Generate embeddings using `SentenceTransformers`
- Build and store the FAISS index

### 6. Run the Chatbot UI
```sh
streamlit run src/ui.py
```
Access the chatbot at `http://localhost:8501/`

---

## Usage
- Enter a query related to the university.
- The system retrieves the top **5 most relevant** text chunks from FAISS.
- The **Gemini LLM** generates a response using retrieved data.
- Chat history is maintained within the session.

---

## Logging & Debugging
- Logs are stored in `assets/chatbot.log`
- All input queries, retrieved text chunks, and responses are logged.
- Errors are reported in `evaluation.log`.

---

## Future Improvements
- **Enhance Web Scraper**: Improve coverage of university pages.
- **Improve Chunking Strategy**: Fine-tune chunk overlap and size.
- **Expand Dataset**: Incorporate additional university-related data sources.

---

## Contributors
- **Vivek Patidar** - Developer


