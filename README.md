# Generative AI Backend

This is the backend service for the Generative AI project. It provides RESTful APIs for interacting with Large Language Models (LLMs), featuring a Retrieval-Augmented Generation (RAG) system. The backend is built with FastAPI and integrates with Ollama for local LLM inference and ChromaDB for vector storage.

## Features

- **FastAPI:** High-performance, async-ready REST API.
- **RAG Capabilities:** Integrates document processing (PDF, DOCX) with vector embeddings using ChromaDB.
- **Ollama Integration:** Uses local models via Ollama by default, ensuring privacy and local execution.
- **SQLite Database:** Stores chat history and metadata locally via `aiosqlite`.

## Prerequisites

- **Python 3.10+**
- **Ollama:** Make sure you have [Ollama](https://ollama.com/) installed and running locally if you're using local models.

## Setup & Installation

1. **Navigate to the backend directory:**
   ```bash
   cd backend
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

The backend is configured using environment variables. You can create a `.env` file in the root of the `backend` directory.

### Key Environment Variables

| Variable | Default Value | Description |
| --- | --- | --- |
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | The endpoint for the Ollama service. |
| `DATABASE_URL` | `data/chat.db` | Path to the SQLite database. |
| `CHROMA_PATH` | `data/chroma` | Path to the ChromaDB vector storage. |
| `EMBED_MODEL` | `nomic-embed-text` | Model used for document embeddings. |
| `CORS_ORIGINS` | `http://localhost:5173,...` | Allowed CORS origins (e.g., frontend URL). |
| `MAX_UPLOAD_BYTES` | `20971520` (20MB) | Max allowed document size. |
| `ADMIN_API_KEY` | None | API key for admin routes (optional). |

## Running the Application

To start the FastAPI development server, use Uvicorn:

```bash
uvicorn src.api.main:app --reload
```

The server will start at `http://127.0.0.1:8000`. You can access the automatic interactive API documentation at:
- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

## Testing

The project uses `pytest` for testing. Run the test suite with:

```bash
pytest
```
