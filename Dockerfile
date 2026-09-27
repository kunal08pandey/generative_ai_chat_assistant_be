FROM python:3.10-slim

WORKDIR /app

# Ollama normally runs on the Docker host during local development.
# Override this environment variable for remote or Linux deployments.
ENV OLLAMA_BASE_URL=http://host.docker.internal:11434

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Declare persistent volume for SQLite database (chat.db) and ChromaDB vector store
VOLUME ["/app/data"]

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
