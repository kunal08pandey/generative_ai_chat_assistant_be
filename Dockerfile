FROM python:3.10-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OLLAMA_BASE_URL=http://host.docker.internal:11434

# Upgrade pip and install dependencies with prefer-binary to prevent source compilation hangs
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir --prefer-binary -r requirements.txt

COPY . .
COPY .env* ./

# Declare persistent volume for SQLite database (chat.db) and ChromaDB vector store
VOLUME ["/app/data"]

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
