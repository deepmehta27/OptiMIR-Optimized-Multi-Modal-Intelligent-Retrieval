FROM python:3.11-slim

WORKDIR /app

# System deps for pdftotext / poppler etc. if you use them
RUN apt-get update && apt-get install -y \
    build-essential \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

COPY backend/pyproject.toml backend/requirements.txt* ./
RUN pip install --no-cache-dir -r requirements.txt

COPY backend /app

# Chroma persistence directory
VOLUME ["/app/chroma_db"]

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
