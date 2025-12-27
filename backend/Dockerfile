FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y \
    build-essential \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# FIX: Copy files directly, do not prefix with "backend/"
COPY requirements.txt* ./
RUN pip install --no-cache-dir -r requirements.txt

# FIX: Copy the current directory (.) into /app
COPY . /app

# Chroma persistence directory
VOLUME ["/app/chroma_db"]

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
