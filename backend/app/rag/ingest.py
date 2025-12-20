import asyncio
import fitz
import chromadb
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from .config import OPENAI_API_KEY, CHROMA_DB_DIR

# Used the 'small' model with reduced dimensions for max speed
embedding_fn = OpenAIEmbeddingFunction(
    api_key=OPENAI_API_KEY,
    model_name="text-embedding-3-small",
    dimensions=512  # Reducing dimensions = faster search & lower I/O
)

def get_chroma_client():
    return chromadb.PersistentClient(path=CHROMA_DB_DIR)

def get_or_create_collection(collection_name: str = "documents"):
    client = get_chroma_client()
    # Let Chroma handle the embedding automatically
    return client.get_or_create_collection(
        name=collection_name, 
        embedding_function=embedding_fn
    )

async def extract_text_async(file_bytes: bytes):
    """Offload heavy PDF parsing to a separate thread."""
    def _parse():
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        texts = [page.get_text("text") for page in doc if page.get_text("text").strip()]
        doc.close()
        return texts
    return await asyncio.to_thread(_parse)

async def ingest_pdf_bytes(file_bytes: bytes, filename: str):
    # 1. Faster parallel extraction
    pages = await extract_text_async(file_bytes)
    if not pages: return {"status": "empty"}

    # 2. Optimized chunking
    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150)
    chunks = splitter.split_text("\n\n".join(pages))

    # 3. Automatic collection handling (Low Latency Add)
    collection = get_or_create_collection()
    
    ids = [f"{filename}_{i}" for i in range(len(chunks))]
    metadatas = [{"source": filename} for _ in chunks]

    # No need to manually 'embed_documents'—Chroma does it via embedding_fn
    collection.add(ids=ids, documents=chunks, metadatas=metadatas)

    return {"pages": len(pages), "chunks": len(chunks)}