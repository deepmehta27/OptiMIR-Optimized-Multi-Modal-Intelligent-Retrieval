import asyncio
import fitz
import chromadb
from typing import List, Dict
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from .vision_ingest import page_needs_vision, summarize_image_with_vision
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
    """
    Page-wise ingest with Conditional Vision:
    - Always index text.
    - Only call Claude vision on pages that need it.
    """
    doc = fitz.open(stream=file_bytes, filetype="pdf")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150,
    )

    page_infos: List[Dict] = []

    # 1) Collect page info + schedule vision where needed
    vision_tasks = []
    for page_index, page in enumerate(doc):
        page_num = page_index + 1
        text_content = (page.get_text("text") or "").strip()

        needs_vision = page_needs_vision(page)
        print(
            f"--- [INGEST] Page {page_num}: "
            + ("Vision Required" if needs_vision else "Text Only (skip vision)")
        )

        entry: Dict = {
            "page_index": page_index,
            "page_num": page_num,
            "text": text_content,
            "needs_vision": needs_vision,
        }
        page_infos.append(entry)

        if needs_vision:
            pix = page.get_pixmap()
            img_bytes = pix.tobytes("png")
            vision_tasks.append(
                summarize_image_with_vision(img_bytes, page_num)
            )

    # 2) Run all vision calls concurrently
    vision_results: List[Dict] = []
    if vision_tasks:
        vision_results = await asyncio.gather(*vision_tasks)

    vision_by_page = {res["page"]: res for res in vision_results}

    # 3) Build final text for each page (super chunks) and chunk them
    all_documents: List[str] = []
    all_metadata: List[Dict] = []

    for entry in page_infos:
        page_num = entry["page_num"]
        text_content = entry["text"]
        needs_vision = entry["needs_vision"]

        if not text_content and not needs_vision:
            continue

        if needs_vision:
            v = vision_by_page.get(page_num)
            if v:
                rich = (
                    f"TEXT CONTENT (page {page_num}):\n{text_content}\n\n"
                    f"VISION SUMMARY:\n{v.get('summary','')}"
                )
                table_md = v.get("table_markdown")
                if table_md:
                    rich += f"\n\nRECONSTRUCTED TABLE:\n{table_md}"
                doc_text = rich
                meta_type = "multimodal"
            else:
                doc_text = text_content
                meta_type = "text"
        else:
            doc_text = text_content
            meta_type = "text"

        if not doc_text.strip():
            continue

        chunks = splitter.split_text(doc_text)
        for idx, ch in enumerate(chunks):
            all_documents.append(ch)
            all_metadata.append(
                {
                    "source": filename,
                    "page": page_num,
                    "type": meta_type,
                    "chunk_index": idx,
                }
            )

    if not all_documents:
        return {"status": "empty", "pages": len(doc), "chunks": 0}

    # 4) Add to Chroma with automatic embeddings
    collection = get_or_create_collection()
    ids = [
        f"{filename}_p{m['page']}_c{m['chunk_index']}" for m in all_metadata
    ]

    collection.add(
        ids=ids,
        documents=all_documents,
        metadatas=all_metadata,
    )

    return {"pages": len(doc), "chunks": len(all_documents)}