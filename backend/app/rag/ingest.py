import asyncio
import fitz
import chromadb
from typing import List, Dict
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from .vision_ingest import page_needs_vision, summarize_image_with_vision
from .config import OPENAI_API_KEY, CHROMA_DB_DIR, FINANCE_CLASSIFIER_MODEL
from openai import AsyncOpenAI
from fastapi import HTTPException
import json

UPLOADED_SOURCES: set[str] = set()
def list_uploaded_sources() -> list[str]:
    """Return sorted list of distinct source filenames that have been ingested."""
    return sorted(UPLOADED_SOURCES)

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
async def classify_finance_document(text: str) -> dict:
    """
    Fast rule-based classifier for finance documents.
    More reliable and faster than LLM-based classification.
    """
    
    if not text or len(text.strip()) < 20:
        return {
            "label": "NOT_FINANCE",
            "confidence": 0.95,
            "reason": "Document text too short"
        }
    
    text_lower = text.lower()
    
    # ❌ REJECT: Clear non-finance indicators (high confidence)
    reject_patterns = {
        "resume": ["curriculum vitae", "resume", "work experience", "education:", "skills:", "references available"],
        "job": ["job description", "responsibilities:", "qualifications:", "applying for"],
        "ml_research": ["arxiv", "abstract", "neural network", "deep learning", "transformer", "llm", "large language model"]
    }
    
    for category, patterns in reject_patterns.items():
        matches = sum(1 for p in patterns if p in text_lower)
        if matches >= 2:  # At least 2 patterns = confident reject
            return {
                "label": "NOT_FINANCE",
                "confidence": 0.95,
                "reason": f"Detected {category} document"
            }
    
    # ✅ ACCEPT: Finance keyword detection
    finance_keywords = {
        "transaction": ["invoice", "receipt", "payment", "transaction", "purchase"],
        "banking": ["bank", "account", "deposit", "withdrawal", "statement"],
        "money": ["money", "currency", "dollar", "euro", "pound", "yen"],
        "investment": ["investment", "stock", "bond", "equity", "portfolio", "trading"],
        "corporate": ["balance sheet", "income statement", "cash flow", "financial statement"],
        "accounting": ["accounting", "bookkeeping", "ledger", "debit", "credit", "journal"],
        "finance_theory": ["finance", "financial", "economics", "monetary", "fiscal"],
        "tax": ["tax", "taxation", "audit", "irs", "revenue"],
        "ops": ["payroll", "expense", "revenue", "profit", "loss"],
        "credit": ["loan", "mortgage", "interest rate", "credit", "debt"],
        "insurance": ["insurance", "premium", "policy", "coverage"]
    }
    
    total_keywords = 0
    categories_found = []
    
    for category, keywords in finance_keywords.items():
        count = sum(1 for kw in keywords if kw in text_lower)
        if count > 0:
            total_keywords += count
            categories_found.append(category)
    
    # Decision logic
    if total_keywords >= 3:  # Strong finance signal
        return {
            "label": "FINANCE",
            "confidence": 0.95,
            "reason": f"Found {total_keywords} finance terms across {len(categories_found)} categories"
        }
    elif total_keywords >= 1:  # Weak finance signal
        return {
            "label": "FINANCE",
            "confidence": 0.70,
            "reason": f"Found {total_keywords} finance terms - accepting with lower confidence"
        }
    else:  # No finance keywords, but no reject patterns either
        return {
            "label": "FINANCE",
            "confidence": 0.60,
            "reason": "No clear rejection signals - accepting by default"
        }

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
    Page-wise ingest with Conditional Vision
    - Always index text.
    - Only call Claude vision on pages that need it.
    """
    
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    
    # Extract first page for classification
    first_page_text = ""
    try:
        if len(doc) > 0:
            first_page_text = doc[0].get_text("text") or ""
    except Exception as e:
        print(f"--- [ERROR] Failed to extract first page: {e}")
        first_page_text = ""
    
    # ✅ Classify with rule-based classifier
    classification = await classify_finance_document(first_page_text)
    
    print(f"--- [CLASSIFIER] Label: {classification['label']}, Confidence: {classification.get('confidence', 0):.2f}, Reason: {classification.get('reason', 'NA')}")
    
    # Reject if NOT_FINANCE with high confidence
    if classification["label"] == "NOT_FINANCE" and classification.get("confidence", 0) >= 0.85:
        doc.close()
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Only financial documents are supported",
                "reason": classification.get("reason"),
                "confidence": classification.get("confidence"),
            },
        )
    
    # 5) Continue with ingestion
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
                "finance_label": classification.get("label"),
                "finance_confidence": classification.get("confidence"),
            }
        )
    UPLOADED_SOURCES.add(filename)
    
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

def delete_document_by_source(filename: str) -> dict:
    collection = get_or_create_collection()
    # delete by where clause on metadata
    collection.delete(where={"source": filename})
    # also drop from the in-memory registry if you're using it
    try:
        UPLOADED_SOURCES.remove(filename)
    except KeyError:
        pass
    return {"status": "ok", "deleted_source": filename}

def refresh_uploaded_sources_from_chroma():
    """
    Scan the persistent Chroma collection and rebuild UPLOADED_SOURCES
    from distinct 'source' metadata values.
    """
    global UPLOADED_SOURCES
    collection = get_or_create_collection()

    results = collection.get(include=["metadatas"])
    raw_metadatas = results.get("metadatas", [])

    sources: set[str] = set()

    for meta in raw_metadatas:
        # meta can be a dict or a string, depending on how Chroma stored it
        if isinstance(meta, dict):
            src = meta.get("source")
            if src:
                sources.add(src)
        elif isinstance(meta, str):
            # older entries might just store the filename as a string
            sources.add(meta)

    UPLOADED_SOURCES = sources

