from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from .rag.ingest import (
    ingest_pdf_bytes,
    delete_document_by_source,
    list_uploaded_sources,
    refresh_uploaded_sources_from_chroma,
    get_or_create_collection,
)
from .rag.types import ( 
    ChatRequest,
)
from .rag.retrieval import (
    stream_chat_answer,
)
from .rag.ragas_eval import (
    run_ragas_eval,
    get_ragas_log_size,
    clear_ragas_log,
)
from pydantic import BaseModel
from .rag.image_ingest import process_financial_image
from .rag.image_routes import router as image_router

class DeletePdfRequest(BaseModel):
    source: str  # the filename used as "source" during ingest

class EvalRequest(BaseModel):
    limit: int | None = 50  # how many recent traces to score

app = FastAPI(title="OptiMIR Backend", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # will tighten later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(image_router)

@app.get("/")
def read_root():
    return {"status": "ok", "message": "OptiMIR backend is running"}


@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    file_bytes = await file.read()
    
    # FIX: add 'await' here because ingest_pdf_bytes is async
    result = await ingest_pdf_bytes(file_bytes, filename=file.filename)

    if result.get("status") == "rejected":
        raise HTTPException(
            status_code=400,
            detail=f"Only financial documents are supported. Reason: {result.get('reason')}",
        )

    # Now 'result' is the actual dictionary returned by your function
    return {
        "status": "success",
        "filename": file.filename,
        "pages_ingested": result["pages"],
        "chunks_created": result["chunks"],
    }
    
@app.post("/chat/stream")
async def chat_stream(payload: ChatRequest):
    async def event_gen():
        async for chunk in stream_chat_answer(
            query=payload.question,
            model=payload.model,
            use_context=payload.use_context,
            history=payload.history
        ):
            yield chunk

    return StreamingResponse(event_gen(), media_type="text/event-stream")

@app.get("/documents")
async def list_all_documents():
    """Get both PDFs and images in one unified list."""
    from .rag.ingest import get_or_create_collection
    
    # Get PDFs
    pdfs = list_uploaded_sources()
    
    # Get images from Chroma
    collection = get_or_create_collection()
    try:
        results = collection.get(include=["metadatas"])
        
        images = set()
        for meta in results.get("metadatas", []):
            if isinstance(meta, dict) and meta.get("type") == "image":
                source = meta.get("source")
                if source:
                    images.add(source)
        
        return {
            "pdfs": sorted(pdfs),
            "images": sorted(list(images)),
            "all": sorted(list(set(pdfs) | images))  # Union of both
        }
    except Exception as e:
        print(f"Error fetching documents: {e}")
        return {
            "pdfs": sorted(pdfs),
            "images": [],
            "all": sorted(pdfs)
        }

@app.get("/pdfs")
async def list_pdfs():
    return {"sources": list_uploaded_sources()}

@app.post("/delete_pdf")
async def delete_pdf(req: DeletePdfRequest):
    if not req.source.strip():
        raise HTTPException(status_code=400, detail="source cannot be empty")

    result = delete_document_by_source(req.source)
    return result

# Update the evaluation endpoint
@app.post("/evaluate/ragas")
async def evaluate_ragas(
    limit: Optional[int] = 50,
    model_filter: Optional[str] = None
):
    """Run RAGAS evaluation on logged interactions."""
    result = await run_ragas_eval(limit=limit, model_filter=model_filter)
    return result

# Add new endpoints
@app.get("/evaluate/ragas/status")
async def ragas_status():
    """Check how many interactions are logged."""
    return {
        "logged_interactions": get_ragas_log_size(),
        "ready_for_eval": get_ragas_log_size() > 0
    }

@app.post("/evaluate/ragas/clear")
async def clear_ragas():
    """Clear evaluation log."""
    clear_ragas_log()
    return {"status": "cleared"}

@app.on_event("startup")
async def startup_event():
    # rebuild the in-memory source list from Chroma at startup
    refresh_uploaded_sources_from_chroma()
