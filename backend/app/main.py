from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from .rag.ingest import ingest_pdf_bytes, delete_document_by_source, list_uploaded_sources, refresh_uploaded_sources_from_chroma
from .rag.retrieval import rag_answer, stream_rag_answer,RAGResponse, QueryRequest,ChatRequest, stream_chat_answer, run_ragas_eval
from pydantic import BaseModel

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
        ):
            yield chunk

    return StreamingResponse(event_gen(), media_type="text/event-stream")

@app.post("/query", response_model=RAGResponse)
async def query_rag(payload: QueryRequest):
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    # FIX: Must await rag_answer
    response = await rag_answer(payload.question, model=payload.model)
    return response

@app.post("/query/stream")
async def query_rag_stream(payload: QueryRequest):
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    # The event_generator simply wraps our retrieval logic
    async def event_generator():
        async for chunk in stream_rag_answer(
            query=payload.question,
            model=payload.model,
        ):
            yield chunk

    # Return with the specific SSE media type
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
    )
@app.get("/pdfs")
async def list_pdfs():
    return {"sources": list_uploaded_sources()}

@app.post("/delete_pdf")
async def delete_pdf(req: DeletePdfRequest):
    if not req.source.strip():
        raise HTTPException(status_code=400, detail="source cannot be empty")

    result = delete_document_by_source(req.source)
    return result

@app.post("/evaluate/ragas")
async def evaluate_ragas(payload: EvalRequest):
    result = await run_ragas_eval(limit=payload.limit)
    return result

@app.on_event("startup")
async def startup_event():
    # rebuild the in-memory source list from Chroma at startup
    refresh_uploaded_sources_from_chroma()
