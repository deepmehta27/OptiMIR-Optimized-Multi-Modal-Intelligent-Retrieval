import json
import time
from typing import List, Literal, AsyncGenerator
from functools import partial
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI
import anthropic
from .config import OPENAI_API_KEY, ANTHROPIC_API_KEY
from .ingest import get_or_create_collection
from .ingest import get_or_create_collection, list_uploaded_sources
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import Faithfulness
import asyncio

RAG_LOG: list[dict] = []

# --- Schemas ---
class QueryRequest(BaseModel):
    question: str
    model: Literal["gpt4o", "claude"] = "gpt4o"
    use_context: bool = True

class ChatRequest(BaseModel):
    question: str
    model: Literal["gpt4o", "claude"] = "gpt4o"
    use_context: bool = True
    
class RetrievedChunk(BaseModel):
    text: str
    source: str
    score: float
    type: str | None = None
    page: int | None = None

class RAGResponse(BaseModel):
    answer: str
    model: str
    chunks: List[RetrievedChunk]

class ChatHistoryItem(BaseModel):
    role: Literal["user", "assistant"]
    text: str

class ChatRequest(BaseModel):
    question: str
    model: Literal["gpt4o", "claude"] = "gpt4o"
    use_context: bool = True
    history: list[ChatHistoryItem] | None = None
    
# Retrieval + Prompt

async def retrieve_chunks(query: str, k: int = 4) -> List[RetrievedChunk]:
    """Sync Chroma query wrapped in async function for consistency."""
    collection = get_or_create_collection()
    results = collection.query(query_texts=[query], n_results=k)
    chunks: List[RetrievedChunk] = []
    for doc_id, doc, meta, dist in zip(results["ids"][0],results["documents"][0],results["metadatas"][0],results["distances"][0],
    ):
        chunks.append(
            RetrievedChunk(
                text=doc,
                source=meta.get("source", doc_id),
                score=float(dist),
                type=meta.get("type"),
                page=meta.get("page"),
            )
        )
    return chunks

def build_prompt(query: str, chunks: List[RetrievedChunk]) -> str:
    # Format chunks with clear ID and Source labels
    context_entries = []
    for i, c in enumerate(chunks):
        context_entries.append(f"<document id='{i+1}'>\nSource: {c.source}\nContent: {c.text}\n</document>")
    context_str = "\n".join(context_entries)

    return f"""
You are OptiMIR - Optimized Multi‑Modal Intelligent Retrieval, a highly precise document QA assistant designed to help users explore and understand their workspace documents.

Additional rules for mathematics:
- Write formulas cleanly using either plain text (e.g., "Var = E[X^2] - (E[X])^2")
  or simple LaTeX when it improves clarity.
- Avoid producing long or complex LaTeX blocks that might render poorly.
- Keep equations on their own lines when possible for readability.
- If the PDF text shows broken math, infer the intended expression from context and fix spacing and subscripts/superscripts.

### INSTRUCTIONS:
1. **Grounding:** Base your entire response solely on the information inside the tags.
2. **Strict Refusal:** If the context does NOT contain the answer, state: "I am sorry, but the provided documents do not contain information to answer this question." Do NOT use your own knowledge.
3. **Citations:** Every factual claim must be followed by a citation in brackets, e.g., [1] or [1, 3].
4. **Irrelevance Filter:** Ignore any information in the context that is not directly related to the question.
5. **No Hallucination:** Do not infer new facts not supported by the documents.

<context>
{context_str}
</context>

<question>
{query}
</question>

Answer:""".strip()

async def stream_chat_chat_endpoint(req: ChatRequest):
    return StreamingResponse(
        stream_chat_answer(
            query=req.question,
            model=req.model,
            use_context=req.use_context,
            history=req.history or [],
        ),
        media_type="text/event-stream",
    )

def build_chat_prompt(
    query: str,
    chunks: List[RetrievedChunk] | None,
    use_context: bool,
    history: list[ChatHistoryItem],
) -> str:
    history_str = ""
    if history:
        lines = []
        for h in history:
            prefix = "User:" if h.role == "user" else "Assistant:"
            lines.append(f"{prefix} {h.text}")
        history_str = "\n\nPrevious conversation:\n" + "\n".join(lines)
    if use_context and chunks:
        context_lines = []
        for i, c in enumerate(chunks):
            context_lines.append(
                f"[Snippet {i+1} | source={c.source} | page={c.page} | type={c.type}]\n"
                f"{c.text}\n"
            )
        context_str = "\n\n".join(context_lines)

        return f"""
You are OptiMIR - Optimized Multi‑Modal Intelligent Retrieval, a highly precise document QA assistant designed to help users explore and understand their workspace documents.

### CORE RULES:
1. Identity: If asked "who are you," identify as OptiMIR. Do not claim to be the author of the documents or the user.
2. Scope: Only answer questions about the uploaded documents or closely related professional topics. 
3. Strict Refusal: Politely refuse questions about unrelated topics (e.g., cooking, movies, music, or politics). State that the system is focused solely on the documents in this workspace.
4. Conversational Tone: Respond naturally and briefly to greetings or light small-talk (e.g., "hi," "how are you").
5. Follow-ups: Use the previous conversation to resolve pronouns like "this", "that", or "it". If the user says "tell me more about that", continue the topic from the last assistant answer, do not switch to a different document.

{history_str}

### GROUNDING & CITATIONS:
1. Grounding: Base your entire response solely on the information inside the snippets below.
2. Strict Refusal: If the snippets truly do NOT contain enough information to answer, say:
   "I am sorry, but the provided documents do not contain information to answer this question."
   Do NOT use outside knowledge.
3. Synthesis: When answering broad or summary-style questions, combine information from ALL
   relevant snippets to describe the main ideas, even if each snippet is partial.
4. Irrelevance Filter: Ignore any information that is not directly related to the question.
5. No Hallucination: Do not invent facts that are not supported by the snippets.

SNIPPETS:
{context_str}

USER QUESTION:
{query}

Answer concisely:""".strip()

    # No context → pure but still narrow chat
    return f"""
You are OptiMIR - Optimized Multi‑Modal Intelligent Retrieval, a highly precise document QA assistant designed to help users explore and understand their workspace documents.

{history_str}

You may answer normal greetings and simple questions, but if the user asks about topics
completely unrelated to this workspace, politely say that this system is focused on
the uploaded documents and light conversation around them.

USER QUESTION:
{query}

Answer concisely:""".strip()

# NON-STREAMING LLM CALLS (for /query)

async def answer_with_openai(prompt: str) -> str:
    start_time = time.time()
    client = AsyncOpenAI(api_key=OPENAI_API_KEY)

    resp = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024,
        temperature=0.1,
    )

    print(f"--- [LATENCY] GPT-4o-mini: {time.time() - start_time:.2f}s ---")
    return resp.choices[0].message.content or ""

async def answer_with_claude(prompt: str) -> str:
    start_time = time.time()
    client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)

    resp = await client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=1024,
        temperature=0.1,
        messages=[{"role": "user", "content": prompt}],
    )

    print(f"--- [LATENCY] Claude-Haiku-4.5: {time.time() - start_time:.2f}s ---")
    return "".join(block.text for block in resp.content if block.type == "text")

async def rag_answer(
    query: str,
    model: Literal["gpt4o", "claude"] = "gpt4o",
) -> RAGResponse:
    chunks = await retrieve_chunks(query)
    prompt = build_prompt(query, chunks)

    print(f"\n--- [LOG] Routing to: {model.upper()} ---")

    if model == "claude":
        answer = await answer_with_claude(prompt)
        model_name = "claude-haiku-4.5"
    else:
        answer = await answer_with_openai(prompt)
        model_name = "gpt-4o-mini"

    print(f"--- [LOG] Answer generated by: {model_name} ---\n")
    # minimal Ragas trace
    RAG_LOG.append(
        {
            "question": query,
            "answer": answer,
            "contexts": [c.text for c in chunks],
        }
    )

    return RAGResponse(answer=answer, model=model_name, chunks=chunks)

async def run_ragas_eval(limit: int | None = 50) -> dict:
    samples = RAG_LOG[-limit:] if (limit and len(RAG_LOG) > limit) else RAG_LOG
    if not samples:
        return {"status": "empty", "count": 0}

    data = {
        "question": [s["question"] for s in samples],
        "answer": [s["answer"] for s in samples],
        "contexts": [s["contexts"] for s in samples],
    }

    ds = Dataset.from_dict(data)

    # IMPORTANT: run RAGAS in a separate thread
    loop = asyncio.get_running_loop()
    eval_fn = partial(evaluate, ds, metrics=[Faithfulness()])
    result = await loop.run_in_executor(None, eval_fn)

    scores = result.scores or []
    faithfulness_score = scores[0] if scores else None

    return {
        "status": "ok",
        "count": len(samples),
        "scores": {
            "faithfulness": faithfulness_score,
        },
    }

# STREAMING LLM CALLS (for /query/stream)

CHEAP_CHAT_MODEL = "gpt-4.1-nano"  # or "gpt-5-nano"

async def stream_chat_openai(prompt: str) -> AsyncGenerator[str, None]:
    client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    start_time = time.time()

    stream = await client.chat.completions.create(
        model=CHEAP_CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
        temperature=0.5,
        stream=True,
    )

    first_token = True
    async for chunk in stream:
        token = chunk.choices[0].delta.content or ""
        if not token:
            continue
        if first_token:
            print(f"--- [TTFT] Chat OpenAI: {time.time() - start_time:.2f}s ---")
            first_token = False
        yield f"data: {json.dumps({'type': 'token', 'token': token})}\n\n"

    print(f"--- [LATENCY] Chat OpenAI Total: {time.time() - start_time:.2f}s ---")


async def stream_chat_claude(prompt: str) -> AsyncGenerator[str, None]:
    client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
    start_time = time.time()
    first_token = True

    async with client.messages.stream(
        model="claude-haiku-4-5",
        max_tokens=512,
        temperature=0.4,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        async for event in stream:
            if event.type == "content_block_delta" and event.delta.type == "text_delta":
                token = event.delta.text
                if not token:
                    continue
                if first_token:
                    print(f"--- [TTFT] Chat Claude: {time.time() - start_time:.2f}s ---")
                    first_token = False
                yield f"data: {json.dumps({'type': 'token', 'token': token})}\n\n"

    print(f"--- [LATENCY] Chat Claude Total: {time.time() - start_time:.2f}s ---")

async def stream_chat_answer(
    query: str,
    model: Literal["gpt4o", "claude"] = "gpt4o",
    use_context: bool = True,
    history: list[ChatHistoryItem] | None = None,
) -> AsyncGenerator[str, None]:
    if history is None:
        history = []
    chunks: List[RetrievedChunk] | None = None

    lower_q = query.lower()

    smalltalk_triggers = ["hi", "hello", "hey", "how are you", "good morning", "good evening"]
    is_smalltalk = any(f" {t} " in f" {lower_q} " for t in smalltalk_triggers)

    off_scope_triggers = [
        "recipe", "cook", "cooking", "pasta", "food",
        "movie", "music", "song", "lyrics",
        "politics", "election",
    ]
    is_off_scope = any(t in lower_q for t in off_scope_triggers)

    # ----- 0) Hard block off-scope -----
    if is_off_scope:
        meta = {
            "type": "meta",
            "mode": "chat",
            "model": "rule-based",
            "use_context": False,
            "is_smalltalk": False,
            "is_off_scope": True,
            "chunks": [],
        }
        yield f"data: {json.dumps(meta)}\n\n"
        msg = (
            "I am focused on helping with your uploaded documents and related questions, "
            "not general topics like recipes or entertainment."
        )
        for tok in msg.split(" "):
            yield f"data: {json.dumps({'type': 'token', 'token': tok + ' '})}\n\n"
        yield "data: [DONE]\n\n"
        return

    # ----- 1) Multi-doc disambiguation -----
    sources = list_uploaded_sources()
    multiple_docs = len(sources) > 1

    generic_doc_triggers = [
        "what is this pdf about",
        "what's the pdf about",
        "what is the pdf about",
        "summarize the pdf",
        "summarize this pdf",
        "summarize the document",
        "summarize this document",
        "summarize for me",
        "give me a summary",
        "summary of the pdf",
    ]
    is_generic_doc_question = any(t in lower_q for t in generic_doc_triggers)

    if multiple_docs and is_generic_doc_question:
        meta = {
            "type": "meta",
            "mode": "chat",
            "model": "router",
            "use_context": False,
            "is_smalltalk": False,
            "is_off_scope": False,
            "needs_disambiguation": True,
            "options": sources,
            "chunks": [],
        }
        yield f"data: {json.dumps(meta)}\n\n"

        msg = (
            "You have multiple documents uploaded:\n"
            + "\n".join(f"- {s}" for s in sources)
            + "\nPlease tell me which document you want me to use."
        )
        for tok in msg.split(" "):
            yield f"data: {json.dumps({'type': 'token', 'token': tok + ' '})}\n\n"
        yield "data: [DONE]\n\n"
        return

    # ----- 2) Retrieval logic (same spirit as query/stream) -----
    summary_triggers = [
        "summarize the pdf",
        "summarize this pdf",
        "summarize this document",
        "summarize the document",
        "summarize for me",
        "give me a summary",
        "high level summary",
    ]
    is_summary = any(t in lower_q for t in summary_triggers)

    if use_context and not is_smalltalk:
        if is_summary:
            chunks = await retrieve_chunks(query, k=20)
        else:
            chunks = await retrieve_chunks(query, k=12)
    else:
        chunks = []

    # ----- 3) Build prompt + send meta with chunks -----
    prompt = build_chat_prompt(query, chunks, use_context=use_context, history=history)
    meta = {
        "type": "meta",
        "mode": "chat",
        "model": "claude-haiku-4.5" if model == "claude" else CHEAP_CHAT_MODEL,
        "use_context": use_context and not is_smalltalk,
        "is_smalltalk": is_smalltalk,
        "is_off_scope": False,
        "chunks": [c.model_dump() for c in chunks] if chunks else [],
    }
    yield f"data: {json.dumps(meta)}\n\n"

    # ----- 4) Stream tokens AND buffer answer for RAGAS -----
    answer_text = ""

    if model == "claude":
        async for ev in stream_chat_claude(prompt):
            try:
                payload = json.loads(ev.replace("data:", "").strip())
                if payload.get("type") == "token":
                    answer_text += payload.get("token", "")
            except Exception:
                pass
            yield ev
    else:
        async for ev in stream_chat_openai(prompt):
            try:
                payload = json.loads(ev.replace("data:", "").strip())
                if payload.get("type") == "token":
                    answer_text += payload.get("token", "")
            except Exception:
                pass
            yield ev

    # ----- 5) Log for RAGAS (single unified stream) -----
    if chunks:
        RAG_LOG.append(
            {
                "question": query,
                "answer": answer_text,
                "contexts": [c.text for c in chunks],
            }
        )

    yield "data: [DONE]\n\n"
    
# STREAMING LLM CALLS (for /query/stream)

async def stream_openai(prompt: str) -> AsyncGenerator[str, None]:
    """Yield SSE events with tokens from GPT-4o-mini."""
    start_time = time.time()
    client = AsyncOpenAI(api_key=OPENAI_API_KEY)

    stream = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024,
        temperature=0.1,
        stream=True,
    )

    first_token = True
    full_answer: List[str] = []

    async for chunk in stream:
        delta = chunk.choices[0].delta
        token = delta.content or ""
        if not token:
            continue

        full_answer.append(token)

        if first_token:
            print(f"--- [TTFT] OpenAI: {time.time() - start_time:.2f}s ---")
            first_token = False

        yield f"data: {json.dumps({'type': 'token', 'token': token})}\n\n"

    print(f"--- [LATENCY] OpenAI Stream Total: {time.time() - start_time:.2f}s ---")
    # full answer available as "".join(full_answer) if needed


async def stream_claude(prompt: str) -> AsyncGenerator[str, None]:
    """Yield SSE events with tokens from Claude Haiku 4.5."""
    start_time = time.time()
    client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)

    first_token = True
    full_answer: List[str] = []

    async with client.messages.stream(
        model="claude-haiku-4-5",
        max_tokens=1024,
        temperature=0.1,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        async for event in stream:
            if event.type == "content_block_delta" and event.delta.type == "text_delta":
                token = event.delta.text
                if not token:
                    continue

                full_answer.append(token)

                if first_token:
                    print(f"--- [TTFT] Claude: {time.time() - start_time:.2f}s ---")
                    first_token = False

                yield f"data: {json.dumps({'type': 'token', 'token': token})}\n\n"

    print(f"--- [LATENCY] Claude Stream Total: {time.time() - start_time:.2f}s ---")
    # full answer available as "".join(full_answer) if needed
