import json
import time
from typing import List, Literal, AsyncGenerator
from functools import wraps
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI
import anthropic
from .config import OPENAI_API_KEY, ANTHROPIC_API_KEY
from .ingest import get_or_create_collection, list_uploaded_sources
from .ragas_eval import log_rag_interaction
from .hybrid_search import hybrid_retrieve_chunks
from .types import (
    RetrievedChunk,
    ChatHistoryItem,
    ChatRequest
)
# ✅ LangSmith Tracing Setup
try:
    from langsmith import traceable, Client
    from langsmith.run_helpers import get_current_run_tree, tracing_context
    from langsmith.wrappers import wrap_openai
    from .config import LANGCHAIN_API_KEY

    LANGSMITH_ENABLED = bool(LANGCHAIN_API_KEY)
    if LANGSMITH_ENABLED:
        print("✅ LangSmith tracing enabled with token tracking")
        langsmith_client = Client()
except ImportError:
    LANGSMITH_ENABLED = False
    print("⚠️  langsmith not installed - tracing disabled")
    langsmith_client = None
    # Dummy decorator when LangSmith is not available
    def traceable(func=None, **kwargs):
        def decorator(f):
            return f
        return decorator(func) if func else decorator
    
# Retrieval + Prompt Building
@traceable(name="retrieve_chunks", run_type="retriever")
async def retrieve_chunks(
    query: str, 
    k: int = 4,
    filter_images_only: bool = False  #  NEW: Filter for standalone images
) -> List[RetrievedChunk]:
    """
    Sync Chroma query wrapped in async function for consistency.
    
    Args:
        query: Search query
        k: Number of chunks to retrieve
        filter_images_only: If True, only return chunks from standalone image files (type='image')
                           Excludes PDF pages with vision summaries (type='multimodal')
    """
    start_time = time.time()
    collection = get_or_create_collection()
    
    # Add metadata filter if needed
    where_filter = None
    if filter_images_only:
        where_filter = {"type": "image"}  # Only standalone uploaded images
    
    results = collection.query(
        query_texts=[query], 
        n_results=k,
        where=where_filter  # Apply filter here
    )
    
    chunks: List[RetrievedChunk] = []
    for doc_id, doc, meta, dist in zip(
        results["ids"][0],
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
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
    # Log metadata to LangSmith
    if LANGSMITH_ENABLED:
        try:
            run_tree = get_current_run_tree()
            if run_tree and chunks:
                run_tree.extra = {
                    "query": query[:200],
                    "chunk_count": len(chunks),
                    "latency_ms": round((time.time() - start_time) * 1000, 2),
                    "sources": list(set([c.source for c in chunks])),
                    "filter_images_only": filter_images_only
                }
        except:
            pass

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

# Map frontend model names to actual API model names
MODEL_MAP = {
    "gpt4o-mini": ("gpt-4o-mini", "openai"),
    "gpt-4.1": ("gpt-4.1", "openai"),
    "claude-haiku": ("claude-haiku-4-5-20251001", "claude"),    
    "claude-sonnet": ("claude-sonnet-4-5-20250929", "claude"),  
}

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

### CRITICAL GROUNDING RULES:

⚠️ **ABSOLUTE REQUIREMENTS - VIOLATING THESE IS A CRITICAL ERROR:**

1. **ZERO INFERENCE**: Do NOT infer, assume, or elaborate beyond what is EXPLICITLY stated in the snippets.
   - ❌ BAD: "These involve activities where banks act as intermediaries, accepting deposits and providing loans"
   - ✅ GOOD: "These involve activities where banks act as intermediaries (as mentioned in the table of contents)"

2. **ADMIT LIMITATIONS**: If snippets contain only section titles, headings, or incomplete information:
   - State clearly: "The provided documents only show [table of contents/section headings/etc.], not the detailed content."
   - Suggest: "I'd recommend reviewing pages X-Y directly for the full information."

3. **NO WORLD KNOWLEDGE**: Do NOT add facts from your training data, even if they're common knowledge.
   - Only use information present in the snippets below.

4. **SNIPPET TRANSPARENCY**: If you see incomplete information (e.g., chapter titles without content), acknowledge this limitation explicitly.

### GROUNDING & CITATIONS:

1. Grounding: Base your entire response solely on the information inside the snippets below.

2. Strict Refusal: If the snippets truly do NOT contain enough information to answer, say:
   "I am sorry, but the provided documents do not contain enough information to answer this question fully. The available snippets only show [what you have], but not [what's needed]."

3. Synthesis: When answering broad or summary-style questions, combine information from ALL
   relevant snippets to describe the main ideas, even if each snippet is partial.

4. Irrelevance Filter: Ignore any information that is not directly related to the question.

5. No Hallucination: Do not invent facts that are not supported by the snippets.

### FORMATTING GUIDELINES:

1. **Structure**: Use clear sections with bold headings when covering multiple topics
   
2. **Lists**: Use bullet points (-) or numbered lists for multiple items
   
3. **Bold text**: Use **bold** for document/file names, section titles, key figures

SNIPPETS:
{context_str}

USER QUESTION:
{query}

Answer concisely with proper formatting. If the snippets are incomplete, acknowledge this limitation:""".strip()
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

# ✅ Token tracking helper
def count_tokens_approximate(text: str) -> int:
    """Approximate token count (1 token ≈ 4 chars for English)."""
    return len(text) // 4

# STREAMING LLM CALLS (for /query/stream)

CHEAP_CHAT_MODEL = "gpt-4o-mini" 

async def stream_chat_openai(
    prompt: str, 
    model: str = "gpt-4o-mini",
    parent_run_id: str = None  # ✅ NEW: For nesting
) -> AsyncGenerator[str, None]:
    """Stream response from OpenAI with token tracking."""
    
    # ✅ Wrap client for auto-tracing
    if LANGSMITH_ENABLED:
        client = wrap_openai(AsyncOpenAI(api_key=OPENAI_API_KEY))
    else:
        client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    
    start_time = time.time()
    first_token_time = None
    response_text = ""  # ✅ Track full response

    stream = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=700,
        temperature=0.2,
        stream=True,
    )

    first_token = True
    async for chunk in stream:
        token = chunk.choices[0].delta.content or ""
        if not token:
            continue
        
        if first_token:
            first_token_time = time.time() - start_time
            print(f"--- [TTFT] Chat OpenAI ({model}): {first_token_time:.2f}s ---")
            first_token = False
        
        response_text += token  # ✅ Collect response
        yield f"data: {json.dumps({'type': 'token', 'token': token})}\n\n"

    total_time = time.time() - start_time
    print(f"--- [LATENCY] Chat OpenAI ({model}) Total: {total_time:.2f}s ---")
    
    # ✅ Log to LangSmith with token counts
    if LANGSMITH_ENABLED and langsmith_client:
        try:
            input_tokens = count_tokens_approximate(prompt)
            output_tokens = count_tokens_approximate(response_text)
            
            langsmith_client.create_run(
                name=f"chat_openai_{model}",
                run_type="llm",
                inputs={"prompt": prompt[:500]},
                outputs={"response": response_text[:500]},
                start_time=start_time,
                end_time=time.time(),
                extra={
                    "model": model,
                    "provider": "openai",
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                    "ttft_ms": round(first_token_time * 1000, 2) if first_token_time else None,
                    "latency_ms": round(total_time * 1000, 2),
                },
                parent_run_id=parent_run_id  # ✅ Nest under parent
            )
        except Exception as e:
            print(f"[LANGSMITH] Failed to log OpenAI run: {e}")

async def stream_chat_claude(
    prompt: str, 
    model: str = "claude-haiku-4-5-20251001",
    parent_run_id: str = None  # ✅ NEW: For nesting
) -> AsyncGenerator[str, None]:
    """Stream response from Claude with token tracking."""
    
    client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
    start_time = time.time()
    first_token_time = None
    response_text = ""  # ✅ Track full response
    first_token = True

    async with client.messages.stream(
        model=model,
        max_tokens=700,
        temperature=0.2,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        async for event in stream:
            if event.type == "content_block_delta" and event.delta.type == "text_delta":
                token = event.delta.text
                if not token:
                    continue
                
                if first_token:
                    first_token_time = time.time() - start_time
                    print(f"--- [TTFT] Chat Claude ({model}): {first_token_time:.2f}s ---")
                    first_token = False
                
                response_text += token  # ✅ Collect response
                yield f"data: {json.dumps({'type': 'token', 'token': token})}\n\n"

    total_time = time.time() - start_time
    print(f"--- [LATENCY] Chat Claude ({model}) Total: {total_time:.2f}s ---")
    
    total_time = time.time() - start_time
    print(f"--- [LATENCY] Chat Claude ({model}) Total: {total_time:.2f}s ---")
    
    # ✅ Log to LangSmith with METADATA (not just extra)
    if LANGSMITH_ENABLED and langsmith_client:
        try:
            input_tokens = count_tokens_approximate(prompt)
            output_tokens = count_tokens_approximate(response_text)
            
            langsmith_client.create_run(
                name=f"chat_claude_{model}",
                run_type="llm",
                inputs={"prompt": prompt[:500]},
                outputs={"response": response_text[:500]},
                start_time=start_time,
                end_time=time.time(),
                extra={
                    "model": model,
                    "provider": "anthropic",
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                },
                tags=["claude", model, "streaming"],  
                metadata={  
                    "ttft_seconds": round(first_token_time, 3) if first_token_time else None,
                    "latency_seconds": round(total_time, 3),
                    "ttft_ms": round(first_token_time * 1000, 2) if first_token_time else None,
                    "latency_ms": round(total_time * 1000, 2),
                },
                parent_run_id=parent_run_id
            )
        except Exception as e:
            print(f"[LANGSMITH] Failed to log Claude run: {e}")
            
@traceable(name="rag_pipeline", run_type="chain")
async def stream_chat_answer(
    query: str,
    model: Literal["gpt4o-mini", "gpt-4.1", "claude-haiku", "claude-sonnet"] = "gpt4o-mini",
    use_context: bool = True,
    history: list[ChatHistoryItem] | None = None,
) -> AsyncGenerator[str, None]:
    if history is None:
        history = []
    chunks: List[RetrievedChunk] | None = None
    
    parent_run_id = None
    if LANGSMITH_ENABLED:
        try:
            run_tree = get_current_run_tree()
            if run_tree:
                parent_run_id = str(run_tree.id)
        except:
            pass
        
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

    # ----- 2) Retrieval logic with smart filtering -----
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

    # NEW: Detect standalone image questions (user uploaded .jpg/.png)
    standalone_image_triggers = [
        "what's in the image",
        "what is in the image",
        "whats in the image",
        "describe the image",
        "tell me about the image",
        "what does the image show",
        "analyze the image",
        "explain the image",
    ]

    is_standalone_image_question = any(t in lower_q for t in standalone_image_triggers)

    # NEW: Questions about all documents (include PDFs with images)
    all_docs_triggers = [
        "what documents do i have",
        "list all documents",
        "list documents",
        "show all files",
        "what files are uploaded",
        "what's uploaded",
        "show me everything",
    ]

    is_all_docs_question = any(t in lower_q for t in all_docs_triggers)

    if use_context and not is_smalltalk:
        if is_summary:
            chunks = await hybrid_retrieve_chunks(query, k=20)  
        elif is_standalone_image_question:
            chunks = await retrieve_chunks(query, k=10, filter_images_only=True)  
        elif is_all_docs_question:
            chunks = await hybrid_retrieve_chunks(query, k=15)  
        else:
            chunks = await hybrid_retrieve_chunks(query, k=12)  
    else:
        chunks = []

    # ----- 3) Build prompt + send meta with chunks -----
    # Map to actual model names
    retrieval_start = time.time()

    # Yield retrieval start event
    
    # Calculate retrieval time (retrieval already happened above in your code)
    retrieval_time = time.time() - retrieval_start if use_context else 0

    # Map to actual model names
    actual_model, provider = MODEL_MAP[model]

    # Build prompt
    prompt = build_chat_prompt(query, chunks, use_context, history or [])

    # Enhanced metadata with retrieval info
    meta_payload = {
        "type": "meta",
        "data": {
            "model": actual_model,
            "provider": provider,
            "timestamp": time.time(),
            "chunks_used": len(chunks) if chunks else 0,
            "retrieval_time_ms": round(retrieval_time * 1000, 2),
            "context_sources": list(set([c.source for c in chunks])) if chunks else [],
            "query_type": "summary" if is_summary else "image" if is_standalone_image_question else "standard",
        },
        "chunks": [
            {
                "source": c.source,
                "page": c.page,
                "score": c.score,
                "text": c.text,
                "type": c.type
            } for c in chunks
        ] if chunks else []
    }
    # Send full metadata with chunks
    yield f"data: {json.dumps(meta_payload)}\n\n"
    # ✅ Log pipeline metadata to LangSmith
    if LANGSMITH_ENABLED:
        try:
            run_tree = get_current_run_tree()
            if run_tree:
                run_tree.extra = {
                    "query": query[:200],
                    "model": actual_model,
                    "provider": provider,
                    "use_context": use_context,
                    "chunk_count": len(chunks) if chunks else 0,
                    "sources": list(set([c.source for c in chunks])) if chunks else [],
                    "retrieval_time_ms": round(retrieval_time * 1000, 2),
                    "query_type": "summary" if is_summary else "image" if is_standalone_image_question else "standard",
                }
        except:
            pass
   # Route to correct provider
    generation_start = time.time()
    answer_text = ""

    if provider == "claude":
        async for chunk in stream_chat_claude(prompt, model=actual_model, parent_run_id=parent_run_id):  # ✅ Added
            yield chunk
            
            if '"type":"token"' in chunk or '"type": "token"' in chunk:
                try:
                    data = json.loads(chunk.replace("data: ", "").strip())
                    if data.get("type") == "token":
                        answer_text += data.get("token", "")
                except:
                    pass
    else:  # openai
        async for chunk in stream_chat_openai(prompt, model=actual_model, parent_run_id=parent_run_id):  # ✅ Added
            yield chunk
            
            if '"type":"token"' in chunk or '"type": "token"' in chunk:
                try:
                    data = json.loads(chunk.replace("data: ", "").strip())
                    if data.get("type") == "token":
                        answer_text += data.get("token", "")
                except:
                    pass

    # Calculate total latency
    generation_time = time.time() - generation_start
    total_latency = time.time() - retrieval_start
    
    # Log interaction for RAGAS
    log_rag_interaction(
        question=query,
        answer=answer_text,
        contexts=[c.text for c in chunks],
        model=actual_model  # Track which model answered
    )

    yield "data: [DONE]\n\n"
    
