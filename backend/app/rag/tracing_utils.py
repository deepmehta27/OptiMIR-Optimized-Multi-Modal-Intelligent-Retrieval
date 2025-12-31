"""
LangSmith tracing utilities for OptiMIR.

Provides decorators and helpers for tracing RAG pipeline components.
"""
from functools import wraps
from typing import Any, Callable
import time
from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree

from .config import LANGCHAIN_API_KEY


def trace_retrieval(func: Callable) -> Callable:
    """
    Decorator for tracing retrieval functions.

    Automatically logs:
    - Query text
    - Number of chunks retrieved
    - Retrieval latency
    - Chunk sources and scores
    """
    if not LANGCHAIN_API_KEY:
        return func

    @traceable(name=f"retrieval_{func.__name__}")
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()

        # Extract query from args
        query = args[0] if args else kwargs.get('query', 'unknown')

        # Call the actual retrieval function
        result = await func(*args, **kwargs)

        # Calculate metrics
        latency_ms = round((time.time() - start_time) * 1000, 2)
        chunk_count = len(result) if result else 0

        # Get current run for metadata
        run_tree = get_current_run_tree()
        if run_tree:
            run_tree.extra = {
                "query": query[:200],  # Truncate long queries
                "chunk_count": chunk_count,
                "latency_ms": latency_ms,
                "sources": list(set([c.source for c in result])) if result else []
            }

        return result

    return wrapper


def trace_llm_call(provider: str = "unknown"):
    """
    Decorator for tracing LLM API calls.

    Args:
        provider: "openai" or "claude"

    Logs:
    - Model name
    - Prompt (first 500 chars)
    - Response length
    - Latency and TTFT
    """
    def decorator(func: Callable) -> Callable:
        if not LANGCHAIN_API_KEY:
            return func

        @traceable(name=f"llm_{provider}_{func.__name__}")
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()

            # Extract prompt and model
            prompt = args[0] if args else kwargs.get('prompt', '')
            model = args[1] if len(args) > 1 else kwargs.get('model', 'unknown')

            # Call LLM
            result = await func(*args, **kwargs)

            # Calculate latency
            latency_ms = round((time.time() - start_time) * 1000, 2)

            # Get current run for metadata
            run_tree = get_current_run_tree()
            if run_tree:
                run_tree.extra = {
                    "provider": provider,
                    "model": model,
                    "prompt_preview": prompt[:500] if isinstance(prompt, str) else "streaming",
                    "latency_ms": latency_ms,
                }

            return result

        return wrapper
    return decorator


def trace_rag_pipeline(func: Callable) -> Callable:
    """
    Decorator for tracing the full RAG pipeline.

    Logs:
    - User query
    - Model selection
    - Whether context was used
    - Total latency
    - Query type (summary, image, standard)
    """
    if not LANGCHAIN_API_KEY:
        return func

    @traceable(name="rag_pipeline", run_type="chain")
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()

        # Extract query and model
        query = kwargs.get('query', args[0] if args else 'unknown')
        model = kwargs.get('model', 'gpt-4.1-nano')
        use_context = kwargs.get('use_context', True)

        # Call the pipeline
        result = wrapper.__wrapped__(*args, **kwargs)

        # For async generators, we'll trace the setup
        run_tree = get_current_run_tree()
        if run_tree:
            run_tree.extra = {
                "query": query[:200],
                "model": model,
                "use_context": use_context,
            }

        return result

    return wrapper