"""
Comprehensive RAGAS evaluation for OptiMIR.
"""

import asyncio
from typing import List, Dict, Optional
from datasets import Dataset
from ragas import evaluate
import math
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from .config import OPENAI_API_KEY

# Global store for evaluation data
RAGAS_LOG: List[Dict] = []

def log_rag_interaction(
    question: str,
    answer: str,
    contexts: List[str],
    ground_truth: Optional[str] = None,
    model: str = "unknown"
):
    """
    Log a RAG interaction for later evaluation.
    
    Args:
        question: User's query
        answer: LLM's generated answer
        contexts: Retrieved context chunks
        ground_truth: Optional reference answer (for recall)
    """
    RAGAS_LOG.append({
        "question": question,
        "answer": answer,
        "contexts": contexts,
        "ground_truth": ground_truth or answer,  # Use answer as fallback
        "model": model,
    })


async def run_ragas_eval(
    limit: Optional[int] = 50,
    model_filter: Optional[str] = None  # ✅ NEW parameter
) -> Dict:
    """
    Run RAGAS evaluation on recent interactions.
    
    Returns comprehensive scores for:
    - Faithfulness: No hallucinations
    - Context Precision: Retrieved chunks are relevant
    - Answer Relevancy: Answer addresses the question
    - Context Recall: All necessary context retrieved
    """
    
    # Get samples
    samples = RAGAS_LOG[-limit:] if limit and len(RAGAS_LOG) > limit else RAGAS_LOG
    
    # ✅ Filter by model if specified
    if model_filter:
        samples = [s for s in samples if s.get("model") == model_filter]
    
    if not samples:
        return {
            "status": "empty",
            "count": 0,
            "message": f"No interactions logged for model: {model_filter}" if model_filter else "No interactions logged yet."
        }
    
    # Prepare dataset
    data = {
        "question": [s["question"] for s in samples],
        "answer": [s["answer"] for s in samples],
        "contexts": [s["contexts"] for s in samples],
        "ground_truth": [s["ground_truth"] for s in samples],
    }
    
    ds = Dataset.from_dict(data)
    
    # Initialize LLM and embeddings for RAGAS
    llm = LangchainLLMWrapper(ChatOpenAI(
        model="gpt-4o-mini",
        api_key=OPENAI_API_KEY
    ))
    
    embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=OPENAI_API_KEY
    ))
    
    # Run evaluation in thread pool
    loop = asyncio.get_running_loop()
    
    def eval_fn():
        return evaluate(
            ds,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            ],
            llm=llm,
            embeddings=embeddings,
        )
    
    try:
        result = await loop.run_in_executor(None, eval_fn)
        
        # ✅ FIX: Handle different result formats
        scores = {}
        
        # Try to extract scores from result
        if hasattr(result, 'to_pandas'):
            # Result is a Dataset, convert to pandas and get mean scores
            df = result.to_pandas()
            scores = {
                "faithfulness": _sanitize_float(df["faithfulness"].mean()) if "faithfulness" in df else 0.0,
                "answer_relevancy": _sanitize_float(df["answer_relevancy"].mean()) if "answer_relevancy" in df else 0.0,
                "context_precision": _sanitize_float(df["context_precision"].mean()) if "context_precision" in df else 0.0,
                "context_recall": _sanitize_float(df["context_recall"].mean()) if "context_recall" in df else 0.0,
            }
        elif isinstance(result, dict):
            # Result is already a dict
            scores = {
                "faithfulness": _sanitize_float(result.get("faithfulness", 0)),
                "answer_relevancy": _sanitize_float(result.get("answer_relevancy", 0)),
                "context_precision": _sanitize_float(result.get("context_precision", 0)),
                "context_recall": _sanitize_float(result.get("context_recall", 0)),
            }
        else:
            # Fallback: try to access as attributes
            scores = {
                "faithfulness": _sanitize_float(getattr(result, "faithfulness", 0)),
                "answer_relevancy": _sanitize_float(getattr(result, "answer_relevancy", 0)),
                "context_precision": _sanitize_float(getattr(result, "context_precision", 0)),
                "context_recall": _sanitize_float(getattr(result, "context_recall", 0)),
            }
        
        return {
            "status": "ok",
            "count": len(samples),
            "scores": scores,
            "interpretation": interpret_scores(scores)
        }
    
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"[RAGAS ERROR] {error_details}")
        
        return {
            "status": "error",
            "count": len(samples),
            "error": str(e),
            "message": "RAGAS evaluation failed. Check logs.",
            "details": error_details[:500]  # First 500 chars of traceback
        }


def interpret_scores(scores: Dict[str, float]) -> Dict[str, str]:
    """
    Provide human-readable interpretation of RAGAS scores.
    """
    interpretations = {}
    
    # Faithfulness (0-1, higher is better)
    faith = scores.get("faithfulness", 0)
    if faith >= 0.9:
        interpretations["faithfulness"] = "Excellent - minimal hallucinations"
    elif faith >= 0.7:
        interpretations["faithfulness"] = "Good - some unsupported claims"
    else:
        interpretations["faithfulness"] = "Poor - significant hallucinations"
    
    # Answer Relevancy (0-1, higher is better)
    rel = scores.get("answer_relevancy", 0)
    if rel >= 0.9:
        interpretations["answer_relevancy"] = "Excellent - answers on point"
    elif rel >= 0.7:
        interpretations["answer_relevancy"] = "Good - mostly relevant"
    else:
        interpretations["answer_relevancy"] = "Poor - answers off-topic"
    
    # Context Precision (0-1, higher is better)
    prec = scores.get("context_precision", 0)
    if prec >= 0.8:
        interpretations["context_precision"] = "Excellent - high-quality retrieval"
    elif prec >= 0.6:
        interpretations["context_precision"] = "Good - some noise in context"
    else:
        interpretations["context_precision"] = "Poor - irrelevant chunks retrieved"
    
    # Context Recall (0-1, higher is better)
    rec = scores.get("context_recall", 0)
    if rec >= 0.8:
        interpretations["context_recall"] = "Excellent - all context found"
    elif rec >= 0.6:
        interpretations["context_recall"] = "Good - missing some context"
    else:
        interpretations["context_recall"] = "Poor - incomplete retrieval"
    
    return interpretations

def clear_ragas_log():
    """Clear the evaluation log."""
    global RAGAS_LOG
    RAGAS_LOG = []


def get_ragas_log_size() -> int:
    """Get number of logged interactions."""
    return len(RAGAS_LOG)

def _sanitize_float(value: float) -> float:
    """
    Convert NaN/Infinity to JSON-safe values.
    Returns 0.0 for invalid floats.
    """
    if value is None:
        return 0.0
    
    try:
        value = float(value)
        if math.isnan(value) or math.isinf(value):
            return 0.0
        return value
    except (ValueError, TypeError):
        return 0.0