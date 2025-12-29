"""
Shared types for RAG system to avoid circular imports.
"""
from pydantic import BaseModel
from typing import List, Literal, Optional

class RetrievedChunk(BaseModel):
    text: str
    source: str
    score: float
    type: str | None = None
    page: int | None = None

class QueryRequest(BaseModel):
    question: str
    model: Literal["gpt4o-mini", "gpt4o", "claude-haiku", "claude-sonnet"] = "gpt4o-mini"
    use_context: bool = True

class RAGResponse(BaseModel):
    answer: str
    model: str
    chunks: List[RetrievedChunk]

class ChatHistoryItem(BaseModel):
    role: Literal["user", "assistant"]
    text: str

class ChatRequest(BaseModel):
    question: str
    model: Literal["gpt4o-mini", "gpt4o", "claude-haiku", "claude-sonnet"] = "gpt4o-mini"
    use_context: bool = True
    history: list[ChatHistoryItem] | None = None
