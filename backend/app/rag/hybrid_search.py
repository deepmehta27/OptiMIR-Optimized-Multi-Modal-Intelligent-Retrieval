"""
Hybrid search combining semantic (vector) + keyword (BM25) retrieval.
"""
from typing import List
from rank_bm25 import BM25Okapi
import numpy as np
from .types import RetrievedChunk
from .ingest import get_or_create_collection 

class HybridRetriever:
    def __init__(self):
        self.collection = get_or_create_collection()
        self.bm25 = None
        self.documents = []
        self.metadata = []
        self._initialize_bm25()
    
    def _initialize_bm25(self):
        """Build BM25 index from Chroma collection."""
        results = self.collection.get(include=["documents", "metadatas"])
        self.documents = results.get("documents", [])
        self.metadata = results.get("metadatas", [])
        
        if not self.documents:
            return
        
        # Tokenize documents for BM25
        tokenized_corpus = [doc.lower().split() for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_corpus)
    
    def bm25_search(self, query: str, k: int = 20) -> List[tuple]:
        """
        Keyword search using BM25.
        Returns: List of (doc_text, metadata, bm25_score)
        """
        if not self.bm25 or not self.documents:
            return []
        
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k results
        top_indices = np.argsort(scores)[::-1][:k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include relevant results
                results.append((
                    self.documents[idx],
                    self.metadata[idx],
                    float(scores[idx])
                ))
        
        return results
    
    def semantic_search(self, query: str, k: int = 20) -> List[tuple]:
        """
        Vector similarity search using Chroma.
        Returns: List of (doc_text, metadata, distance_score)
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=k
        )
        
        search_results = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            search_results.append((doc, meta, float(dist)))
        
        return search_results
    
    def hybrid_search(self, query: str, k: int = 12, alpha: float = 0.5) -> List[RetrievedChunk]:
        """
        Combine semantic + keyword search with weighted scoring.
        
        Args:
            query: Search query
            k: Final number of results to return
            alpha: Weight for semantic vs keyword (0=all keyword, 1=all semantic)
        
        Returns:
            List of RetrievedChunk objects, reranked by hybrid score
        """
        # Get results from both methods
        semantic_results = self.semantic_search(query, k=20)
        keyword_results = self.bm25_search(query, k=20)
        
        # Normalize scores to [0, 1]
        def normalize_scores(results):
            if not results:
                return []
            scores = [r[2] for r in results]
            max_score = max(scores) if scores else 1
            min_score = min(scores) if scores else 0
            range_score = max_score - min_score if max_score != min_score else 1
            
            normalized = []
            for doc, meta, score in results:
                norm_score = (score - min_score) / range_score
                normalized.append((doc, meta, norm_score))
            return normalized
        
        semantic_norm = normalize_scores(semantic_results)
        keyword_norm = normalize_scores(keyword_results)
        
        # Merge results with hybrid scoring
        doc_scores = {}
        for doc, meta, score in semantic_norm:
            key = doc[:100]  # Use first 100 chars as key
            doc_scores[key] = {
                "doc": doc,
                "meta": meta,
                "semantic_score": score,
                "keyword_score": 0
            }
        
        for doc, meta, score in keyword_norm:
            key = doc[:100]
            if key in doc_scores:
                doc_scores[key]["keyword_score"] = score
            else:
                doc_scores[key] = {
                    "doc": doc,
                    "meta": meta,
                    "semantic_score": 0,
                    "keyword_score": score
                }
        
        # Calculate hybrid score: alpha * semantic + (1-alpha) * keyword
        ranked_results = []
        for key, data in doc_scores.items():
            hybrid_score = (alpha * data["semantic_score"] + 
                          (1 - alpha) * data["keyword_score"])
            ranked_results.append((data["doc"], data["meta"], hybrid_score))
        
        # Sort by hybrid score and take top-k
        ranked_results.sort(key=lambda x: x[2], reverse=True)
        top_results = ranked_results[:k]
        
        # Convert to RetrievedChunk objects
        chunks = []
        for doc, meta, score in top_results:
            chunks.append(RetrievedChunk(
                text=doc,
                source=meta.get("source", "unknown"),
                score=score,
                type=meta.get("type"),
                page=meta.get("page")
            ))
        
        return chunks

# Global retriever instance
_hybrid_retriever = None

def get_hybrid_retriever() -> HybridRetriever:
    """Get or create singleton hybrid retriever."""
    global _hybrid_retriever
    if _hybrid_retriever is None:
        _hybrid_retriever = HybridRetriever()
    return _hybrid_retriever

async def hybrid_retrieve_chunks(query: str, k: int = 12) -> List[RetrievedChunk]:
    """Async wrapper for hybrid search."""
    retriever = get_hybrid_retriever()
    return retriever.hybrid_search(query, k=k, alpha=0.6)  # 60% semantic, 40% keyword
