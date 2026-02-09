"""Advanced Retrieval Techniques - BM25 + Hybrid Search with RRF"""
import re
import logging
from typing import List, Tuple
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


class AdvancedRetriever:
    """Advanced retrieval with BM25 keyword search and hybrid search"""
    
    def __init__(self, vector_db):
        self.vector_db = vector_db
        self.bm25_index = None
        self.documents = []
    
    def build_bm25_index(self, documents: List[Document]):
        """
        Build keyword-based search index for hybrid retrieval
        Uses BM25 algorithm for keyword matching
        """
        try:
            self.documents = documents
            tokenized_docs = [self._tokenize(doc.page_content) for doc in documents]
            self.bm25_index = BM25Okapi(tokenized_docs)
        except Exception as e:
            logger.error(f"Failed to build BM25 index: {e}")
            self.bm25_index = None
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text for BM25: lowercase, remove punctuation, split on whitespace"""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)  # Replace punctuation with space
        return text.split()
    
    def bm25_search(self, query: str, k: int = 10) -> List[Tuple[Document, float]]:
        """
        Keyword-based search using BM25
        """
        if not self.bm25_index or not self.documents:
            return []
        
        try:
            tokenized_query = self._tokenize(query)
            scores = self.bm25_index.get_scores(tokenized_query)
            
            # Get top k results
            top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
            results = [(self.documents[i], scores[i]) for i in top_indices if scores[i] > 0]
            
            return results
            
        except Exception as e:
            logger.error(f"BM25 search failed: {e}")
            return []
    
    def hybrid_search(self, query: str, k: int = 10, rrf_k: int = 60) -> List[Document]:
        """
        Hybrid Search using Reciprocal Rank Fusion (RRF)
        Combines vector similarity + keyword matching (BM25) using rank-based fusion.
        
        RRF is more robust than weighted score combination because it only uses
        rank positions, avoiding issues with incompatible score scales.
        
        Formula: RRF(d) = sum(1 / (rrf_k + rank(d))) across all retrievers
        
        Args:
            rrf_k: RRF constant (default 60, standard value from the original paper)
        """
        vector_results = []
        try:
            # Vector search
            vector_results = self.vector_db.similarity_search_with_score(query, k=k)
            
            # BM25 search
            bm25_results = self.bm25_search(query, k=k) if self.bm25_index else []
            
            if not bm25_results:
                return [doc for doc, _ in vector_results]
            
            # Build a map of all documents by content
            all_docs = {doc.page_content: doc for doc, _ in vector_results + bm25_results}
            
            # Compute RRF scores
            rrf_scores = {}
            
            # Vector results: lower distance = better rank in ChromaDB
            for rank, (doc, _) in enumerate(vector_results):
                rrf_scores[doc.page_content] = rrf_scores.get(doc.page_content, 0) + 1 / (rrf_k + rank + 1)
            
            # BM25 results: already sorted by descending score
            for rank, (doc, _) in enumerate(bm25_results):
                rrf_scores[doc.page_content] = rrf_scores.get(doc.page_content, 0) + 1 / (rrf_k + rank + 1)
            
            # Sort by RRF score (higher is better)
            sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:k]
            
            results = [all_docs[content] for content, _ in sorted_docs]
            return results
            
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            return [doc for doc, _ in vector_results[:k]]
    
    def intelligent_retrieve(
        self,
        query: str,
        strategy: str = "hybrid",
        k: int = 10,
        **kwargs
    ) -> List[Document]:
        """
        Master retrieval function using BM25, vector, or hybrid strategy
        
        Strategies:
        - "vector": Pure semantic search
        - "bm25": Pure keyword search
        - "hybrid": Combines both (recommended)
        """
        try:
            if strategy == "hybrid" and self.bm25_index:
                results = self.hybrid_search(query, k=k)
            elif strategy == "bm25" and self.bm25_index:
                results = [doc for doc, _ in self.bm25_search(query, k=k)]
            else:  # Default to vector
                results = [doc for doc, _ in self.vector_db.similarity_search_with_score(query, k=k)]
            
            return results[:k]
            
        except Exception as e:
            logger.error(f"Retrieval failed (strategy={strategy}): {e}")
            # Fallback to basic vector search
            try:
                return [doc for doc, _ in self.vector_db.similarity_search_with_score(query, k=k)]
            except Exception as fallback_error:
                logger.error(f"Fallback vector search also failed: {fallback_error}")
                return []
