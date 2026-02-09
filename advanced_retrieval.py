"""Advanced Retrieval Techniques - BM25 + Hybrid Search"""
from typing import List, Tuple
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi


class AdvancedRetriever:
    """Advanced retrieval with BM25 keyword search and hybrid search"""
    
    def __init__(self, llm, vector_db):
        self.llm = llm
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
            tokenized_docs = [doc.page_content.lower().split() for doc in documents]
            self.bm25_index = BM25Okapi(tokenized_docs)
        except Exception as e:
            self.bm25_index = None
    
    def bm25_search(self, query: str, k: int = 10) -> List[Tuple[Document, float]]:
        """
        Keyword-based search using BM25
        """
        if not self.bm25_index or not self.documents:
            return []
        
        try:
            tokenized_query = query.lower().split()
            scores = self.bm25_index.get_scores(tokenized_query)
            
            # Get top k results
            top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
            results = [(self.documents[i], scores[i]) for i in top_indices if scores[i] > 0]
            
            return results
            
        except Exception as e:
            return []
    
    def hybrid_search(self, query: str, k: int = 10, alpha: float = 0.5) -> List[Document]:
        """
        Hybrid Search: Combines vector similarity + keyword matching (BM25)
        
        Args:
            alpha: Weight for vector search (0=pure BM25, 1=pure vector)
        """
        try:
            # Vector search
            vector_results = self.vector_db.similarity_search_with_score(query, k=k)
            
            # BM25 search
            bm25_results = self.bm25_search(query, k=k) if self.bm25_index else []
            
            if not bm25_results:
                return [doc for doc, _ in vector_results]
            
            # Normalize scores
            def normalize_scores(results):
                if not results:
                    return {}
                scores = [score for _, score in results]
                min_score, max_score = min(scores), max(scores)
                if max_score == min_score:
                    return {doc.page_content: 0.5 for doc, _ in results}
                return {
                    doc.page_content: (score - min_score) / (max_score - min_score)
                    for doc, score in results
                }
            
            # Normalize (for vector, lower is better in ChromaDB, so invert)
            vector_scores = {
                doc.page_content: 1 - score for doc, score in vector_results
            }
            bm25_scores = normalize_scores(bm25_results)
            
            # Combine scores
            all_docs = {doc.page_content: doc for doc, _ in vector_results + bm25_results}
            combined_scores = {}
            
            for content, doc in all_docs.items():
                vec_score = vector_scores.get(content, 0)
                bm25_score = bm25_scores.get(content, 0)
                combined_scores[content] = (alpha * vec_score + (1 - alpha) * bm25_score)
            
            # Sort by combined score
            sorted_docs = sorted(
                combined_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )[:k]
            
            results = [all_docs[content] for content, _ in sorted_docs]
            return results
            
        except Exception as e:
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
            # Fallback to basic vector search
            try:
                return [doc for doc, _ in self.vector_db.similarity_search_with_score(query, k=k)]
            except:
                return []
