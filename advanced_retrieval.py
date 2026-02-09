"""Advanced Retrieval Techniques"""
from typing import List, Dict, Tuple
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi


class AdvancedRetriever:
    """Advanced retrieval with query expansion, hybrid search, and MMR"""
    
    def __init__(self, llm, vector_db):
        self.llm = llm
        self.vector_db = vector_db
        self.bm25_index = None
        self.documents = []
    
    def expand_query(self, query: str) -> List[str]:
        """
        ADVANCED TECHNIQUE 1: Query Expansion
        Generate alternative phrasings to improve recall
        """
        try:
            expansion_prompt = f"""Generate 2-3 alternative ways to phrase this query for document search.
Return ONLY a Python list of strings.

Original query: "{query}"

Examples:
["alternative 1", "alternative 2", "alternative 3"]"""

            response = self.llm.invoke(expansion_prompt)
            
            # Parse the response
            import ast
            alternatives = ast.literal_eval(response.content.strip())
            
            # Combine with original
            expanded_queries = [query] + alternatives
            return expanded_queries
            
        except Exception as e:
            return [query]
    
    def rewrite_query(self, query: str, context: str = "") -> str:
        """
        ADVANCED TECHNIQUE 2: Query Rewriting
        Optimize query for better retrieval based on context
        """
        try:
            if not context:
                return query
            
            rewrite_prompt = f"""Rewrite this query to be more specific and retrieval-friendly.
Keep it concise but add relevant technical terms if needed.

Original query: "{query}"
Context: {context}

Return ONLY the rewritten query, nothing else."""

            response = self.llm.invoke(rewrite_prompt)
            rewritten = response.content.strip().strip('"')
            return rewritten
            
        except Exception as e:
            return query
    
    def build_bm25_index(self, documents: List[Document]):
        """
        ADVANCED TECHNIQUE 3: BM25 Index
        Build keyword-based search index for hybrid retrieval
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
        ADVANCED TECHNIQUE 4: Hybrid Search
        Combines vector similarity + keyword matching (BM25)
        
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
    
    def mmr_rerank(self, query: str, documents: List[Document], k: int = 5, lambda_param: float = 0.5) -> List[Document]:
        """
        ADVANCED TECHNIQUE 5: Maximal Marginal Relevance (MMR)
        Rerank to balance relevance and diversity
        
        Args:
            lambda_param: Trade-off between relevance (1) and diversity (0)
        """
        if not documents or len(documents) <= k:
            return documents[:k]
        
        try:
            selected = []
            remaining = documents.copy()
            
            # Select first most relevant
            selected.append(remaining.pop(0))
            
            while len(selected) < k and remaining:
                mmr_scores = []
                
                for doc in remaining:
                    # Calculate similarity to query (using position as proxy)
                    relevance = 1.0 / (documents.index(doc) + 1)
                    
                    # Calculate max similarity to already selected
                    if selected:
                        diversity = min(
                            self._doc_similarity(doc, sel_doc)
                            for sel_doc in selected
                        )
                    else:
                        diversity = 1.0
                    
                    # MMR score
                    mmr_score = lambda_param * relevance - (1 - lambda_param) * diversity
                    mmr_scores.append((doc, mmr_score))
                
                # Select best MMR score
                best_doc = max(mmr_scores, key=lambda x: x[1])[0]
                selected.append(best_doc)
                remaining.remove(best_doc)
            
            return selected
            
        except Exception as e:
            return documents[:k]
    
    def _doc_similarity(self, doc1: Document, doc2: Document) -> float:
        """Simple Jaccard similarity between documents"""
        try:
            words1 = set(doc1.page_content.lower().split())
            words2 = set(doc2.page_content.lower().split())
            
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            
            return intersection / union if union > 0 else 0.0
        except:
            return 0.0
    
    def intelligent_retrieve(
        self,
        query: str,
        strategy: str = "hybrid",
        k: int = 10,
        use_expansion: bool = True,
        use_reranking: bool = True
    ) -> List[Document]:
        """
        Master retrieval function combining all advanced techniques
        
        Strategies:
        - "vector": Pure semantic search
        - "bm25": Pure keyword search
        - "hybrid": Combines both (recommended)
        """
        try:
            # Query expansion
            queries = self.expand_query(query) if use_expansion else [query]
            
            # Retrieve with chosen strategy
            all_results = []
            for q in queries:
                if strategy == "hybrid" and self.bm25_index:
                    results = self.hybrid_search(q, k=k)
                elif strategy == "bm25" and self.bm25_index:
                    results = [doc for doc, _ in self.bm25_search(q, k=k)]
                else:  # Default to vector
                    results = [doc for doc, _ in self.vector_db.similarity_search_with_score(q, k=k)]
                
                all_results.extend(results)
            
            # Remove duplicates
            unique_docs = []
            seen = set()
            for doc in all_results:
                doc_id = doc.page_content[:100]
                if doc_id not in seen:
                    seen.add(doc_id)
                    unique_docs.append(doc)
            
            # Rerank for diversity
            final_results = self.mmr_rerank(query, unique_docs, k=k) if use_reranking else unique_docs[:k]
            
            return final_results
            
        except Exception as e:
            # Fallback to basic vector search
            try:
                return [doc for doc, _ in self.vector_db.similarity_search_with_score(query, k=k)]
            except:
                return []
