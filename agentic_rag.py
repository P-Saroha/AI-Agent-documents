"""Agentic RAG System with intelligent query processing"""
from typing import List, Dict
from langchain_core.documents import Document
import json
import time


def call_llm_with_retry(llm, prompt, max_retries=3):
    """Call LLM with automatic retry on rate limit (429) errors"""
    for attempt in range(max_retries):
        try:
            return llm.invoke(prompt)
        except Exception as e:
            error_msg = str(e)
            if '429' in error_msg or 'RESOURCE_EXHAUSTED' in error_msg:
                wait_time = 20 * (attempt + 1)  # 20s, 40s, 60s
                print(f"[Rate Limit] Waiting {wait_time}s before retry ({attempt+1}/{max_retries})...")
                time.sleep(wait_time)
            else:
                raise e
    raise Exception("Rate limit exceeded. Please wait a minute and try again.")


class AgenticRAG:
    """Agentic RAG System with 5-step intelligent query processing"""
    
    def __init__(self, llm, vector_db, advanced_retriever=None):
        self.llm = llm
        self.vector_db = vector_db
        self.advanced_retriever = advanced_retriever
        self.conversation_history = []
        self.use_advanced_retrieval = advanced_retriever is not None
    
    def analyze_query(self, query: str) -> Dict:
        """Analyze query intent and complexity"""
        analysis_prompt = f"""You are an intelligent query analyzer. Analyze this user query:

Query: "{query}"

Provide a JSON response with:
1. "complexity": "simple" or "complex"
2. "intent": What does the user want to know?
3. "requires_multi_step": true if query needs multiple retrieval steps
4. "key_concepts": List of main concepts to search for

Respond ONLY with valid JSON."""

        try:
            response = call_llm_with_retry(self.llm, analysis_prompt)
            analysis = json.loads(response.content)
            return analysis
        except:
            # Fallback
            return {
                "complexity": "simple",
                "intent": query,
                "requires_multi_step": False,
                "key_concepts": [query]
            }
    
    def decompose_query(self, query: str, analysis: Dict) -> List[str]:
        """Break complex queries into sub-queries"""
        # Always decompose summary/list questions to get comprehensive coverage
        needs_decomposition = (
            analysis.get("requires_multi_step", False) or
            any(keyword in query.lower() for keyword in ["summary", "summarize", "list", "techniques", "methods", "all", "what are"])
        )
        
        if not needs_decomposition:
            return [query]
        
        decomposition_prompt = f"""Break down this query into 3-4 focused sub-queries to gather comprehensive information:

Query: "{query}"

If the query asks for a summary or list, create sub-queries that explore different aspects.
For example, if asking for "fine-tuning techniques", create queries like:
- "What are the parameter-efficient fine-tuning methods?"
- "What are traditional fine-tuning approaches?"
- "What preference optimization techniques are mentioned?"

Respond with a JSON list of sub-queries.
Example: ["sub-query 1", "sub-query 2", "sub-query 3"]"""

        try:
            response = call_llm_with_retry(self.llm, decomposition_prompt)
            sub_queries = json.loads(response.content)
            return sub_queries
        except:
            return [query]
    
    def retrieve_with_strategy(self, query: str, num_results: int = 5, strategy: str = "auto") -> List[Document]:
        """
        AGENT STEP 3: Intelligent retrieval with strategy selection
        Supports advanced retrieval techniques when available
        
        Args:
            strategy: "auto", "vector", "hybrid", "bm25"
        """
        try:
            # Use advanced retrieval if available
            if self.use_advanced_retrieval and strategy != "vector":
                # Auto-select strategy based on query
                if strategy == "auto":
                    # Use hybrid for comprehensive queries
                    if any(k in query.lower() for k in ["list", "all", "summary", "techniques"]):
                        strategy = "hybrid"
                    else:
                        strategy = "vector"
                
                results = self.advanced_retriever.intelligent_retrieve(
                    query,
                    strategy=strategy if strategy != "auto" else "hybrid",
                    k=num_results
                )
                
                if results:
                    return results
            
            # Fallback to standard vector search
            results = self.vector_db.similarity_search_with_score(query, k=num_results)
        
            if not results:
                return []
            
            # Filter by relevance score (lower is better for ChromaDB)
            filtered_results = []
            for doc, score in results:
                if score < 1.5:  # Threshold for relevance  
                    filtered_results.append(doc)
            
            return filtered_results if filtered_results else [doc for doc, _ in results[:3]]
            
        except Exception as e:
            print(f"Warning: Retrieval failed: {str(e)}")
            # Emergency fallback
            try:
                results = self.vector_db.similarity_search(query, k=num_results)
                return results
            except:
                return []
    
    def synthesize_answer(self, query: str, context_docs: List[Document], analysis: Dict, conversation_context: str = "") -> str:
        """Synthesize answer from retrieved context"""
        if not context_docs:
            return "I couldn't find relevant information in the documents to answer your question."
        
        # Prepare context
        context = "\n\n".join([
            f"[Source: {doc.metadata.get('source', 'Unknown')}]\n{doc.page_content}"
            for doc in context_docs
        ])
        
        # Build conversation history section
        conv_section = ""
        if conversation_context:
            conv_section = f"""\n{conversation_context}\nUse the conversation history above to understand follow-up questions and maintain context.\n"""
        
        synthesis_prompt = f"""You are an intelligent AI assistant. Answer the user's question comprehensively based on the provided context.
{conv_section}
Context from documents:
{context}

User Question: {query}

Instructions:
1. Be comprehensive and structured: Organize your answer with clear sections and numbered/bulleted lists
2. Extract ALL relevant information from the context
3. Use markdown formatting (headers, bold, lists) for clarity
4. Be detailed: Include explanations, formulas, and specific details from the context
5. DO NOT include source citations inline or after paragraphs. Sources are shown separately by the system
6. Structure like this:
   - Start with a brief overview
   - Break complex topics into numbered sections
   - Use clear headings
   - List techniques/concepts with brief explanations
   - Include technical details (formulas, numbers, specifics)
7. For "list X" or "what techniques" questions: Extract and list EVERY instance mentioned in the context

Answer:"""

        try:
            response = call_llm_with_retry(self.llm, synthesis_prompt)
            answer = response.content

            return answer
        except Exception as e:
            return f"Error generating answer: {str(e)}"
    
    def verify_answer(self, query: str, answer: str) -> Dict:
        """
        AGENT STEP 5: Self-reflection to verify answer quality
        """
        verification_prompt = f"""Verify if this answer adequately addresses the question:

Question: {query}
Answer: {answer}

Provide JSON with:
- "is_adequate": true/false
- "confidence": 0-100
- "suggestion": Any improvement suggestions

Respond ONLY with valid JSON."""

        try:
            response = call_llm_with_retry(self.llm, verification_prompt)
            verification = json.loads(response.content)
            return verification
        except:
            return {"is_adequate": True, "confidence": 80, "suggestion": ""}
    
    def process_query(self, query: str, strategy: str = "auto") -> Dict:
        """
        Main agentic workflow - orchestrates all steps
        
        Args:
            query: User question
            strategy: Retrieval strategy ("auto", "hybrid", "vector", "bm25")
        """
        # STEP 1: Analyze query
        analysis = self.analyze_query(query)
        
        # STEP 2: Decompose if needed
        sub_queries = self.decompose_query(query, analysis)
        
        # STEP 3: Retrieve documents for each sub-query
        all_docs = []
        for sq in sub_queries:
            docs = self.retrieve_with_strategy(sq, num_results=8, strategy=strategy)
            all_docs.extend(docs)
        
        # Remove duplicates
        unique_docs = []
        seen = set()
        for doc in all_docs:
            doc_id = doc.page_content[:100]  # Use first 100 chars as ID
            if doc_id not in seen:
                seen.add(doc_id)
                unique_docs.append(doc)
        
        # STEP 4: Synthesize answer (use more context for comprehensive answers)
        conversation_context = self.get_conversation_context()
        answer = self.synthesize_answer(query, unique_docs[:10], analysis, conversation_context)
        
        # STEP 5: Verify answer
        verification = self.verify_answer(query, answer)
        
        # Store in conversation history
        self.conversation_history.append({
            "query": query,
            "answer": answer,
            "sources": [doc.metadata.get('source', 'Unknown') for doc in unique_docs[:10]]
        })
        
        return {
            "query": query,
            "answer": answer,
            "sources": unique_docs[:10],
            "analysis": analysis,
            "verification": verification,
            "num_sources": len(unique_docs[:10])
        }
    
    def get_conversation_context(self) -> str:
        """Get conversation history for context"""
        if not self.conversation_history:
            return ""
        
        context = "Previous conversation:\n"
        for item in self.conversation_history[-3:]:  # Last 3 exchanges
            context += f"Q: {item['query']}\nA: {item['answer'][:200]}...\n\n"
        
        return context
