# Agentic RAG System - System Design Document


---

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [System Architecture](#system-architecture)
3. [Agentic Workflow Design](#agentic-workflow-design)
4. [Context Construction Strategy](#context-construction-strategy)
5. [Technology Choices and Rationale](#technology-choices-and-rationale)
6. [Key Design Decisions](#key-design-decisions)
7. [Implementation Details](#implementation-details)
8. [Limitations](#limitations)
9. [Future Enhancements](#future-enhancements)

---

## 1. Executive Summary

The Agentic RAG (Retrieval-Augmented Generation) System is an intelligent document question-answering application that combines the power of Large Language Models with semantic search capabilities. Unlike traditional RAG systems, this implementation features an **agentic workflow** where the AI actively analyzes queries, plans retrieval strategies, and verifies its own responses.

### Key Innovations
- **Multi-step reasoning**: Agent breaks down complex queries automatically
- **Self-verification**: Agent evaluates its own answer quality
- **Adaptive retrieval**: Dynamic search strategy based on query complexity
- **Multi-format support**: Handles PDF, DOCX, PPTX, Excel, and text files

---

## 2. System Architecture

### 2.1 High-Level Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    User Interface Layer                   │
│                      (Streamlit)                          │
└────────────────────────┬─────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────┐
│                 Agentic RAG Engine                        │
│  ┌────────────────────────────────────────────────────┐  │
│  │ 1. Query Analyzer    → Understand intent           │  │
│  │ 2. Query Decomposer  → Break complex queries       │  │
│  │ 3. Retrieval Manager → Smart document search       │  │
│  │ 4. Answer Synthesizer→ Generate comprehensive      │  │
│  │ 5. Verifier          → Self-check quality          │  │
│  └────────────────────────────────────────────────────┘  │
└────────────────┬──────────────────────┬──────────────────┘
                 │                      │
                 ▼                      ▼
    ┌────────────────────┐  ┌──────────────────────┐
    │   LLM Service      │  │   Vector Database    │
    │(Gemini 2.5 Flash)  │  │    (ChromaDB)        │
    │                    │  │  + BM25 Index        │
    └────────────────────┘  └──────────┬───────────┘
                                       │
                                       ▼
                            ┌───────────────────────┐
                            │ Document Processor    │
                            │ • PDF Parser          │
                            │ • DOCX Parser         │
                            │ • PPTX Parser         │
                            │ • Excel Parser        │
                            │ • Text Parser         │
                            └───────────────────────┘
```

### 2.2 Component Descriptions

#### User Interface Layer
- **Technology**: Streamlit
- **Responsibilities**: 
  - File upload management
  - Chat interface
  - Results visualization
  - Agent analysis display

#### Agentic RAG Engine
- **Location**: `agentic_rag.py`
- **Responsibilities**:
  - Orchestrate the 5-step agentic workflow
  - Manage conversation context
  - Coordinate LLM and vector DB interactions

#### LLM Service
- **Provider**: Google Gemini 2.5 Flash
- **Usage**:
  - Query analysis
  - Query decomposition
  - Answer generation
  - Self-verification

#### Vector Database
- **Technology**: ChromaDB
- **Responsibilities**:
  - Store document embeddings
  - Perform semantic similarity search
  - Manage document metadata

#### Document Processor
- **Location**: `document_processor.py`
- **Responsibilities**:
  - Parse multiple document formats
  - Chunk documents intelligently
  - Preserve metadata

---

## 3. Agentic Workflow Design

### 3.1 The Five-Step Agentic Process

```
User Query
    ↓
┌─────────────────────────────────────────────────────────┐
│ STEP 1: Query Analysis                                  │
│ ─────────────────────────────────────────────────────── │
│ Agent analyzes:                                         │
│ • Query complexity (simple vs complex)                  │
│ • User intent                                           │
│ • Whether multi-step reasoning needed                   │
│ • Key concepts to search for                            │
│                                                         │
│ Output: {complexity, intent, requires_multi_step, [...]}│
└────────────────────────┬────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│ STEP 2: Query Decomposition (if needed)                 │
│ ─────────────────────────────────────────────────────── │
│ For complex queries:                                    │
│ • Break into 2-3 sub-queries                            │
│ • Each sub-query targets specific aspect                │
│ • Ensures comprehensive coverage                        │
│                                                         │
│ Example: "Compare ML types and their applications"      │
│ → ["What are the types of ML?",                        │
│    "What are applications of each ML type?"]           │
└────────────────────────┬────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│ STEP 3: Intelligent Retrieval                           │
│ ─────────────────────────────────────────────────────── │
│ Strategy selection:                                     │
│ • Determine number of documents to retrieve             │
│ • Execute similarity search for each sub-query          │
│ • Apply relevance filtering (threshold: 1.5)            │
│ • Deduplicate results                                   │
│                                                         │
│ Output: Ranked list of relevant document chunks         │
└────────────────────────┬────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│ STEP 4: Answer Synthesis                                │
│ ─────────────────────────────────────────────────────── │
│ Agent generates answer:                                 │
│ • Injects conversation history (last 3 Q&A pairs)      │
│ • Combines information from multiple sources            │
│ • Maintains context coherence for follow-up questions   │
│ • Formats response clearly with markdown                │
│                                                         │
│ Prompt includes: Conversation + Context + Query + Rules │
└────────────────────────┬────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│ STEP 5: Self-Verification                               │
│ ─────────────────────────────────────────────────────── │
│ Agent evaluates its own answer:                         │
│ • Adequacy check (does it answer the question?)         │
│ • Confidence score (0-100)                              │
│ • Improvement suggestions                               │
│                                                         │
│ Output: {is_adequate, confidence, suggestion}           │
└────────────────────────┬────────────────────────────────┘
                         ↓
                  Final Answer + Metadata
```

### 3.2 Why This is "Agentic"

Traditional RAG systems follow a fixed pipeline: **Query → Retrieve → Generate**

Our Agentic RAG system demonstrates intelligence through:

1. **Reasoning**: Analyzes query complexity before acting
2. **Planning**: Decomposes complex tasks into sub-tasks
3. **Adaptation**: Adjusts retrieval strategy based on query type
4. **Tool Use**: Dynamically uses vector search and LLM
5. **Self-Evaluation**: Reflects on answer quality
6. **Memory**: Maintains conversation context (last 3 exchanges) for follow-ups
7. **Hybrid Retrieval**: Combines vector + keyword search via Reciprocal Rank Fusion

### 3.3 Example Agentic Flow

**User Query**: *"What are the main differences between supervised and unsupervised learning, and give examples of each?"*

**Agent's Internal Process**:
```
1. ANALYSIS:
   - Complexity: Complex (multi-part question)
   - Intent: Compare two ML paradigms + examples
   - Multi-step: Yes

2. DECOMPOSITION:
   - "What is supervised learning?"
   - "What is unsupervised learning?"
   - "What are examples of each?"

3. RETRIEVAL:
   - Search for "supervised learning" → 4 chunks
   - Search for "unsupervised learning" → 4 chunks
   - Search for "ML examples" → 4 chunks
   - Deduplicate → 8 unique chunks

4. SYNTHESIS:
   - Combine information about supervised learning
   - Combine information about unsupervised learning
   - Extract examples for both
   - Structure as comparison

5. VERIFICATION:
   - Check if both paradigms explained ✓
   - Check if examples provided ✓
   - Confidence: 92%
```

---

## 4. Context Construction Strategy

### 4.1 Document Processing Pipeline

```
Input Document
    ↓
┌─────────────────────┐
│  Format Detection   │ ← Identify file type by extension
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│  Content Extraction │ ← Type-specific parser
│  • PDF: pypdf       │
│  • DOCX: python-docx│
│  • PPTX: python-pptx│
│  • Excel: openpyxl  │
│  • TXT: native      │
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│  Text Chunking      │ ← RecursiveCharacterTextSplitter
│  • Size: 1000 chars │   - Preserves semantic boundaries
│  • Overlap: 200     │   - Maintains context continuity
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│ Metadata Attachment │ ← Add source, type, page/slide info
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│  Embedding          │ ← HuggingFace all-MiniLM-L6-v2 (local)
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│ Vector DB Storage   │ ← ChromaDB persistent storage
└─────────────────────┘
```

### 4.2 Chunking Strategy Rationale

**Chunk Size: 1000 characters**
- **Pros**: 
  - Large enough to preserve context
  - Small enough for focused retrieval
  - Fits well within embedding model limits
- **Cons**: 
  - May split some concepts
  - Mitigated by overlap

**Overlap: 200 characters**
- **Purpose**: Ensure continuity at chunk boundaries
- **Benefit**: Concepts spanning boundaries still retrievable
- **Trade-off**: Slightly redundant storage (acceptable)

### 4.3 Embedding Strategy

**Model**: HuggingFace all-MiniLM-L6-v2 (runs locally)
- **Dimensions**: 384
- **Max Input**: 512 tokens
- **Advantages**:
  - Runs locally (no API calls)
  - No cost, no rate limits
  - Fast inference on CPU
  - Good quality semantic representations

### 4.4 Retrieval Strategy

**Dual Retrieval System**:
- **Vector Search**: ChromaDB semantic similarity (L2 distance)
- **BM25 Keyword Search**: rank-bm25 (BM25Okapi) for exact term matching
- **Fusion**: Reciprocal Rank Fusion (RRF) combines both ranked lists

**RRF Formula**: `RRF(d) = Σ 1/(k + rank(d))` where k=60 (standard constant)

**Why RRF over Weighted Scores**:
- ChromaDB returns Euclidean distances, BM25 returns TF-IDF scores — incompatible scales
- RRF uses only rank positions, making it scale-free and robust
- Documents appearing in both result sets are naturally boosted

**Retrieval Strategies**:
| Strategy | Description |
|----------|-------------|
| `auto` | System selects based on query keywords |
| `hybrid` | Vector + BM25 with RRF fusion (best quality) |
| `vector` | Pure semantic embedding search |
| `bm25` | Pure keyword-based search |

**K Selection**: k=8 per sub-query, top 10 unique chunks sent to LLM

**Relevance Filtering**:
- Threshold: 1.5 (L2 distance)
- Ensures only relevant chunks used
- Prevents noise in context

---

## 5. Technology Choices and Rationale

### 5.1 Technology Stack

| Technology | Choice | Rationale |
|-----------|--------|-----------|
| **LLM** | Google Gemini 2.5 Flash | • Free API access<br>• Fast inference<br>• Strong reasoning capabilities<br>• Good context window<br>• Latest model |
| **Vector DB** | ChromaDB | • Simple setup<br>• Local persistence<br>• No external service needed<br>• Good performance for medium datasets |
| **Embeddings** | HuggingFace all-MiniLM-L6-v2 | • Runs locally on CPU<br>• Free, no API needed<br>• Fast inference<br>• 384 dimensions |
| **Framework** | LangChain | • RAG utilities<br>• Document loaders<br>• Text splitters<br>• Vector store integrations |
| **UI** | Streamlit | • Rapid development<br>• Clean interface<br>• Built-in chat components<br>• Easy deployment |
| **Language** | Python 3.8+ | • ML ecosystem<br>• Library availability<br>• Easy to maintain |

### 5.2 Alternative Considerations

**Milvus vs ChromaDB**
- **Chose ChromaDB**: Simpler setup, sufficient for project scope
- **Milvus advantages**: Better scalability, production-ready
- **Decision**: ChromaDB for simplicity, can migrate to Milvus if needed

**OpenAI vs Gemini**
- **Chose Gemini**: Free API, no credit card required
- **OpenAI advantages**: Slightly better at some tasks
- **Decision**: Gemini provides excellent quality at no cost

**LangChain vs Custom Implementation**
- **Chose LangChain**: Proven patterns, rapid development
- **Custom advantages**: More control, less dependencies
- **Decision**: LangChain accelerates development significantly

---

## 6. Key Design Decisions

### 6.1 Agentic Architecture

**Decision**: Implement five-step agentic workflow

**Reasoning**:
- Goes beyond simple RAG
- Demonstrates AI reasoning capabilities
- Improves answer quality through planning
- Provides transparency (user sees agent thinking)
- Enables self-improvement through verification

**Trade-offs**:
- More LLM calls (higher latency)
- Increased API costs
- More complex debugging
- **Mitigation**: Optimized prompts, error handling

### 6.2 Local Vector Storage

**Decision**: Use local ChromaDB instead of cloud service

**Reasoning**:
- No external dependencies
- Data privacy
- No API costs
- Simple deployment
- Sufficient performance

**Trade-offs**:
- Limited scalability
- No multi-user support
- Local storage required
- **Mitigation**: Suitable for demonstration and single-user scenarios

### 6.3 Synchronous Processing

**Decision**: Process queries synchronously with visual feedback

**Reasoning**:
- Simpler implementation
- Better user experience (see progress)
- Easier debugging
- Appropriate for user volume

**Trade-offs**:
- No concurrent query handling
- User must wait for completion
- **Mitigation**: Fast processing (<10s typical), clear progress indicators

### 6.4 Document Format Support

**Decision**: Support 5 common formats (PDF, DOCX, PPTX, Excel, TXT)

**Reasoning**:
- Covers 95% of business documents
- Libraries available and stable
- Each format has unique challenges
- Demonstrates versatility

**Trade-offs**:
- More complex parsing logic
- Different quality per format
- **Mitigation**: Comprehensive error handling per format

---

## 7. Implementation Details

### 7.1 Code Organization

```
AI-Intern/
├── app.py                  # Entry point, UI logic, session management
├── agentic_rag.py         # Core 5-step agent workflow
├── advanced_retrieval.py  # BM25 + RRF hybrid search
├── vector_db.py           # ChromaDB operations
├── document_processor.py  # Multi-format document parsing
└── requirements.txt       # Dependencies
```

**Separation of Concerns**:
- **Presentation**: Streamlit UI (app.py)
- **Business Logic**: Agentic workflow (agentic_rag.py)
- **Retrieval**: BM25 + RRF hybrid search (advanced_retrieval.py)
- **Data Access**: Vector operations (vector_db.py)
- **Utilities**: Document processing (document_processor.py)

### 7.2 Error Handling Strategy

**Level 1: Document Processing**
```python
try:
    chunks = process_document(file)
except Exception as e:
    log_error(f"Failed to process {file}: {e}")
    continue  # Process other documents
```

**Level 2: Vector Operations**
```python
try:
    results = vector_db.search(query)
except Exception as e:
    return []  # Graceful degradation
```

**Level 3: LLM Calls**
```python
try:
    response = llm.invoke(prompt)
except Exception as e:
    return fallback_response()  # Use simpler approach
```

**Level 4: UI**
```python
try:
    result = agent.process_query(query)
except Exception as e:
    st.error(f"Error: {e}")  # Show user-friendly message
```

### 7.3 Performance Optimizations

1. **Embedding Caching**: Documents embedded once, reused forever
2. **Batch Processing**: Multiple documents processed together
3. **Deduplication**: Remove duplicate chunks before synthesis
4. **Lazy Loading**: Vector DB loaded on first use
5. **Streaming Display**: Show results as they arrive

---

## 8. Limitations

### 8.1 Technical Limitations

1. **LLM Context Window**
   - Limited to ~30,000 tokens
   - Cannot process entire large documents at once
   - **Mitigation**: Chunking strategy

2. **API Rate Limits**
   - Free Gemini API has usage quotas
   - May encounter limits with heavy use
   - **Mitigation**: Error handling, retry logic

3. **Local Storage**
   - ChromaDB stored locally
   - Not suitable for cloud deployment without modification
   - **Mitigation**: Can migrate to hosted Milvus

4. **Single User**
   - No concurrent query support
   - No user authentication
   - **Mitigation**: Designed for demo/single-user scenarios

### 8.2 Functional Limitations

1. **Language Support**
   - Optimized for English
   - May work with other languages but not tested
   - **Improvement**: Add multilingual embeddings

2. **Document Size**
   - Very large documents (>100 pages) slow to process
   - **Improvement**: Background processing, progress bars

3. **Answer Accuracy**
   - Depends on document quality
   - May hallucinate if context insufficient
   - **Mitigation**: Self-verification step alerts user

4. **Complex Tables**
   - Excel/table parsing may lose structure
   - **Improvement**: Specialized table parsing

### 8.3 Security Limitations

1. **No Input Validation**
   - Assumes trusted users
   - **Improvement**: Add file size limits, format validation

2. **No Authentication**
   - Anyone can access
   - **Improvement**: Add user authentication

3. **API Key Handling**
   - Stored in .env file
   - **Improvement**: Use secret management service

---

## 9. Future Enhancements

### 9.1 Short-Term (1-2 months)

1. **Enhanced Retrieval**
   - Cross-encoder reranking model for better precision
   - Query expansion (LLM-generated alternative phrasings)
   - MMR (Maximal Marginal Relevance) for result diversity
   - HyDE (Hypothetical Document Embeddings)

2. **Better UI**
   - Document preview
   - Chunk visualization
   - Interactive source exploration
   - Streaming LLM responses

3. **Better Tokenization**
   - Stemming (PorterStemmer)
   - Stopword removal
   - N-gram support for BM25

### 9.2 Medium-Term (3-6 months)

1. **Milvus Integration**
   - Replace ChromaDB with Milvus
   - Better scalability
   - Production-ready deployment

2. **Self-Hosted LLM**
   - Integrate Ollama
   - Support Llama, Mistral models
   - Reduce API dependency

3. **MCP Server**
   - Build Model Context Protocol server
   - Enable IDE integration
   - Programmatic access

### 9.3 Long-Term (6+ months)

1. **Multi-User Support**
   - User authentication
   - Personal document collections
   - Shared knowledge bases

2. **Advanced Features**
   - Document summarization
   - Citation graph
   - Related question suggestions
   - Query history and analytics

3. **Enterprise Features**
   - RBAC (Role-Based Access Control)
   - Audit logging
   - Compliance features
   - SSO integration

---

## Appendix A: Prompt Engineering

### Query Analysis Prompt
```
You are an intelligent query analyzer. Analyze this user query:

Query: "{query}"

Provide a JSON response with:
1. "complexity": "simple" or "complex"
2. "intent": What does the user want to know?
3. "requires_multi_step": true if query needs multiple retrieval steps
4. "key_concepts": List of main concepts to search for

Respond ONLY with valid JSON.
```

### Answer Synthesis Prompt
```
You are an intelligent AI assistant. Answer the user's question comprehensively based on the provided context.

{conversation_history (last 3 Q&A pairs, if any)}

Context from documents:
{context}

User Question: {query}

Instructions:
1. Be comprehensive and structured with clear sections and lists
2. Extract ALL relevant information from the context
3. Use markdown formatting (headers, bold, lists) for clarity
4. DO NOT include source citations inline (system shows them separately)
5. For "list X" or "what techniques" questions: Extract EVERY instance

Answer:
```

---

## Appendix B: Testing Checklist

- [ ] Upload PDF document
- [ ] Upload DOCX document
- [ ] Upload PPTX document
- [ ] Upload Excel file
- [ ] Upload TXT file
- [ ] Ask simple question
- [ ] Ask complex multi-part question
- [ ] Verify sources shown
- [ ] Check agent analysis display
- [ ] Test conversation continuity
- [ ] Clear database and verify
- [ ] Test error handling (invalid file)
- [ ] Test with no API key
- [ ] Test with no documents uploaded

---

## Conclusion

This Agentic RAG System demonstrates a sophisticated approach to document question-answering by implementing true agentic behavior. The five-step workflow (Analysis → Decomposition → Retrieval → Synthesis → Verification) goes beyond traditional RAG systems to provide intelligent, self-aware responses.

The system balances simplicity with effectiveness, using free and accessible technologies while maintaining production-quality design patterns. Future enhancements can scale the system for enterprise use while maintaining its core agentic capabilities.

---

**Document Version**: 1.0  
**Last Updated**: February 2026  
**Status**: Complete
