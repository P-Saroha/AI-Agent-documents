# ðŸ¤– Agentic RAG System

**An intelligent document Q&A system powered by AI agents with multi-step reasoning, advanced retrieval techniques, and comprehensive error handling.**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.54+-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**ðŸ’¡ Live Demo:** Upload documents â†’ Ask questions â†’ Get intelligent answers with source citations

---

## âœ¨ Features

### ðŸŽ¯ Core Capabilities
âœ… **Multi-Format Document Support** - PDF, DOCX, PPTX, Excel, TXT  
âœ… **Agentic Workflow** - 5-step intelligent reasoning process  
âœ… **Vector Database** - ChromaDB for semantic search  
âœ… **LLM Integration** - Google Gemini 2.5 Flash  
âœ… **Clean UI** - Streamlit-based conversational interface  

### ðŸŽ **Bonus Features** (Advanced Implementation)
â­ **Advanced Retrieval Techniques**
- Query Expansion (generates alternative phrasings)
- Hybrid Search (BM25 + Vector semantic search)
- MMR Reranking (Maximal Marginal Relevance for diversity)
- Smart Strategy Selection (auto/hybrid/vector/bm25)

â­ **Comprehensive Error Handling**
- Custom exception types for different error scenarios
- Graceful degradation with fallback strategies
- Detailed logging to `agentic_rag.log`
- User-friendly error messages
- Retry mechanism with exponential backoff

â­ **Session Management**
- Track uploaded documents per session
- Auto-remove temporary documents
- Clean database management

### ðŸ¤– Agentic Workflow (5 Steps)

```mermaid
graph LR
    A[User Query] --> B[1. Analysis]
    B --> C[2. Decomposition]
    C --> D[3. Retrieval]
    D --> E[4. Synthesis]
    E --> F[5. Verification]
    F --> G[Answer + Confidence]
```

1. **Query Analysis** - Understand user intent and complexity
2. **Query Decomposition** - Break complex queries into sub-queries
3. **Intelligent Retrieval** - Multi-strategy document search
4. **Answer Synthesis** - Generate comprehensive, structured answers
5. **Self-Verification** - Validate answer quality and confidence

---

## ðŸ—ï¸ System Architecture

```
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚                    Streamlit UI                          â”‚
â”‚              (Chat Interface + Controls)                 â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                     â”‚
                     â–¼
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚                  Agentic RAG Engine                      â”‚
â”‚  â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”   â”‚
â”‚  â”‚  1. Query Analysis      â†’ Intent Recognition     â”‚   â”‚
â”‚  â”‚  2. Query Decomposition â†’ Sub-query Generation   â”‚   â”‚
â”‚  â”‚  3. Advanced Retrieval  â†’ Hybrid Search         â”‚   â”‚
â”‚  â”‚  4. Answer Synthesis    â†’ Context Integration    â”‚   â”‚
â”‚  â”‚  5. Self-Verification   â†’ Confidence Scoring    â”‚   â”‚
â”‚  â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜   â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
          â”‚                            â”‚
          â–¼                            â–¼
   â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”           â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
   â”‚  Google      â”‚           â”‚  Vector Database â”‚
   â”‚  Gemini      â”‚           â”‚    (ChromaDB)    â”‚
   â”‚  2.5 Flash   â”‚           â”‚  + BM25 Index    â”‚
   â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜           â””â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                                       â”‚
                                       â–¼
                              â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
                              â”‚ Document Processor  â”‚
                              â”‚  PDFâ”‚DOCXâ”‚PPTXâ”‚XL  â”‚
                              â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
```

**Key Components:**
- **app.py** - Streamlit frontend with error handling
- **agentic_rag.py** - Core agentic workflow engine
- **advanced_retrieval.py** - Bonus: Query expansion, hybrid search, MMR
- **error_handler.py** - Bonus: Comprehensive error management
- **document_processor.py** - Multi-format document ingestion
- **vector_db.py** - ChromaDB wrapper with session management

---

## ðŸš€ Quick Start

### Prerequisites
- **Python 3.8+** (tested on 3.11.9)
- **Google Gemini API Key** - [Get it FREE here](https://makersuite.google.com/app/apikey)

### Installation (3 steps)

**1ï¸âƒ£ Clone & Navigate**
```bash
git clone https://github.com/P-Saroha/AI-Agent-documents.git
cd AI-Agent-documents
```

**2ï¸âƒ£ Setup Virtual Environment**
```bash
# Create virtual environment
python -m venv venv

# Activate it
venv\Scripts\activate          # Windows
source venv/bin/activate       # Linux/Mac
```

**3ï¸âƒ£ Install Dependencies**
```bash
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root:
```env
GOOGLE_API_KEY=your_actual_gemini_api_key_here
```

**ðŸ’¡ Tip:** Never commit your `.env` file (already in `.gitignore`)

### Run the Application

```bash
streamlit run app.py
```

ðŸŽ‰ **App opens at:** http://localhost:8501

---

## ï¿½ Usage Guide

### Step-by-Step Walkthrough

**1. Initialize System** ðŸ”‘
```
âœ“ API key auto-loaded from .env
âœ“ System initialized!
```

**2. Upload Documents** ðŸ“„
- Sidebar â†’ "Choose files"
- Select: PDF, DOCX, PPTX, Excel, or TXT
- Click "Process Documents"
- Wait for: `âœ“ Successfully processed N documents (X chunks)`

**3. Ask Questions** ðŸ’¬
```
ðŸ’­ Type: "What are the main fine-tuning techniques discussed?"

ðŸ¤– Agent Processing:
  â†’ Analyzing query intent...
  â†’ Breaking into sub-queries...
  â†’ Retrieving relevant chunks...
  â†’ Synthesizing answer...
  â†’ Verifying quality (Confidence: 95%)

ðŸ“ Answer displayed with:
  âœ“ Comprehensive response
  âœ“ Source citations
  âœ“ Agent analysis (expandable)
```

**4. Explore Bonus Features** âš¡
- Toggle "Advanced Mode" â†’ Enable hybrid search
- Select strategy: `auto` | `hybrid` | `vector` | `bm25`
- Enable "Session Mode" â†’ Auto-cleanup temporary docs

### Example Queries

| Query Type | Example |
|------------|---------|
| **Summary** | "Give me a summary of this document" |
| **List** | "What fine-tuning techniques are discussed?" |
| **Explain** | "Explain the transformer architecture" |
| **Compare** | "What's the difference between LoRA and QLoRA?" |
| **Details** | "How does the attention mechanism work?" |

---

## ðŸ› ï¸ Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **ðŸ¤– LLM** | Google Gemini 2.5 Flash | Natural language understanding & generation |
| **ðŸ“Š Vector DB** | ChromaDB | Persistent vector storage & semantic search |
| **ðŸ”¤ Embeddings** | HuggingFace all-MiniLM-L6-v2 | CPU-optimized text embeddings (80MB) |
| **ðŸ” Keyword Search** | BM25 (rank-bm25) | **BONUS:** Hybrid retrieval |
| **ðŸ§© Framework** | LangChain | RAG pipeline orchestration |
| **ðŸŽ¨ UI** | Streamlit 1.54+ | Interactive web interface |
| **ðŸ“„ Parsing** | pypdf, python-docx, python-pptx, openpyxl | Multi-format document support |
| **ðŸ›¡ï¸ Error Handling** | Custom framework | **BONUS:** Comprehensive error management |

### Dependencies

```txt
# Core
streamlit==1.54.0
langchain-google-genai
chromadb
sentence-transformers

# Advanced Features (Bonus)
rank-bm25              # Hybrid search
logging                # Error tracking

# Document Processing  
pypdf, python-docx, python-pptx, openpyxl, pandas
```

---

## ðŸ“ Project Structure

```
AI-Agent-documents/
â”‚
â”œâ”€â”€ ðŸ“± Frontend
â”‚   â””â”€â”€ app.py                      # Streamlit UI with error handling
â”‚
â”œâ”€â”€ ðŸ§  Core Engine
â”‚   â”œâ”€â”€ agentic_rag.py              # 5-step agentic workflow
â”‚   â”œâ”€â”€ document_processor.py       # Multi-format ingestion
â”‚   â””â”€â”€ vector_db.py                # ChromaDB wrapper + session mgmt
â”‚
â”œâ”€â”€ ðŸŽ Bonus Features
â”‚   â”œâ”€â”€ advanced_retrieval.py       # Query expansion, hybrid search, MMR
â”‚   â””â”€â”€ error_handler.py            # Custom exceptions, logging, retry
â”‚
â”œâ”€â”€ ðŸ“„ Configuration
â”‚   â”œâ”€â”€ requirements.txt            # Python dependencies
â”‚   â”œâ”€â”€ .env                        # API keys (not in git)
â”‚   â””â”€â”€ .gitignore                  # Git ignore rules
â”‚
â”œâ”€â”€ ðŸ“š Documentation
â”‚   â”œâ”€â”€ README.md                   # This file
â”‚   â””â”€â”€ SYSTEM_DESIGN.md            # Architecture & design decisions
â”‚
â”œâ”€â”€ ðŸ“‚ Data
â”‚   â”œâ”€â”€ sample_data/                # Sample test documents
â”‚   â”‚   â”œâ”€â”€ ai_overview.txt
â”‚   â”‚   â””â”€â”€ machine_learning.txt
â”‚   â”œâ”€â”€ uploads/                    # User uploads (auto-created)
â”‚   â””â”€â”€ chroma_db/                  # Vector database (auto-created)
â”‚
â””â”€â”€ ðŸ“ Logs
    â””â”€â”€ agentic_rag.log             # Application logs (auto-created)
```

**Core Files Explained:**

| File | Lines | Purpose |
|------|-------|---------|
| `app.py` | ~430 | Main application, UI, session management |
| `agentic_rag.py` | ~290 | Agentic workflow orchestration |
| `advanced_retrieval.py` | ~280 | **BONUS:** Advanced retrieval techniques |
| `error_handler.py` | ~220 | **BONUS:** Error handling framework |
| `document_processor.py` | ~155 | Multi-format document parsing |
| `vector_db.py` | ~150 | ChromaDB operations |

**Total:** ~2,500 lines of code

---

---

## ðŸ”§ Advanced Configuration

### Bonus Features Activation

**Enable Advanced Retrieval:**
```python
# In UI: Toggle "âš¡ Enable Advanced Mode"
# Provides: Query expansion, hybrid search, MMR reranking
```

**Retrieval Strategies:**
- `auto` - Smart selection based on query type (recommended)
- `hybrid` - Best performance (BM25 + Vector search)
- `vector` - Pure semantic search
- `bm25` - Pure keyword search

**Session Mode:**
- Track documents uploaded in current session
- Click "Done (Remove)" to auto-cleanup
- Keeps database clean without manual management

### Customization

**Chunking Strategy** (`document_processor.py`):
```python
chunk_size = 1000        # Characters per chunk
chunk_overlap = 200      # Overlap for context continuity
```

**LLM Settings** (`app.py`):
```python
model = "gemini-2.5-flash"
temperature = 0.3         # Lower = more focused, higher = more creative
```

**Retrieval Configuration** (`agentic_rag.py`):
```python
num_results = 8           # Documents per sub-query
max_chunks = 10           # Chunks for answer synthesis
```

---

## ðŸ› Troubleshooting

| Issue | Solution |
|-------|----------|
| âŒ **API Key Error** | â€¢ Verify key is correct<br>â€¢ Check no extra spaces in `.env`<br>â€¢ Generate new key if expired |
| âš ï¸ **"No module named..."** | â€¢ Activate virtual environment<br>â€¢ Run `pip install -r requirements.txt` |
| ðŸ“„ **"No content extracted"** | â€¢ Check file format is supported<br>â€¢ Verify file isn't corrupted<br>â€¢ Try different document |
| ðŸŒ **Slow Processing** | â€¢ Large PDFs take time<br>â€¢ Enable advanced mode for better results<br>â€¢ Check internet for API calls |
| ðŸ’¾ **Database Errors** | â€¢ Click "Clear All" in sidebar<br>â€¢ Delete `chroma_db/` folder<br>â€¢ Restart application |
| ðŸ” **Poor Retrieval Quality** | â€¢ Enable "Advanced Mode"<br>â€¢ Use "hybrid" strategy<br>â€¢ Upload more relevant documents |

### Logs & Debugging

Check `agentic_rag.log` for detailed error traces:
```bash
tail -f agentic_rag.log  # Live monitoring
```

---

## ðŸ“Š Performance & Limitations

### Performance Metrics
- **Document Processing:** ~5-10 seconds for typical PDF (20 pages)
- **Query Response:** 3-8 seconds depending on complexity
- **Embedding Model:** ~80MB (cached locally after first use)
- **Memory Usage:** ~500MB RAM for typical workload

### Known Limitations

| Limitation | Impact | Workaround |
|------------|--------|------------|
| **API Rate Limits** | 60 requests/min (free tier) | Use smaller queries |
| **Context Window** | ~30K tokens max | Split very large docs |
| **Language** | Optimized for English | May work with other languages |
| **File Size** | Best < 50MB per doc | Split large files |
| **Accuracy** | Depends on doc quality | Use clean, well-formatted docs |

---

## ðŸ”„ Future Enhancements

- [ ] Support for more document types (CSV, JSON)
- [ ] Advanced retrieval techniques (HyDE, Multi-query)
- [ ] Conversation memory across sessions
- [ ] Document summarization feature
- [ ] Export chat history
- [ ] Milvus integration option
- [ ] Self-hosted LLM support

## ðŸ“ License

This project is open source and available for educational purposes.

## ðŸ‘¥ Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues.

## ðŸ“§ Contact

For questions or feedback, please open an issue in the repository.

---

**Built with â¤ï¸ for intelligent document understanding**

<div align="center">

**Built with ❤️ for Intelligent Document Understanding**

⭐ **Star this repo if you found it helpful!**

[![GitHub stars](https://img.shields.io/github/stars/P-Saroha/AI-Agent-documents?style=social)](https://github.com/P-Saroha/AI-Agent-documents/stargazers)

**Agentic RAG System** | 2026

</div>
