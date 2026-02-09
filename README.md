# Agentic RAG System

An intelligent document Q&A system powered by AI agents with multi-step reasoning and advanced retrieval techniques.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.54%2B-red)](https://streamlit.io/)

> Upload documents → Ask questions → Get intelligent answers with source citations

---

## Features

### Core Capabilities

- **Multi-Format Document Support** - PDF, DOCX, PPTX, Excel, TXT
- **Agentic Workflow** - 5-step intelligent reasoning process
- **Vector Database** - ChromaDB for semantic search
- **LLM Integration** - Google Gemini 2.5 Flash
- **Clean UI** - Streamlit-based conversational interface
- **Session Management** - Track and auto-cleanup documents
- **Error Handling** - Comprehensive try-catch with graceful fallbacks

### Bonus: Advanced Retrieval

- **BM25 Keyword Search** - Term frequency based document matching
- **Hybrid Search** - Combines BM25 (keyword) + Vector (semantic) search
- **Smart Strategy Selection** - Auto-selects best strategy based on query type

### Agentic Workflow (5 Steps)

1. **Query Analysis** - Understand user intent and complexity
2. **Query Decomposition** - Break complex queries into sub-queries
3. **Intelligent Retrieval** - Multi-strategy document search
4. **Answer Synthesis** - Generate comprehensive, structured answers
5. **Self-Verification** - Validate answer quality and confidence

---

## System Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    Streamlit UI                          │
│              (Chat Interface + Controls)                 │
└────────────────────┬─────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────────┐
│                  Agentic RAG Engine                      │
│  ┌────────────────────────────────────────────────────┐  │
│  │  1. Query Analysis      → Intent Recognition     │  │
│  │  2. Query Decomposition → Sub-query Generation   │  │
│  │  3. Advanced Retrieval  → Hybrid Search          │  │
│  │  4. Answer Synthesis    → Context Integration    │  │
│  │  5. Self-Verification   → Confidence Scoring     │  │
│  └────────────────────────────────────────────────────┘  │
└──────────┬─────────────────────────┬────────────────────┘
           │                         │
           ▼                         ▼
   ┌───────────────┐         ┌──────────────────┐
   │  Google       │         │  Vector Database │
   │  Gemini       │         │    (ChromaDB)    │
   │  2.5 Flash    │         │  + BM25 Index    │
   └───────────────┘         └────────┬─────────┘
                                      │
                                      ▼
                             ┌─────────────────────┐
                             │ Document Processor  │
                             │  PDF│DOCX│PPTX│XL   │
                             └─────────────────────┘
```

---

## Quick Start

### Prerequisites

- Python 3.8+
- Google Gemini API Key - [Get it FREE here](https://makersuite.google.com/app/apikey)

### Installation

```bash
# Clone and navigate
git clone https://github.com/P-Saroha/AI-Agent-documents.git
cd AI-Agent-documents

# Create virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_actual_gemini_api_key_here
```

### Run

```bash
streamlit run app.py
```

App opens at: http://localhost:8501

---

## Usage

1. **API key** auto-loads from `.env` on startup
2. **Upload documents** via sidebar (PDF, DOCX, PPTX, Excel, TXT)
3. **Ask questions** in the chat box
4. **View sources** in expandable sections below answers
5. **Toggle Advanced Mode** for hybrid search (BM25 + Vector)

### Retrieval Strategies

| Strategy | Description |
|----------|-------------|
| `auto` | Smart selection based on query type (recommended) |
| `hybrid` | BM25 + Vector search combined |
| `vector` | Pure semantic search |
| `bm25` | Pure keyword search |

---

## Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **LLM** | Google Gemini 2.5 Flash | Natural language understanding & generation |
| **Vector DB** | ChromaDB | Persistent vector storage & semantic search |
| **Embeddings** | HuggingFace all-MiniLM-L6-v2 | Local CPU-optimized text embeddings (384-dim) |
| **Keyword Search** | BM25 (rank-bm25) | Hybrid retrieval |
| **Framework** | LangChain | RAG pipeline orchestration |
| **UI** | Streamlit | Interactive web interface |
| **Doc Parsing** | pypdf, python-docx, python-pptx, openpyxl | Multi-format document support |

---

## Project Structure

```
AI-Agent-documents/
├── app.py                    # Streamlit UI, session management
├── agentic_rag.py            # 5-step agentic workflow engine
├── advanced_retrieval.py     # BM25 + hybrid search (bonus)
├── document_processor.py     # Multi-format document ingestion
├── vector_db.py              # ChromaDB wrapper
├── requirements.txt          # Python dependencies
├── .env                      # API keys (not in git)
├── SYSTEM_DESIGN.md          # Architecture & design decisions
├── sample_data/              # Sample test documents
├── uploads/                  # User uploads (auto-created)
└── chroma_db/                # Vector database (auto-created)
```

---

## Configuration Options

### Chunking (`document_processor.py`)

```python
chunk_size = 1000        # Characters per chunk
chunk_overlap = 200      # Overlap for context continuity
```

### LLM (`app.py`)

```python
model = "gemini-2.5-flash"
temperature = 0.3
```

### Retrieval (`agentic_rag.py`)

```python
num_results = 8          # Documents per sub-query
max_chunks = 10          # Chunks for answer synthesis
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| API Key Error | Verify key in `.env`, no extra spaces, generate new key if expired |
| Module not found | Activate venv, run `pip install -r requirements.txt` |
| No content extracted | Check file format is supported, verify file isn't corrupted |
| Rate limit (429) | Wait 60 seconds, system auto-retries with backoff |
| Database errors | Click "Clear All" in sidebar or delete `chroma_db/` folder |

---

## License

This project is open source and available for educational purposes.

---

<div align="center">

**Agentic RAG System** | 2026

[![GitHub stars](https://img.shields.io/github/stars/P-Saroha/AI-Agent-documents?style=social)](https://github.com/P-Saroha/AI-Agent-documents)

</div>
