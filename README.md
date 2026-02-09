# 🤖 Agentic RAG System

An intelligent document Question-Answering system powered by AI agents that can understand, reason, and retrieve information from multiple document types.

## 🌟 Features

### Core Capabilities
- **🤖 Agentic Workflow**: Intelligent query processing with multi-step reasoning
- **📚 Multi-Format Support**: PDF, DOCX, PPTX, Excel, and TXT files
- **🔍 Smart Retrieval**: Context-aware document search using vector embeddings
- **💬 Conversational Interface**: Clean UI built with Streamlit
- **🧠 Query Analysis**: Automatic query complexity analysis and decomposition
- **✅ Self-Verification**: Agent validates its own answers for quality assurance

### Agentic Behavior Highlights
The system demonstrates true agentic behavior through:

1. **Query Analysis**: Understands user intent and query complexity
2. **Query Decomposition**: Breaks complex questions into manageable sub-queries
3. **Dynamic Retrieval Strategy**: Adapts search approach based on query type
4. **Answer Synthesis**: Combines information from multiple sources intelligently
5. **Self-Reflection**: Verifies answer quality and provides confidence scores

## 🏗️ System Architecture

```
┌─────────────┐
│   User UI   │ (Streamlit)
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│  Agentic RAG    │ ◄── Intelligent Query Processing
│     Engine      │     • Query Analysis
└────────┬────────┘     • Decomposition
         │              • Strategy Selection
         │              • Self-Verification
         │
    ┌────┴────┐
    ▼         ▼
┌────────┐ ┌──────────────┐
│  LLM   │ │ Vector DB    │
│(Gemini)│ │ (ChromaDB)   │
└────────┘ └──────────────┘
              ▲
              │
        ┌─────┴─────┐
        │ Document  │
        │ Processor │
        └───────────┘
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Google Gemini API Key ([Get it free here](https://makersuite.google.com/app/apikey))

### Installation

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd AI-Intern
```

2. **Create and activate virtual environment**
```bash
# Create environment
python -m venv venv

# Activate it
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Add your API key**

Open the `.env` file and replace `your_google_api_key_here` with your actual API key:
```
GOOGLE_API_KEY=AIzaSyD...your_actual_key
```

### Running the Application

```bash
streamlit run app.py
```

The app will open automatically at `http://localhost:8501`

## 📖 How to Use

### Step 1: Configure API Key
- Enter your Google Gemini API key in the sidebar
- The system will automatically initialize

### Step 2: Upload Documents
- Click "Browse files" in the sidebar
- Select one or more documents (PDF, DOCX, PPTX, Excel, TXT)
- Click "Process Documents"
- Wait for processing to complete

### Step 3: Ask Questions
- Type your question in the chat input
- The agent will:
  - Analyze your query
  - Retrieve relevant information
  - Generate a comprehensive answer
  - Verify the answer quality
- View sources and agent analysis in expandable sections

### Example Questions
- "What are the main types of Machine Learning?"
- "Explain the difference between supervised and unsupervised learning"
- "What are the challenges in AI development?"
- "How does deep learning work?"

## 🛠️ Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **LLM** | Google Gemini 2.0 Flash | Natural language understanding and generation |
| **Vector Database** | ChromaDB | Document storage and semantic search |
| **Embeddings** | Google Embedding-001 | Text vectorization |
| **Framework** | LangChain | RAG pipeline orchestration |
| **UI** | Streamlit | User interface |
| **Document Processing** | PyPDF, python-docx, python-pptx, openpyxl | Multi-format document parsing |

## 📁 Project Structure

```
AI-Intern/
├── app.py                    # Main Streamlit application
├── agentic_rag.py           # Agentic RAG engine (core logic)
├── document_processor.py    # Document ingestion module
├── vector_db.py             # Vector database manager
├── requirements.txt         # Python dependencies
├── .env                     # Environment variables (API key)
├── .gitignore              # Git ignore file
├── README.md               # This file
├── SYSTEM_DESIGN.md        # System design document
├── sample_data/            # Sample documents for testing
│   ├── ai_overview.txt
│   └── machine_learning.txt
└── uploads/                # Uploaded documents directory (auto-created)
```

## 🧠 Agentic Workflow Explained

The system implements a sophisticated agentic workflow:

### 1. Query Analysis
```python
{
  "complexity": "simple" | "complex",
  "intent": "What user wants to know",
  "requires_multi_step": true | false,
  "key_concepts": ["concept1", "concept2"]
}
```

### 2. Query Decomposition
- Complex queries are broken into simpler sub-queries
- Each sub-query is processed independently
- Results are combined for comprehensive answers

### 3. Intelligent Retrieval
- Dynamic number of results based on query complexity
- Relevance scoring and filtering
- Deduplication of retrieved documents

### 4. Answer Synthesis
- Context-aware answer generation
- Source citation
- Structured response formatting

### 5. Self-Verification
```python
{
  "is_adequate": true | false,
  "confidence": 0-100,
  "suggestion": "Improvement suggestions"
}
```

## 🎯 Key Features Explained

### Data Engineering
- **Chunking Strategy**: Documents are split into 1000-character chunks with 200-character overlap
- **Metadata Preservation**: File type, source, and structure information retained
- **Error Handling**: Comprehensive error handling for each document type

### Vector Database
- **Persistent Storage**: Documents stored in local ChromaDB
- **Semantic Search**: Cosine similarity for relevance matching
- **Scalable**: Can handle large document collections

### LLM Integration
- **Temperature Control**: Set to 0.3 for consistent, focused responses
- **Prompt Engineering**: Structured prompts for each agent step
- **Context Management**: Efficient context window usage

## 📊 Sample Data

The repository includes sample documents in the `sample_data/` folder:
- `ai_overview.txt`: Introduction to Artificial Intelligence
- `machine_learning.txt`: Comprehensive ML guide

Use these to test the system before uploading your own documents.

## 🔧 Configuration Options

### Document Processing
Edit in `document_processor.py`:
```python
chunk_size = 1000        # Size of text chunks
chunk_overlap = 200      # Overlap between chunks
```

### Vector Database
Edit in `vector_db.py`:
```python
collection_name = "documents"
persist_directory = "./chroma_db"
```

### LLM Settings
Edit in `app.py`:
```python
temperature = 0.3              # Lower = more focused
model = "gemini-2.0-flash-exp" # Model selection
```

## ⚠️ Limitations

1. **API Rate Limits**: Free Gemini API has rate limits
2. **Document Size**: Very large documents may take time to process
3. **Context Window**: Limited by LLM context window (~30,000 tokens)
4. **Language**: Optimized for English content
5. **Accuracy**: Answers depend on document content quality

## 🐛 Troubleshooting

### "API Key Error"
- Verify your API key is correct
- Check API key has necessary permissions
- Ensure no extra spaces in `.env` file

### "No documents found"
- Upload documents before asking questions
- Check file format is supported
- Verify files are not corrupted

### "Slow Response"
- Large documents take time to process
- Complex queries may take longer
- Check internet connection for API calls

## 🔄 Future Enhancements

- [ ] Support for more document types (CSV, JSON)
- [ ] Advanced retrieval techniques (HyDE, Multi-query)
- [ ] Conversation memory across sessions
- [ ] Document summarization feature
- [ ] Export chat history
- [ ] Milvus integration option
- [ ] Self-hosted LLM support

## 📝 License

This project is open source and available for educational purposes.

## 👥 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues.

## 📧 Contact

For questions or feedback, please open an issue in the repository.

---

**Built with ❤️ for intelligent document understanding**
#   A I - A g e n t - d o c u m e n t s  
 #   A I - A g e n t - d o c u m e n t s  
 