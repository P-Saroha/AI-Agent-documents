"""Agentic RAG System - Main Application"""
import streamlit as st
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from document_processor import DocumentProcessor
from vector_db import VectorDBManager
from agentic_rag import AgenticRAG

try:
    from advanced_retrieval import AdvancedRetriever
    ADVANCED_RETRIEVAL_AVAILABLE = True
except ImportError:
    ADVANCED_RETRIEVAL_AVAILABLE = False
    print("Advanced retrieval not available - install rank-bm25")

# Load environment variables
load_dotenv()

# Cache embedding model to prevent reloading
@st.cache_resource
def get_embeddings():
    """Load and cache the embedding model"""
    from langchain_community.embeddings import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )

# Page configuration
st.set_page_config(
    page_title="Agentic RAG System",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        padding: 1rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    .source-box {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables"""
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
    if 'agent' not in st.session_state:
        st.session_state.agent = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'doc_count' not in st.session_state:
        st.session_state.doc_count = 0
    if 'session_doc_ids' not in st.session_state:
        st.session_state.session_doc_ids = []
    if 'session_mode' not in st.session_state:
        st.session_state.session_mode = False
    if 'advanced_mode' not in st.session_state:
        st.session_state.advanced_mode = ADVANCED_RETRIEVAL_AVAILABLE
    if 'retrieval_strategy' not in st.session_state:
        st.session_state.retrieval_strategy = "auto"


def initialize_system(api_key: str):
    """Initialize the RAG system components"""
    try:
        if not api_key or len(api_key) < 20:
            st.error("Invalid API key")
            return False
        
        # Initialize LLM (Gemini 2.5 Flash)
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=api_key,
            temperature=0.3
        )
        
        # Initialize Vector Database with cached embeddings
        embeddings = get_embeddings()
        vector_db = VectorDBManager(
            api_key=api_key,
            collection_name="agentic_rag_docs",
            embeddings=embeddings
        )
        vector_db.create_or_load_collection()
        
        # Initialize Advanced Retriever (bonus feature)
        advanced_retriever = None
        if ADVANCED_RETRIEVAL_AVAILABLE and st.session_state.advanced_mode:
            try:
                advanced_retriever = AdvancedRetriever(llm, vector_db)
            except Exception as e:
                print(f"Advanced retrieval initialization failed: {str(e)}")
        
        # Initialize Agentic RAG
        agent = AgenticRAG(
            llm=llm,
            vector_db=vector_db,
            advanced_retriever=advanced_retriever
        )
        
        st.session_state.agent = agent
        st.session_state.vector_db = vector_db
        st.session_state.doc_processor = DocumentProcessor()
        st.session_state.advanced_retriever = advanced_retriever
        st.session_state.initialized = True
        
        # Get document count
        st.session_state.doc_count = vector_db.get_collection_count()
        
        # Build BM25 index if advanced mode enabled
        if advanced_retriever and st.session_state.doc_count > 0:
            try:
                all_docs = vector_db.vectorstore.get()["documents"]
                if all_docs:
                    from langchain_core.documents import Document
                    docs = [Document(page_content=doc) for doc in all_docs]
                    advanced_retriever.build_bm25_index(docs)
            except Exception as e:
                print(f"Warning: BM25 index build failed: {str(e)}")
        
        return True
    except Exception as e:
        st.error(f"Error initializing system: {str(e)}")
        return False


def main():
    """Main application"""
    
    # Header
    st.markdown('<h1 class="main-header">Agentic RAG System</h1>', unsafe_allow_html=True)
    st.markdown("### Intelligent Document Q&A with AI Agents")
    
    initialize_session_state()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Get API Key from environment
        env_api_key = os.getenv("GOOGLE_API_KEY", "")
        
        # Only show input if no API key in .env
        if not env_api_key or env_api_key == "your_google_api_key_here":
            api_key = st.text_input(
                "Google Gemini API Key",
                type="password",
                help="Get your free API key from https://makersuite.google.com/app/apikey"
            )
        else:
            api_key = env_api_key
            st.success("API Key loaded from .env")
        
        if api_key and api_key != "your_google_api_key_here" and not st.session_state.initialized:
            with st.spinner("Initializing system..."):
                if initialize_system(api_key):
                    st.success("System initialized!")
        
        st.divider()
        
        # Document upload
        if st.session_state.initialized:
            st.header("Upload Documents")
            
            uploaded_files = st.file_uploader(
                "Choose files",
                type=['pdf', 'docx', 'pptx', 'xlsx', 'txt'],
                accept_multiple_files=True,
                help="Supported: PDF, DOCX, PPTX, Excel, TXT"
            )
            
            if uploaded_files and st.button("Process Documents", type="primary"):
                process_documents(uploaded_files)
            
            st.divider()
            
            # Stats
            st.header("Statistics")
            st.metric("Documents in Database", st.session_state.doc_count)
            st.metric("Session Documents", len(st.session_state.session_doc_ids))
            
            # Session mode toggle
            st.session_state.session_mode = st.checkbox(
                "🔄 Session Mode",
                value=st.session_state.session_mode,
                help="Auto-remove documents when session ends or when you click 'Done'"
            )
            
            # Advanced Retrieval (BONUS FEATURE)
            if ADVANCED_RETRIEVAL_AVAILABLE:
                st.divider()
                st.header("Advanced Retrieval")
                
                st.session_state.advanced_mode = st.checkbox(
                    "Enable Advanced Mode",
                    value=st.session_state.advanced_mode,
                    help="Query expansion, hybrid search (BM25 + Vector), MMR reranking"
                )
                
                if st.session_state.advanced_mode:
                    st.session_state.retrieval_strategy = st.selectbox(
                        "Retrieval Strategy",
                        options=["auto", "hybrid", "vector", "bm25"],
                        help="auto: Smart selection | hybrid: Best (Vector+BM25) | vector: Semantic only | bm25: Keyword only"
                    )
                    
                    st.info("**Advanced Features Active:**\n- Query Expansion\n- Hybrid Search\n- MMR Reranking")
            
            st.divider()
            
            # Cleanup buttons
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🗑️ Clear All", type="secondary", use_container_width=True):
                    if st.session_state.vector_db.clear_collection():
                        st.session_state.doc_count = 0
                        st.session_state.chat_history = []
                        st.session_state.session_doc_ids = []
                        st.rerun()
            
            with col2:
                if len(st.session_state.session_doc_ids) > 0:
                    if st.button("Done (Remove)", type="primary", use_container_width=True):
                        # Remove session documents
                        removed = st.session_state.vector_db.remove_documents_by_source(
                            st.session_state.session_doc_ids
                        )
                        if removed:
                            st.session_state.doc_count = st.session_state.vector_db.get_collection_count()
                            st.session_state.session_doc_ids = []
                            st.success("Session documents removed!")
                            st.rerun()
                    st.success("Database cleared!")
                    st.rerun()
    
    # Main content
    # Main content
    if not api_key or api_key == "your_google_api_key_here":
        st.info("👈 Please add your Google Gemini API Key to get started.")
        st.markdown("""
        ### How to get started:
        1. Get a free API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
        2. Add it to the `.env` file: `GOOGLE_API_KEY=your_key_here`
        3. Restart the app
        4. Upload your documents (PDF, DOCX, PPTX, Excel, TXT)
        5. Ask questions about your documents!
        
        ### Features:
        - **Agentic Workflow**: Intelligent query analysis and multi-step reasoning
        - **Multi-format Support**: PDF, Word, PowerPoint, Excel, Text
        - **Smart Retrieval**: Context-aware document search
        - 💬 **Conversational**: Ask follow-up questions
        """)
        return
    
    if not st.session_state.initialized:
        st.warning("Initializing system...")
        return
    
    # Chat interface
    st.header("💬 Ask Questions")
    
    # Display chat history
    for chat in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(chat['query'])
        with st.chat_message("assistant"):
            st.write(chat['answer'])
            if chat.get('sources'):
                with st.expander("View Sources"):
                    for i, source in enumerate(chat['sources'], 1):
                        st.markdown(f"**{i}.** {Path(source).name}", unsafe_allow_html=True)
    
    # Query input
    query = st.chat_input("Ask a question about your documents...")
    
    if query:
        if st.session_state.doc_count == 0:
            st.warning("⚠️ Please upload some documents first!")
        else:
            process_query(query)


def process_documents(uploaded_files):
    """Process uploaded documents with comprehensive error handling"""
    try:
        # Create uploads directory
        upload_dir = Path("uploads")
        upload_dir.mkdir(exist_ok=True)
        
        # Save and process files
        file_paths = []
        for uploaded_file in uploaded_files:
            try:
                file_path = upload_dir / uploaded_file.name
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                file_paths.append(str(file_path))
            except Exception as e:
                st.warning(f"⚠️ Failed to save {uploaded_file.name}: {str(e)}")
        
        if not file_paths:
            st.error("No files could be saved")
            return
        
        # Process documents
        with st.spinner(f"Processing {len(file_paths)} documents..."):
            try:
                # Extract and chunk documents
                chunks = st.session_state.doc_processor.process_multiple_documents(file_paths)
                
                if chunks:
                    # Add to vector database
                    st.session_state.vector_db.add_documents(chunks)
                    st.session_state.doc_count = st.session_state.vector_db.get_collection_count()
                    
                    # Rebuild BM25 index if advanced mode enabled
                    if st.session_state.get('advanced_retriever'):
                        try:
                            all_docs = st.session_state.vector_db.vectorstore.get()["documents"]
                            if all_docs:
                                from langchain_core.documents import Document
                                docs = [Document(page_content=doc) for doc in all_docs]
                                st.session_state.advanced_retriever.build_bm25_index(docs)
                        except Exception as e:
                            print(f"Warning: BM25 rebuild failed: {str(e)}")
                    
                    # Track session documents
                    for file_path in file_paths:
                        if file_path not in st.session_state.session_doc_ids:
                            st.session_state.session_doc_ids.append(file_path)
                    
                    st.success(f"Successfully processed {len(file_paths)} documents ({len(chunks)} chunks)")
                    if st.session_state.session_mode:
                        st.info("🔄 Session Mode: Click 'Done (Remove)' to auto-remove these documents")
                    if st.session_state.get('advanced_mode'):
                        st.info("Advanced retrieval features active")
                    st.rerun()
                else:
                    st.error("No content extracted from documents")
            except Exception as e:
                handle_streamlit_errors(e, "Document Processing")
                
    except Exception as e:
        handle_streamlit_errors(e, "File Upload")


def process_query(query: str):
    """Process user query with agentic RAG and comprehensive error handling"""
    # Display user message
    with st.chat_message("user"):
        st.write(query)
    
    # Process with agent
    with st.chat_message("assistant"):
        with st.spinner("🤖 Agent is thinking..."):
            try:
                # Get retrieval strategy
                strategy = st.session_state.get('retrieval_strategy', 'auto')
                
                result = st.session_state.agent.process_query(
                    query,
                    strategy=strategy
                )
                
                # Display answer
                st.write(result['answer'])
                
                # Display analysis
                with st.expander("🧠 Agent Analysis"):
                    analysis_data = {
                        "Intent": result['analysis'].get('intent', 'N/A'),
                        "Complexity": result['analysis'].get('complexity', 'N/A'),
                        "Confidence": f"{result['verification'].get('confidence', 'N/A')}%",
                        "Sources Used": result['num_sources']
                    }
                    
                    if st.session_state.get('advanced_mode'):
                        analysis_data["Retrieval Strategy"] = strategy
                    
                    st.json(analysis_data)
                
                # Display sources
                if result['sources']:
                    with st.expander("📚 Sources"):
                        for i, doc in enumerate(result['sources'], 1):
                            source_name = Path(doc.metadata.get('source', 'Unknown')).name
                            st.markdown(f"**{i}. {source_name}**")
                            st.markdown(f"_{doc.page_content[:200]}..._")
                            st.divider()
                
                # Save to history
                st.session_state.chat_history.append({
                    'query': query,
                    'answer': result['answer'],
                    'sources': [doc.metadata.get('source', 'Unknown') for doc in result['sources']]
                })
                
            except Exception as e:
                st.error(f"Error processing query: {str(e)}")


if __name__ == "__main__":
    main()
