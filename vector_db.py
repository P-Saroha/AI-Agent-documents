"""Vector Database Manager using ChromaDB"""
from typing import List, Dict, Optional
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma


class VectorDBManager:
    """ChromaDB vector database manager"""
    
    def __init__(self, api_key: str, collection_name: str = "documents", persist_directory: str = "./chroma_db", embeddings=None):
        """Initialize vector database with local embeddings"""
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        
        # Use provided embeddings or create new ones
        if embeddings is not None:
            self.embeddings = embeddings
        else:
            # Fallback: create new embeddings (for compatibility)
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )
        
        # Initialize ChromaDB
        self.vectorstore = None
    
    def create_or_load_collection(self):
        """Create or load existing collection"""
        try:
            self.vectorstore = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory
            )
            return True
        except Exception as e:
            print(f"Error creating/loading collection: {str(e)}")
            return False
    
    def add_documents(self, documents: List[Document]) -> bool:
        """Add documents to vector database"""
        try:
            if not self.vectorstore:
                self.create_or_load_collection()
            
            self.vectorstore.add_documents(documents)
            return True
        except Exception as e:
            error_msg = str(e)
            print(f"✗ Error adding documents: {error_msg}")
            # Also raise to propagate to UI
            raise Exception(f"Failed to add documents: {error_msg}")
    
    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """Search for similar documents"""
        try:
            if not self.vectorstore:
                self.create_or_load_collection()
            
            results = self.vectorstore.similarity_search(query, k=k)
            return results
        except Exception as e:
            print(f"Error during similarity search: {str(e)}")
            return []
    
    def similarity_search_with_score(self, query: str, k: int = 5) -> List[tuple]:
        """Search with similarity scores"""
        try:
            if not self.vectorstore:
                self.create_or_load_collection()
            
            results = self.vectorstore.similarity_search_with_score(query, k=k)
            return results
        except Exception as e:
            print(f"Error during similarity search: {str(e)}")
            return []
    
    def get_retriever(self, k: int = 5):
        """Get retriever for RAG chain"""
        if not self.vectorstore:
            self.create_or_load_collection()
        
        return self.vectorstore.as_retriever(search_kwargs={"k": k})
    
    def clear_collection(self) -> bool:
        """Clear all documents from collection"""
        try:
            if self.vectorstore:
                self.vectorstore.delete_collection()
                self.create_or_load_collection()
                print("✓ Collection cleared")
                return True
        except Exception as e:

    
    def get_collection_count(self) -> int:
        """Get number of documents in collection"""
        try:
            if not self.vectorstore:
                self.create_or_load_collection()
            
            collection = self.vectorstore._collection
            return collection.count()
        except:
            return 0
    
    def remove_documents_by_source(self, source_paths: List[str]) -> bool:
        """Remove documents by their source file paths"""
        try:
            if not self.vectorstore:
                return False
            
            # Get the ChromaDB collection
            collection = self.vectorstore._collection
            
            # Query all documents
            all_docs = collection.get()
            
            if not all_docs or not all_docs.get('ids'):
                return False
            
            # Find IDs to delete
            ids_to_delete = []
            for i, metadata in enumerate(all_docs.get('metadatas', [])):
                if metadata and metadata.get('source') in source_paths:
                    ids_to_delete.append(all_docs['ids'][i])
            
            # Delete the documents
            if ids_to_delete:
                collection.delete(ids=ids_to_delete)
                print(f"✓ Removed {len(ids_to_delete)} document chunks from {len(source_paths)} files")
                return True
            
            return False
        except Exception as e:
            return False
