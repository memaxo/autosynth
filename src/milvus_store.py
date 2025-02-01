from pathlib import Path
from typing import List, Dict, Any, Optional
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection
import numpy as np
from langchain_core.documents import Document
from langchain_milvus import Milvus
from langchain_community.embeddings import HuggingFaceEmbeddings

class MilvusStore:
    """Vector store using Milvus Lite for local storage"""
    
    def __init__(
        self,
        db_path: str = "./milvus_db",
        collection_name: str = "autosynth_embeddings",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        """
        Initialize Milvus Lite store
        
        Args:
            db_path: Path to store Milvus Lite database
            collection_name: Name of the collection
            embedding_model: HuggingFace model to use for embeddings
        """
        self.db_path = Path(db_path)
        self.collection_name = collection_name
        
        # Create database directory if it doesn't exist
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize embedding function
        self.embedding_function = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Initialize Milvus Lite store
        self.vector_store = Milvus(
            embedding_function=self.embedding_function,
            collection_name=collection_name,
            connection_args={"uri": str(self.db_path)},
        )
        
        # Track if store is initialized
        self._initialized = False
    
    def initialize(self):
        """Initialize the vector store if not already done"""
        if not self._initialized:
            # Create collection with default schema
            self.vector_store.create_collection()
            self._initialized = True
    
    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> List[str]:
        """
        Add texts and their metadata to the vector store
        
        Args:
            texts: List of text strings to add
            metadatas: Optional list of metadata dicts
            
        Returns:
            List of IDs for the added texts
        """
        self.initialize()
        return self.vector_store.add_texts(texts=texts, metadatas=metadatas)
    
    def add_documents(
        self,
        documents: List[Document],
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """
        Add LangChain documents to the vector store
        
        Args:
            documents: List of Document objects
            ids: Optional list of IDs for the documents
            
        Returns:
            List of IDs for the added documents
        """
        self.initialize()
        return self.vector_store.add_documents(documents=documents, ids=ids)
    
    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Perform similarity search
        
        Args:
            query: Query text
            k: Number of results to return
            filter: Optional filter dict
            
        Returns:
            List of similar documents
        """
        self.initialize()
        return self.vector_store.similarity_search(
            query=query,
            k=k,
            filter=filter
        )
    
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[tuple[Document, float]]:
        """
        Perform similarity search with scores
        
        Args:
            query: Query text
            k: Number of results to return
            filter: Optional filter dict
            
        Returns:
            List of (document, score) tuples
        """
        self.initialize()
        return self.vector_store.similarity_search_with_score(
            query=query,
            k=k,
            filter=filter
        )
    
    async def clear(self):
        """Clear all data from the vector store"""
        if self._initialized:
            await self.vector_store.delete_collection()
            self._initialized = False
    
    def __len__(self) -> int:
        """Get number of documents in store"""
        if not self._initialized:
            return 0
        return len(self.vector_store)