import sys
from pathlib import Path

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
sys.path.insert(0, str(project_root))

import os
from logger.custom_logger import CustomLogger
from exceptions.custom_exception import DocumentPortalException
from models.embedding import EmbeddingModel, EmbeddingModelFactory

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

logger = CustomLogger().get_logger()

# Abstract Base Class - Dependency Inversion Principle
class VectorStore(ABC):
    """
    Abstract base class for vector stores.
    Defines the contract for all vector database implementations.
    """
    
    @abstractmethod
    def create_from_documents(self, documents: List[Document], embedding_model: Any) -> Any:
        """
        Create vector store from documents.
        
        Args:
            documents: List of Document objects to embed
            embedding_model: Embedding model to use
            
        Returns:
            Any: Vector store instance
        """
        pass
    
    @abstractmethod
    def create_from_texts(self, texts: List[str], embedding_model: Any, metadatas: Optional[List[Dict]] = None) -> Any:
        """
        Create vector store from text chunks.
        
        Args:
            texts: List of text chunks
            embedding_model: Embedding model to use
            metadatas: Optional metadata for each text
            
        Returns:
            Any: Vector store instance
        """
        pass
    
    @abstractmethod
    def save(self, vector_store: Any, destination_path: str) -> None:
        """
        Save vector store to disk.
        
        Args:
            vector_store: The vector store instance to save
            destination_path: Path where to save the vector store
        """
        pass
    
    @abstractmethod
    def load(self, source_path: str, embedding_model: Any) -> Any:
        """
        Load vector store from disk.
        
        Args:
            source_path: Path to load the vector store from
            embedding_model: Embedding model to use
            
        Returns:
            Any: Loaded vector store instance
        """
        pass


# Concrete Implementation - Single Responsibility Principle
class FAISSVectorStore(VectorStore):
    """
    FAISS implementation of the VectorStore interface.
    Single Responsibility: Only handles FAISS vector database operations.
    """
    
    def __init__(self):
        """Initialize FAISS vector store handler."""
        logger.info("FAISSVectorStore initialized")
    
    def create_from_documents(self, documents: List[Document], embedding_model: Any) -> FAISS:
        """
        Create FAISS vector store from documents.
        
        Args:
            documents: List of Document objects to embed
            embedding_model: Embedding model to use
            
        Returns:
            FAISS: FAISS vector store instance
            
        Raises:
            DocumentPortalException: If creation fails
        """
        try:
            if not documents:
                raise ValueError("Documents list cannot be empty")
            
            logger.info(f"Creating FAISS vector store from {len(documents)} documents")
            vector_store = FAISS.from_documents(documents, embedding_model)
            logger.info("FAISS vector store created successfully from documents")
            
            return vector_store
        except Exception as e:
            logger.error(f"Failed to create FAISS vector store from documents: {str(e)}")
            raise DocumentPortalException(e, sys)
    
    def create_from_texts(
        self, 
        texts: List[str], 
        embedding_model: Any, 
        metadatas: Optional[List[Dict]] = None
    ) -> FAISS:
        """
        Create FAISS vector store from text chunks.
        Implements batch processing for large datasets.
        
        Args:
            texts: List of text chunks
            embedding_model: Embedding model to use
            metadatas: Optional metadata for each text
            
        Returns:
            FAISS: FAISS vector store instance
            
        Raises:
            DocumentPortalException: If creation fails
        """
        try:
            if not texts:
                raise ValueError("Texts list cannot be empty")
            
            if metadatas and len(metadatas) != len(texts):
                raise ValueError("Metadatas length must match texts length")
            
            logger.info(f"Creating FAISS vector store from {len(texts)} text chunks")
            
            # Batch processing for large datasets
            batch_size = 100  # Process 100 chunks at a time
            vector_store = None
            
            for i in range(0, len(texts), batch_size):
                batch_end = min(i + batch_size, len(texts))
                batch_texts = texts[i:batch_end]
                batch_metadatas = metadatas[i:batch_end] if metadatas else None
                
                logger.info(f"Processing batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}: chunks {i+1}-{batch_end}/{len(texts)}")
                
                if vector_store is None:
                    # Create initial vector store with first batch
                    vector_store = FAISS.from_texts(batch_texts, embedding_model, metadatas=batch_metadatas)
                    logger.info(f"Created initial vector store with {len(batch_texts)} chunks")
                else:
                    # Add subsequent batches to existing vector store
                    batch_store = FAISS.from_texts(batch_texts, embedding_model, metadatas=batch_metadatas)
                    vector_store.merge_from(batch_store)
                    logger.info(f"Merged batch {i//batch_size + 1} into vector store")
            
            logger.info(f"FAISS vector store created successfully from {len(texts)} texts")
            
            return vector_store
        except Exception as e:
            logger.error(f"Failed to create FAISS vector store from texts: {str(e)}")
            raise DocumentPortalException(e, sys)
    
    def save(self, vector_store: FAISS, destination_path: str) -> None:
        """
        Save FAISS vector store to disk.
        
        Args:
            vector_store: The FAISS vector store instance to save
            destination_path: Path where to save the vector store
            
        Raises:
            DocumentPortalException: If save operation fails
        """
        try:
            # Create directory if it doesn't exist
            Path(destination_path).parent.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Saving FAISS vector store to: {destination_path}")
            vector_store.save_local(destination_path)
            logger.info(f"FAISS vector store saved successfully to: {destination_path}")
        except Exception as e:
            logger.error(f"Failed to save FAISS vector store: {str(e)}")
            raise DocumentPortalException(e, sys)
    
    def load(self, source_path: str, embedding_model: Any) -> FAISS:
        """
        Load FAISS vector store from disk.
        
        Args:
            source_path: Path to load the vector store from
            embedding_model: Embedding model to use
            
        Returns:
            FAISS: Loaded FAISS vector store instance
            
        Raises:
            DocumentPortalException: If load operation fails
        """
        try:
            if not Path(source_path).exists():
                raise FileNotFoundError(f"Vector store not found at: {source_path}")
            
            logger.info(f"Loading FAISS vector store from: {source_path}")
            vector_store = FAISS.load_local(
                source_path, 
                embedding_model,
                allow_dangerous_deserialization=True
            )
            logger.info(f"FAISS vector store loaded successfully from: {source_path}")
            
            return vector_store
        except Exception as e:
            logger.error(f"Failed to load FAISS vector store: {str(e)}")
            raise DocumentPortalException(e, sys)


# Service Class - Single Responsibility Principle
# Orchestrates the embedding and persistence pipeline
class EmbedAndPersistService:
    """
    Service class for embedding chunks and persisting to vector store.
    Single Responsibility: Orchestrates the embedding and persistence workflow.
    Follows Dependency Injection pattern.
    """
    
    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_store: VectorStore
    ):
        """
        Initialize the embed and persist service.
        
        Args:
            embedding_model: The embedding model to use
            vector_store: The vector store implementation to use
        """
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self._embeddings = None
        logger.info("EmbedAndPersistService initialized")
    
    def _get_embeddings(self) -> Any:
        """
        Get the embeddings model instance (lazy initialization).
        
        Returns:
            Any: Embeddings model instance
        """
        if self._embeddings is None:
            self._embeddings = self.embedding_model.get_embeddings()
        return self._embeddings
    
    def embed_texts(
        self,
        chunks: List[str],
        metadatas: Optional[List[Dict]] = None
    ) -> Any:
        """
        Embed text chunks and create vector store.
        
        Args:
            chunks: List of text chunks to embed
            metadatas: Optional metadata for each chunk
            
        Returns:
            Any: Vector store instance
            
        Raises:
            DocumentPortalException: If embedding fails
        """
        try:
            if not chunks:
                raise ValueError("Chunks list cannot be empty")
            
            logger.info(f"Starting embedding process for {len(chunks)} chunks")
            embeddings = self._get_embeddings()
            vector_store = self.vector_store.create_from_texts(chunks, embeddings, metadatas)
            logger.info("Successfully embedded chunks and created vector store")
            
            return vector_store
        except Exception as e:
            logger.error(f"Failed to embed texts: {str(e)}")
            raise DocumentPortalException(e, sys)
    
    def embed_documents(self, documents: List[Document]) -> Any:
        """
        Embed documents and create vector store.
        
        Args:
            documents: List of Document objects to embed
            
        Returns:
            Any: Vector store instance
            
        Raises:
            DocumentPortalException: If embedding fails
        """
        try:
            if not documents:
                raise ValueError("Documents list cannot be empty")
            
            logger.info(f"Starting embedding process for {len(documents)} documents")
            embeddings = self._get_embeddings()
            vector_store = self.vector_store.create_from_documents(documents, embeddings)
            logger.info("Successfully embedded documents and created vector store")
            
            return vector_store
        except Exception as e:
            logger.error(f"Failed to embed documents: {str(e)}")
            raise DocumentPortalException(e, sys)
    
    def embed_and_save(
        self,
        chunks: List[str],
        destination_path: str,
        metadatas: Optional[List[Dict]] = None
    ) -> None:
        """
        Embed text chunks and save vector store to specified destination.
        
        Args:
            chunks: List of text chunks to embed
            destination_path: Path where to save the vector store
            metadatas: Optional metadata for each chunk
            
        Raises:
            DocumentPortalException: If process fails
        """
        try:
            logger.info(f"Starting embed and save process for {len(chunks)} chunks")
            vector_store = self.embed_texts(chunks, metadatas)
            self.vector_store.save(vector_store, destination_path)
            logger.info(f"Successfully embedded and saved vector store to: {destination_path}")
        except Exception as e:
            logger.error(f"Failed to embed and save: {str(e)}")
            raise DocumentPortalException(e, sys)
    
    def embed_documents_and_save(
        self,
        documents: List[Document],
        destination_path: str
    ) -> None:
        """
        Embed documents and save vector store to specified destination.
        
        Args:
            documents: List of Document objects to embed
            destination_path: Path where to save the vector store
            
        Raises:
            DocumentPortalException: If process fails
        """
        try:
            logger.info(f"Starting embed and save process for {len(documents)} documents")
            vector_store = self.embed_documents(documents)
            self.vector_store.save(vector_store, destination_path)
            logger.info(f"Successfully embedded and saved vector store to: {destination_path}")
        except Exception as e:
            logger.error(f"Failed to embed documents and save: {str(e)}")
            raise DocumentPortalException(e, sys)
    
    def load_vector_store(self, source_path: str) -> Any:
        """
        Load a previously saved vector store.
        
        Args:
            source_path: Path to load the vector store from
            
        Returns:
            Any: Loaded vector store instance
            
        Raises:
            DocumentPortalException: If load fails
        """
        try:
            embeddings = self._get_embeddings()
            vector_store = self.vector_store.load(source_path, embeddings)
            return vector_store
        except Exception as e:
            logger.error(f"Failed to load vector store: {str(e)}")
            raise DocumentPortalException(e, sys)


# Factory Class - Open/Closed Principle
class VectorStoreFactory:
    """
    Factory class for creating vector store instances.
    Follows Open/Closed Principle - can extend with new vector stores.
    """
    
    _stores = {
        "faiss": FAISSVectorStore,
    }
    
    @classmethod
    def register_store(cls, store_name: str, store_class: type) -> None:
        """
        Register a new vector store implementation.
        
        Args:
            store_name: Name of the vector store
            store_class: Class implementing VectorStore
        """
        if not issubclass(store_class, VectorStore):
            raise ValueError(f"{store_class} must inherit from VectorStore")
        
        cls._stores[store_name.lower()] = store_class
        logger.info(f"Registered new vector store: {store_name}")
    
    @classmethod
    def create_vector_store(cls, store_type: str = "faiss") -> VectorStore:
        """
        Factory method to create a vector store.
        
        Args:
            store_type: The type of vector store (default: "faiss")
            
        Returns:
            VectorStore: An instance of the requested vector store
            
        Raises:
            DocumentPortalException: If store type is not supported
        """
        try:
            store_type_lower = store_type.lower()
            
            if store_type_lower not in cls._stores:
                error_msg = f"Unsupported vector store: {store_type}. Available stores: {list(cls._stores.keys())}"
                logger.error(error_msg)
                raise DocumentPortalException(error_msg, sys)
            
            store_class = cls._stores[store_type_lower]
            store = store_class()
            
            logger.info(f"Created vector store: {store_type}")
            return store
        except Exception as e:
            logger.error(f"Failed to create vector store: {str(e)}")
            raise DocumentPortalException(e, sys)
    
    @classmethod
    def get_available_stores(cls) -> List[str]:
        """
        Get list of available vector stores.
        
        Returns:
            List[str]: List of store names
        """
        return list(cls._stores.keys())


# Convenience function for quick setup
def create_embed_and_persist_service(
    embedding_provider: str = "azure",
    vector_store_type: str = "faiss",
    **embedding_kwargs
) -> EmbedAndPersistService:
    """
    Convenience function to create a fully configured EmbedAndPersistService.
    
    Args:
        embedding_provider: Embedding provider to use (default: "azure")
        vector_store_type: Vector store type to use (default: "faiss")
        **embedding_kwargs: Additional kwargs for embedding model
        
    Returns:
        EmbedAndPersistService: Configured service instance
    """
    embedding_model = EmbeddingModelFactory.create_embedding_model(
        provider=embedding_provider,
        **embedding_kwargs
    )
    vector_store = VectorStoreFactory.create_vector_store(store_type=vector_store_type)
    
    service = EmbedAndPersistService(
        embedding_model=embedding_model,
        vector_store=vector_store
    )
    
    return service


if __name__ == "__main__":
    from config.settings_loader import load_config
    config = load_config("config/config.yaml")
    # Example usage
    try:
        # Create service
        service = create_embed_and_persist_service()
        
        # Sample chunks
        sample_chunks = [
            "This is the first chunk of text about AI.",
            "This is the second chunk discussing machine learning.",
            "The third chunk covers natural language processing.",
        ]
        
        # Specify your destination path
        destination = config["vector_database"]["persist_directory"]
        
        # Embed and save
        service.embed_and_save(
            chunks=sample_chunks,
            destination_path=destination,
        )
        
        logger.info(f"Vector store saved to: {destination}")
        
        # Load the vector store
        loaded_store = service.load_vector_store(destination)
        logger.info("Vector store loaded successfully")
        
        # Perform a similarity search
        results = loaded_store.similarity_search("AI and machine learning", k=2)
        for i, result in enumerate(results):
            logger.info(f"Result {i+1}: {result.page_content}")
            
    except DocumentPortalException as e:
        logger.error(f"Error in embed and persist pipeline: {str(e)}")

