import sys
from pathlib import Path

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
sys.path.insert(0, str(project_root))

import os
from logger.custom_logger import CustomLogger
from exceptions.custom_exception import DocumentPortalException
from langchain_openai import AzureOpenAIEmbeddings

from dotenv import load_dotenv
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

logger = CustomLogger().get_logger()

load_dotenv()

class EmbeddingModel(ABC):
    """
    Abstract base class for embedding models.
    Follows Interface Segregation Principle - only essential methods.
    """
    
    @abstractmethod
    def get_embeddings(self) -> Any:
        """
        Returns the embedding model instance.
        
        Returns:
            Any: The embedding model object
        """
        pass
    
    @abstractmethod
    def validate_configuration(self) -> bool:
        """
        Validates the model configuration.
        
        Returns:
            bool: True if configuration is valid
        """
        pass

class AzureOpenAIEmbeddingModel(EmbeddingModel):
    """
    Azure OpenAI implementation of the EmbeddingModel interface.
    Single Responsibility: Only handles Azure OpenAI embedding model creation.
    """
    
    def __init__(
        self,
        api_version: Optional[str] = None,
        api_key: Optional[str] = None,
        azure_endpoint: Optional[str] = None,
        azure_deployment: Optional[str] = None
    ):
        """
        Initialize Azure OpenAI embedding model.
        
        Args:
            api_version: Azure OpenAI API version
            api_key: Azure OpenAI API key
            azure_endpoint: Azure OpenAI endpoint URL
            azure_deployment: Azure deployment name
        """
        self.api_version = api_version or os.getenv("AZURE_OPENAI_EMBEDDING_API_VERSION")
        self.api_key = api_key or os.getenv("AZURE_OPENAI_EMBEDDING_API_KEY")
        self.azure_endpoint = azure_endpoint or os.getenv("AZURE_OPENAI_EMBEDDING_MODEL_ENDPOINT")
        self.azure_deployment = azure_deployment or os.getenv("AZURE_OPENAI_EMBEDDING_MODEL")
        
        self._embeddings = None
        logger.info("AzureOpenAIEmbeddingModel initialized")
    
    def validate_configuration(self) -> bool:
        """
        Validates that all required configuration parameters are present.
        
        Returns:
            bool: True if configuration is valid
            
        Raises:
            DocumentPortalException: If configuration is invalid
        """
        try:
            if not all([self.api_version, self.api_key, self.azure_endpoint, self.azure_deployment]):
                missing_params = []
                if not self.api_version:
                    missing_params.append("api_version")
                if not self.api_key:
                    missing_params.append("api_key")
                if not self.azure_endpoint:
                    missing_params.append("azure_endpoint")
                if not self.azure_deployment:
                    missing_params.append("azure_deployment")
                
                error_msg = f"Missing required configuration parameters: {', '.join(missing_params)}"
                logger.error(error_msg)
                raise DocumentPortalException(error_msg, sys)
            
            logger.info("Azure OpenAI embedding configuration validated successfully")
            return True
        except Exception as e:
            logger.error(f"Configuration validation failed: {str(e)} in model/embedding.py")
            raise DocumentPortalException(e, sys)
    
    def get_embeddings(self) -> AzureOpenAIEmbeddings:
        """
        Returns the Azure OpenAI embedding model instance.
        Implements lazy initialization pattern.
        
        Returns:
            AzureOpenAIEmbeddings: The configured embedding model
            
        Raises:
            DocumentPortalException: If model creation fails
        """
        try:
            if self._embeddings is None:
                self.validate_configuration()
                
                self._embeddings = AzureOpenAIEmbeddings(
                    api_version=self.api_version,
                    api_key=self.api_key,
                    azure_endpoint=self.azure_endpoint,
                    azure_deployment=self.azure_deployment,
                )
                logger.info("Azure OpenAI embeddings model created successfully")
            
            return self._embeddings
        except Exception as e:
            logger.error(f"Failed to create Azure OpenAI embeddings: {str(e)}")
            raise DocumentPortalException(e, sys)

class EmbeddingModelFactory:
    """
    Factory class for creating embedding models.
    Follows Open/Closed Principle - can extend with new providers without modifying existing code.
    """
    _providers = {
        "azure": AzureOpenAIEmbeddingModel,
    }
    
    @classmethod
    def register_provider(cls, provider_name: str, provider_class: type) -> None:
        """
        Register a new embedding provider.
        Allows extension without modification.
        
        Args:
            provider_name: Name of the provider
            provider_class: Class implementing EmbeddingModel
        """
        if not issubclass(provider_class, EmbeddingModel):
            raise ValueError(f"{provider_class} must inherit from EmbeddingModel")
        
        cls._providers[provider_name.lower()] = provider_class
        logger.info(f"Registered new embedding provider: {provider_name}")
    
    @classmethod
    def create_embedding_model(
        cls,
        provider: str = "azure",
        **kwargs
    ) -> EmbeddingModel:
        """
        Factory method to create an embedding model.
        
        Args:
            provider: The provider name (default: "azure")
            **kwargs: Provider-specific configuration parameters
            
        Returns:
            EmbeddingModel: An instance of the requested embedding model
            
        Raises:
            DocumentPortalException: If provider is not supported or creation fails
        """
        try:
            provider_lower = provider.lower()
            
            if provider_lower not in cls._providers:
                error_msg = f"Unsupported embedding provider: {provider}. Available providers: {list(cls._providers.keys())}"
                logger.error(error_msg)
                raise DocumentPortalException(error_msg, sys)
            
            provider_class = cls._providers[provider_lower]
            model = provider_class(**kwargs)
            
            logger.info(f"Created embedding model for provider: {provider}")
            return model
        except Exception as e:
            logger.error(f"Failed to create embedding model: {str(e)}")
            raise DocumentPortalException(e, sys)
    
    @classmethod
    def get_available_providers(cls) -> List[str]:
        """
        Get list of available embedding providers.
        
        Returns:
            List[str]: List of provider names
        """
        return list(cls._providers.keys())


# Convenience function for backward compatibility and ease of use
def get_default_embeddings() -> Any:
    """
    Get the default Azure OpenAI embeddings model.
    Convenience function for quick access.
    
    Returns:
        Any: The configured embedding model
    """
    factory = EmbeddingModelFactory()
    model = factory.create_embedding_model(provider="azure")
    return model.get_embeddings()

if __name__ == "__main__":
    # Example usage
    try:
        embedding_model = EmbeddingModelFactory.create_embedding_model(provider="azure")
        embeddings = embedding_model.get_embeddings()
        logger.info("Successfully obtained embeddings model")
    except DocumentPortalException as e:
        logger.error(f"Error obtaining embeddings model: {str(e)}")