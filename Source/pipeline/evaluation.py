import sys
from pathlib import Path

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
sys.path.insert(0, str(project_root))

from logger.custom_logger import CustomLogger
from exceptions.custom_exception import DocumentPortalException
from config.settings_loader import load_config

from pipeline.embed_and_persist import create_embed_and_persist_service

config = load_config("config/config.yaml")
logger = CustomLogger().get_logger()

# class evaluation_pipeline:
#     def __init__(self):
try:
    logger.info("initializing retrieval pipeline")
    
    retriever = create_embed_and_persist_service()
    # Load the vector store
    loaded_store = retriever.load_vector_store(config["retriever"]["vector_database_directory"])
    logger.info("Vector store loaded successfully")
    
    # Perform a similarity search
    results = loaded_store.similarity_search("pf contribution of employee below base salary of 15,333", k=config["retriever"]["top_k"])    for i, result in enumerate(results):
        logger.info(f"Result {i+1}: {result.page_content}")
except Exception as e:  
    logger.error(f"Error in RAG_pipeline initialization(retrieval): {str(e)}")
    raise DocumentPortalException(e, sys) from e