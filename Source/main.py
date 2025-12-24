import sys
from pathlib import Path

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
sys.path.insert(0, str(project_root))

from logger.custom_logger import CustomLogger
from exceptions.custom_exception import DocumentPortalException
from config.settings_loader import load_config

from pipeline.data_ingestion import DocumentIngestionFactory
from pipeline.data_chunking import RecursiveDataChunker

config = load_config("config/config.yaml")
logger = CustomLogger().get_logger()

class RAG_pipeline:
    def __init__(self):
        try:
            logger.info("initializing data ingestion")
            pdf_ingestion = DocumentIngestionFactory.create_ingestion(config['data_source']['pdf_path'])
            pdf_text = pdf_ingestion.extract_text()
            logger.info(f"completed data ingestion")
        except Exception as e:
            logger.error(f"Error in RAG_pipeline initialization(data ingestion): {str(e)}")
            raise DocumentPortalException(e, sys) from e
        
        try:
            logger.info("initializing data chunking")
            chunker = RecursiveDataChunker(chunk_size=config["data_chunking"]["recursive_text_splitter"]["chunk_size"], chunk_overlap=config["data_chunking"]["recursive_text_splitter"]["chunk_overlap"])
            chunks = chunker.chunk_data(text=pdf_text)
            logger.info(f"completed data chunking with {len(chunks)} chunks created")
        except Exception as e:
            logger.error(f"Error in RAG_pipeline initialization(data chunking): {str(e)}")
            raise DocumentPortalException(e, sys) from e
        
if __name__ == "__main__":
    try:
        rag_pipeline = RAG_pipeline()
    except Exception as e:
        logger.error(f"Error in RAG_pipeline execution: {str(e)}")
        raise DocumentPortalException(e, sys) from e
