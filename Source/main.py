import sys
from pathlib import Path

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
sys.path.insert(0, str(project_root))

from logger.custom_logger import CustomLogger
from exceptions.custom_exception import DocumentPortalException
from config.settings_loader import load_config

from pipeline.data_ingestion import DocumentIngestionFactory

config = load_config("config/config.yaml")
logger = CustomLogger().get_logger()

class RAG_pipeline:
    def __init__(self):
        try:
            logger.info("initializing data ingestion")
            pdf_ingestion = DocumentIngestionFactory.create_ingestion(config['data_source']['pdf_path'])
            pdf_text = pdf_ingestion.extract_structured()
            logger.info(f"completed data ingestion")
        except Exception as e:
            logger.error(f"Error in RAG_pipeline initialization(data ingestion): {str(e)}")
            raise DocumentPortalException(e, sys) from e
