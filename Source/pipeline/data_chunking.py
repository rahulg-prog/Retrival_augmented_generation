import sys
from pathlib import Path

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
sys.path.insert(0, str(project_root))

from logger.custom_logger import CustomLogger
from exceptions.custom_exception import DocumentPortalException

from abc import ABC, abstractmethod
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = CustomLogger().get_logger()

class DataChunker(ABC):
    @abstractmethod
    def chunk_data(self, text: str) -> List[str]:
        pass
    
class RecursiveDataChunker(DataChunker):
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    
    def chunk_data(self, text: str) -> List[str]:
        try:
            chunks = self.text_splitter.split_text(text)
            return chunks
        except Exception as e:
            logger.error(f"Error during recursive data chunking: {str(e)} in data_chunking.py")
            raise DocumentPortalException("Failed to chunk data recursively", e)
        
if __name__ == "__main__":
    from config.settings_loader import load_config
    config = load_config("config/config.yaml")
    try:
        sample_text = "This is a sample text. " * 100
        chunker = RecursiveDataChunker(chunk_size=config["data_chunking"]["recursive_text_splitter"]["chunk_size"], chunk_overlap=config["data_chunking"]["recursive_text_splitter"]["chunk_overlap"])
        chunks = chunker.chunk_data(sample_text)
        for i, chunk in enumerate(chunks):
            print(f"Chunk {i+1}:\n{chunk}\n")
    except DocumentPortalException as e:
        logger.error(f"Data chunking failed: {str(e)}")