import sys
from pathlib import Path

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
sys.path.insert(0, str(project_root))

from logger.custom_logger import CustomLogger
from exceptions.custom_exception import DocumentPortalException
from models.embedding import get_default_embeddings

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

logger = CustomLogger().get_logger()

