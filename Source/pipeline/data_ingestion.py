import sys
from pathlib import Path

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
sys.path.insert(0, str(project_root))

from logger.custom_logger import CustomLogger
from config.settings_loader import load_config
from exceptions.custom_exception import DocumentPortalException

from abc import ABC, abstractmethod
from dotenv import load_dotenv

config = load_config("config/config.yaml")
logger = CustomLogger().get_logger()

load_dotenv()