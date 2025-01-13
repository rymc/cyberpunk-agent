from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import List
import json
from pathlib import Path
from dotenv import load_dotenv
import logging
import os

logger = logging.getLogger(__name__)

env_path = Path(__file__).parent.parent.parent.parent / '.env'
logger.info(f"Loading environment variables from: {env_path}")
load_dotenv(dotenv_path=env_path, override=True)

class Settings(BaseSettings):
    """Application settings."""
    
    # LLM Configuration
    llm_base_url: str
    llm_api_key: str
    
    # Search Configuration
    tavily_api_key: str
    
    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    
    # Logging
    log_level: str = "INFO"
    
    # Security
    cors_origins: List[str] = ["http://localhost:8000", "http://127.0.0.1:8000"]
    max_requests_per_minute: int = 60
    
    class Config:
        env_file = ".env"
        env_prefix = "AGENT_"

@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    settings = Settings()
    logger.info(f"Settings loaded - LLM URL: {settings.llm_base_url}, Tavily API key present: {'tavily_api_key' in settings.__dict__}")
    return settings 