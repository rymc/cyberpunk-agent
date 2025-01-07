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
logger.info(f"Current working directory: {os.getcwd()}")
logger.info(f"Environment file exists: {env_path.exists()}")
if env_path.exists():
    logger.info(f"Environment file contents:")
    with open(env_path) as f:
        logger.info(f.read())

load_dotenv(dotenv_path=env_path, override=True)

# Log all environment variables starting with AGENT
for key, value in os.environ.items():
    if key.startswith('AGENT_'):
        logger.info(f"Environment variable {key}: {value}")

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
        case_sensitive = False
        
        @classmethod
        def parse_env_var(cls, field_name: str, raw_val: str):
            if field_name == "cors_origins":
                return json.loads(raw_val)
            return raw_val
            
    class Config:
        env_prefix = "AGENT_"

@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    settings = Settings()
    logger.info(f"Loaded settings with LLM_BASE_URL: {settings.llm_base_url}")
    return settings 