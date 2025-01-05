from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import List
import json

class Settings(BaseSettings):
    """Application settings."""
    
    # LLM Configuration
    llm_base_url: str
    llm_api_key: str
    
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

@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings() 