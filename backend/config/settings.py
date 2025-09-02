"""
Configuration settings for the DR Classification API
"""

import os
from typing import List
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Application settings"""

    # API Configuration
    API_TITLE: str = "Diabetic Retinopathy Classification API"
    API_VERSION: str = "1.0.0"
    API_DESCRIPTION: str = "Medical AI system for automated diabetic retinopathy detection"

    # Server Configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = True

    # Model Configuration
    IMAGE_SIZE: int = 224
    NUM_CLASSES: int = 5
    CONFIDENCE_THRESHOLD: float = 0.7

    # Model Paths
    MODEL_DIR: str = "data/models"
    EFFICIENTNET_MODEL_PATH: str = os.path.join(MODEL_DIR, "efficientnet_model.pth")
    VIT_MODEL_PATH: str = os.path.join(MODEL_DIR, "vit_model.pth")
    HYBRID_MODEL_PATH: str = os.path.join(MODEL_DIR, "hybrid_model.pth")

    # Data Processing
    USE_BEN_GRAHAM: bool = True
    USE_CLAHE: bool = True
    MAX_BATCH_SIZE: int = 10
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB

    # Allowed file formats
    ALLOWED_EXTENSIONS: List[str] = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
    ALLOWED_CONTENT_TYPES: List[str] = [
        "image/jpeg", "image/png", "image/bmp", "image/tiff"
    ]

    # CORS Configuration
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8080"
    ]

    class Config:
        env_file = ".env"
        case_sensitive = True

def get_settings() -> Settings:
    """Get settings based on environment"""
    return Settings()

# Global settings instance
settings = get_settings()
