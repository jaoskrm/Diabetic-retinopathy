"""
Pydantic schemas for API request/response models
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

class RiskLevel(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"

class DRSeverity(str, Enum):
    NO_DR = "No Diabetic Retinopathy"
    MILD = "Mild Diabetic Retinopathy"
    MODERATE = "Moderate Diabetic Retinopathy"
    SEVERE = "Severe Diabetic Retinopathy"
    PROLIFERATIVE = "Proliferative Diabetic Retinopathy"

class PredictionResponse(BaseModel):
    """Response model for DR prediction"""

    predicted_class: int = Field(..., description="Predicted class (0-4)", ge=0, le=4)
    class_name: str = Field(..., description="Human-readable class name")
    severity_description: str = Field(..., description="Detailed severity description")
    confidence: float = Field(..., description="Prediction confidence", ge=0.0, le=1.0)
    risk_level: RiskLevel = Field(..., description="Risk assessment level")
    probabilities: Dict[str, float] = Field(..., description="Probabilities for each class")

    model_used: str = Field(..., description="Name of the model used for prediction")
    recommendations: List[str] = Field(..., description="Clinical recommendations")
    requires_immediate_attention: bool = Field(..., description="Whether immediate medical attention is needed")

    # Optional metadata
    filename: Optional[str] = Field(None, description="Original filename")
    processing_time: Optional[float] = Field(None, description="Processing time in seconds")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Prediction timestamp")

class ModelInfo(BaseModel):
    """Information about available models"""

    name: str = Field(..., description="Model name")
    architecture: str = Field(..., description="Model architecture")
    parameters: int = Field(..., description="Number of parameters")
    input_size: int = Field(..., description="Expected input image size")
    num_classes: int = Field(..., description="Number of output classes")
    accuracy: Optional[float] = Field(None, description="Reported accuracy on test set")

class HealthCheck(BaseModel):
    """Health check response"""

    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(..., description="Current timestamp")
    models_loaded: List[str] = Field(..., description="List of loaded models")
    device: str = Field(..., description="Computing device (CPU/GPU)")
    memory_usage: Optional[Dict[str, Any]] = Field(None, description="Memory usage statistics")
