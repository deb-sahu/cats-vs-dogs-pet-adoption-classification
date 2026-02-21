"""Pydantic schemas for API request/response validation."""

from typing import Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str = Field(..., description="Service status", examples=["healthy"])
    model_loaded: bool = Field(..., description="Whether the model is loaded")
    version: str = Field(..., description="API version", examples=["1.0.0"])


class PredictionResponse(BaseModel):
    """Prediction response."""
    
    prediction: int = Field(
        ...,
        description="Predicted class (0=cat, 1=dog)",
        ge=0,
        le=1,
    )
    label: str = Field(
        ...,
        description="Human-readable class label",
        examples=["cat", "dog"],
    )
    confidence: float = Field(
        ...,
        description="Prediction confidence score",
        ge=0.0,
        le=1.0,
    )
    probability_cat: float = Field(
        ...,
        description="Probability of being a cat",
        ge=0.0,
        le=1.0,
    )
    probability_dog: float = Field(
        ...,
        description="Probability of being a dog",
        ge=0.0,
        le=1.0,
    )


class ErrorResponse(BaseModel):
    """Error response."""
    
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")


class ModelInfoResponse(BaseModel):
    """Model information response."""
    
    model_type: str = Field(..., description="Model architecture type")
    input_size: list = Field(..., description="Expected input image size [H, W]")
    classes: list = Field(..., description="Class labels")
    version: str = Field(..., description="Model version")
