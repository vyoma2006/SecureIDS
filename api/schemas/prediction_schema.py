"""
Pydantic schemas for /defender/predict, /defender/train, /defender/retrain.

Describes the canonical response shapes returned by the Defender endpoints.
"""

from typing import Optional
from pydantic import BaseModel, Field


class RowPrediction(BaseModel):
    """One classified row from the uploaded CSV."""

    row_index: int = Field(description="0-based row index in the uploaded file.")
    predicted_class: str
    confidence: float = Field(description="Probability assigned to predicted_class.")
    probabilities: dict[str, float] = Field(
        description="Probability for every one of the 8 classes, keyed by class name."
    )


class PredictResponse(BaseModel):
    """Response shape for POST /defender/predict."""

    model_used: str = Field(
        description="'robust_model_all_eps' if available, else 'baseline_model'."
    )
    n_rows: int
    predictions: list[RowPrediction]


class TrainResponse(BaseModel):
    """Response shape for POST /defender/train."""

    status: str
    test_accuracy: Optional[float] = None
    message: Optional[str] = None


class RetrainResponse(BaseModel):
    """Response shape for POST /defender/retrain."""

    status: str
    experiment_name: Optional[str] = None
    robust_clean_accuracy: Optional[float] = None
    robust_evasion_rate: Optional[float] = None
    message: Optional[str] = None
