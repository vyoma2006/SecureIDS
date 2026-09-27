"""
Pydantic schemas for /defender/predict, /defender/features, /defender/train,
/defender/retrain.

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
    actual_label: Optional[str] = Field(
        default=None,
        description="Value of the 'Label' column for this row, if the uploaded CSV had one.",
    )


class SkippedRow(BaseModel):
    """A row that could not be classified (and why)."""

    row_index: int = Field(description="0-based row index in the uploaded file.")
    reason: str


class PredictResponse(BaseModel):
    """Response shape for POST /defender/predict."""

    model_used: str = Field(
        description="'robust_model_all_eps' if available, else 'baseline_model'."
    )
    n_rows: int = Field(description="Number of data rows in the uploaded file.")
    n_classified: int = Field(description="How many of those rows were classified.")
    predictions: list[RowPrediction]
    skipped_rows: list[SkippedRow] = Field(
        default_factory=list,
        description="Rows left out because they had missing / infinite / non-numeric feature values.",
    )


class FeatureInfoResponse(BaseModel):
    """Response shape for GET /defender/features."""

    model_used: str
    class_names: list[str]
    feature_columns: list[str] = Field(description="The 78 required columns, in model order.")
    feature_means: dict[str, float] = Field(
        description="Training-set mean of each feature in RAW (unscaled) units."
    )


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
