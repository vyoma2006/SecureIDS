"""
Pydantic schemas for /metrics/* endpoints.

These describe the API's canonical response shapes (response_model). The
raw files teammates actually write to results/metrics/ don't match these
1:1 in every case (e.g. baseline_metrics.json is a raw sklearn
classification_report dump) — api/services/metrics_service.py adapts
between the two.
"""

from typing import Optional
from pydantic import BaseModel, Field


class PerClassMetrics(BaseModel):
    precision: float
    recall: float
    f1: float
    support: int


class BaselineMetrics(BaseModel):
    """Canonical baseline performance shape, adapted from baseline_metrics.json."""

    model_name: str
    accuracy: float
    macro_f1: float
    classes: list[str]
    per_class: dict[str, PerClassMetrics]
    confusion_matrix: list[list[int]] = Field(
        description="Rows = true label, columns = predicted label, order matches `classes`."
    )
    trained_at: Optional[str] = None
    notes: Optional[str] = Field(
        default=None,
        description="Free-text caveats from the Defender (e.g. known weak/confusable classes) shown as-is on the dashboard.",
    )


class BaselineFindings(BaseModel):
    """Raw markdown from results/reports/baseline_findings.md, passed through as-is."""

    markdown: str


class ModelSnapshot(BaseModel):
    """One side (baseline or robust) of the before/after comparison."""

    accuracy: float
    detection_rate: float = Field(
        description="Fraction of adversarial traffic still caught, i.e. (1 - evasion_rate)."
    )
    macro_f1: float
    evasion_rate: Optional[float] = Field(
        default=None, description="Fraction of adversarial samples that fooled the model."
    )


class ComparisonMetrics(BaseModel):
    """Adapted from results/metrics/comparison_<experiment>.json (written by src/defender/adversarial_training.py)."""

    experiment_name: str = Field(description="'all_eps' or 'high_eps_only' — which ablation run this is.")
    baseline: ModelSnapshot
    robust: ModelSnapshot
    evaluated_at: Optional[str] = None


class DetectionRateResponse(BaseModel):
    model_name: str
    detection_rate: float
    source: str = Field(description="'robust' if adversarial training has run, else 'baseline'.")


class EvasionRecord(BaseModel):
    attack_class: str = Field(description="Traffic class the sample was crafted from, e.g. 'Bot'.")
    epsilon: float
    evasion_rate: float = Field(description="Fraction of adversarial samples that fooled the model at this epsilon.")
    confidence_drop: Optional[float] = Field(
        default=None, description="Average drop in true-class confidence after perturbation."
    )
    n_samples: Optional[int] = None


class EvasionMetrics(BaseModel):
    """Canonical evasion shape, adapted from the raw bare-array evasion_metrics.json."""

    attack_type: str = "FGSM"
    target_model: Optional[str] = None
    results: list[EvasionRecord]
    generated_at: Optional[str] = None


class NotAvailable(BaseModel):
    """Returned (as the detail of a 404) when the upstream file hasn't been produced yet."""

    available: bool = False
    message: str