"""
Metrics routes — consumed by the Visualizer's dashboard.

Reads pre-computed results from results/metrics/ (via api/services/metrics_service.py)
and reshapes them per api/schemas/metrics_schema.py. This layer never
recomputes anything.

Every endpoint returns 404 (not a crash, not a silent {}) when the upstream
file hasn't been produced yet, so the dashboard can distinguish
"not ready" from "broken."
"""

from fastapi import APIRouter, HTTPException

from api.schemas.metrics_schema import (
    BaselineFindings,
    BaselineMetrics,
    ComparisonMetrics,
    DetectionRateResponse,
    EvasionMetrics,
)
from api.services import metrics_service

router = APIRouter()


@router.get("/baseline", response_model=BaselineMetrics)
def get_baseline_metrics():
    """Baseline model performance: per-class precision/recall/F1 + confusion matrix."""
    try:
        return metrics_service.get_baseline_metrics()
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/baseline-findings", response_model=BaselineFindings)
def get_baseline_findings():
    """Raw markdown from results/reports/baseline_findings.md, e.g. the Benign/Infilteration writeup."""
    try:
        return {"markdown": metrics_service.get_baseline_findings()}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/comparison", response_model=ComparisonMetrics)
def get_before_after_comparison(experiment: str = "all_eps"):
    """
    Before/after accuracy, detection rate, and F1 from adversarial training.

    `experiment` selects which ablation run to read:
      - "all_eps" (default): trained on adversarial samples across all epsilons
      - "high_eps_only": trained only on high-epsilon adversarial samples
    """
    try:
        return metrics_service.get_comparison_metrics(experiment)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        status = 400 if "Unknown experiment" in str(e) else 500
        raise HTTPException(status_code=status, detail=str(e))


@router.get("/detection-rate", response_model=DetectionRateResponse)
def get_detection_rate():
    """Current model's detection rate (robust model if available, else baseline)."""
    try:
        return metrics_service.get_detection_rate()
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/evasion-rate", response_model=EvasionMetrics)
def get_evasion_rate():
    """Per-class, per-epsilon FGSM evasion rate (+ optional confidence drop)."""
    try:
        return metrics_service.get_evasion_metrics()
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))