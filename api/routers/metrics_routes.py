"""
Metrics routes — consumed by the Visualizer's dashboard.
Aggregates results from results/metrics/ produced by defender + attacker.
"""

from fastapi import APIRouter

router = APIRouter()


@router.get("/comparison")
def get_before_after_comparison():
    """Return before/after accuracy, detection rate, evasion rate for dashboard charts."""
    # TODO: read from results/metrics/ and return structured JSON
    return {"message": "TODO: implement metrics comparison endpoint"}


@router.get("/detection-rate")
def get_detection_rate():
    """Return current model detection rate."""
    # TODO: implement
    return {"message": "TODO: implement detection rate endpoint"}
