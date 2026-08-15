"""
API client — every call the dashboard makes to the SecureIDS FastAPI backend.

Deliberately dumb: no caching, no Streamlit imports here. Caching lives in
dashboard_app.py (st.cache_data) so this module stays testable on its own.

Every function returns (data, error) where exactly one of the two is None:
    data, error = get_baseline_metrics()
    if error:
        st.warning(error)
    else:
        ...use data...
"""

import os
from typing import Any, Optional

import httpx

API_BASE_URL = os.environ.get("SECUREIDS_API_URL", "http://127.0.0.1:8000")
TIMEOUT_SECONDS = 5.0


def _get(path: str) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    url = f"{API_BASE_URL}{path}"
    try:
        response = httpx.get(url, timeout=TIMEOUT_SECONDS)
    except httpx.ConnectError:
        return None, (
            f"Can't reach the API at {API_BASE_URL}. "
            f"Is it running? (`uvicorn api.main:app --reload`)"
        )
    except httpx.TimeoutException:
        return None, f"API request to {path} timed out after {TIMEOUT_SECONDS}s."
    except httpx.HTTPError as e:
        return None, f"API request to {path} failed: {e}"

    if response.status_code == 404:
        detail = response.json().get("detail", "Data not available yet.")
        return None, detail
    if response.status_code >= 400:
        return None, f"API returned {response.status_code} for {path}: {response.text}"

    return response.json(), None


def get_baseline_metrics() -> tuple[Optional[dict[str, Any]], Optional[str]]:
    """GET /metrics/baseline — per-class precision/recall/F1 + confusion matrix."""
    return _get("/metrics/baseline")


def get_baseline_findings() -> tuple[Optional[dict[str, Any]], Optional[str]]:
    """GET /metrics/baseline-findings — raw markdown writeup from the Defender, if any."""
    return _get("/metrics/baseline-findings")


def get_comparison(experiment: str = "all_eps") -> tuple[Optional[dict[str, Any]], Optional[str]]:
    """
    GET /metrics/comparison — baseline vs. robust model, before/after.
    `experiment`: "all_eps" (default) or "high_eps_only".
    """
    return _get(f"/metrics/comparison?experiment={experiment}")


def get_detection_rate() -> tuple[Optional[dict[str, Any]], Optional[str]]:
    """GET /metrics/detection-rate — current best model's detection rate."""
    return _get("/metrics/detection-rate")


def get_evasion_rate() -> tuple[Optional[dict[str, Any]], Optional[str]]:
    """GET /metrics/evasion-rate — per-class, per-epsilon FGSM evasion rate."""
    return _get("/metrics/evasion-rate")


def predict(file_bytes: bytes, filename: str) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    """
    POST /defender/predict — live prediction demo (optional dashboard feature).
    Sends an uploaded traffic file and returns the model's classification.
    """
    url = f"{API_BASE_URL}/defender/predict"
    try:
        response = httpx.post(
            url,
            files={"file": (filename, file_bytes)},
            timeout=TIMEOUT_SECONDS,
        )
    except httpx.HTTPError as e:
        return None, f"Prediction request failed: {e}"

    if response.status_code >= 400:
        return None, f"API returned {response.status_code}: {response.text}"

    data = response.json()
    if isinstance(data, dict) and data.get("message", "").startswith("TODO"):
        return None, "The /defender/predict endpoint isn't implemented yet."
    return data, None