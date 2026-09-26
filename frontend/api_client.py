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
TIMEOUT_SECONDS = 5.0           # metrics reads: small JSON files
PREDICT_TIMEOUT_SECONDS = 60.0  # live prediction: upload + model inference

Result = tuple[Optional[dict[str, Any]], Optional[str]]


def _error_detail(response: httpx.Response) -> str:
    """Pull FastAPI's {"detail": ...} out of an error response, else raw text."""
    try:
        detail = response.json().get("detail")
    except Exception:  # not JSON
        detail = None
    if isinstance(detail, list):  # pydantic validation errors
        detail = "; ".join(str(d.get("msg", d)) for d in detail)
    return str(detail) if detail else (response.text or f"HTTP {response.status_code}")


def _connect_error() -> str:
    return (
        f"Can't reach the API at {API_BASE_URL}. "
        f"Is it running? (`python -m uvicorn api.main:app --reload`)"
    )


def _get(path: str, timeout: float = TIMEOUT_SECONDS) -> Result:
    url = f"{API_BASE_URL}{path}"
    try:
        response = httpx.get(url, timeout=timeout)
    except httpx.ConnectError:
        return None, _connect_error()
    except httpx.TimeoutException:
        return None, f"API request to {path} timed out after {timeout:.0f}s."
    except httpx.HTTPError as e:
        return None, f"API request to {path} failed: {e}"

    if response.status_code >= 400:
        return None, _error_detail(response)

    return response.json(), None


def get_baseline_metrics() -> Result:
    """GET /metrics/baseline — per-class precision/recall/F1 + confusion matrix."""
    return _get("/metrics/baseline")


def get_baseline_findings() -> Result:
    """GET /metrics/baseline-findings — raw markdown writeup from the Defender, if any."""
    return _get("/metrics/baseline-findings")


def get_comparison(experiment: str = "all_eps") -> Result:
    """
    GET /metrics/comparison — baseline vs. robust model, before/after.
    `experiment`: "all_eps" (default) or "high_eps_only".
    """
    return _get(f"/metrics/comparison?experiment={experiment}")


def get_detection_rate() -> Result:
    """GET /metrics/detection-rate — current best model's detection rate."""
    return _get("/metrics/detection-rate")


def get_evasion_rate() -> Result:
    """GET /metrics/evasion-rate — per-class, per-epsilon FGSM evasion rate."""
    return _get("/metrics/evasion-rate")


def get_features() -> Result:
    """
    GET /defender/features — required columns (model order), class names,
    training-set feature means, and which model file is serving predictions.
    Doubles as a "is the live model loaded?" check for the demo page.
    """
    return _get("/defender/features")


def predict(file_bytes: bytes, filename: str) -> Result:
    """
    POST /defender/predict — live prediction demo.
    Sends CSV bytes and returns the model's per-row classification.
    """
    url = f"{API_BASE_URL}/defender/predict"
    try:
        response = httpx.post(
            url,
            files={"file": (filename, file_bytes, "text/csv")},
            timeout=PREDICT_TIMEOUT_SECONDS,
        )
    except httpx.ConnectError:
        return None, _connect_error()
    except httpx.TimeoutException:
        return None, (
            f"Prediction timed out after {PREDICT_TIMEOUT_SECONDS:.0f}s. "
            f"Try a smaller file."
        )
    except httpx.HTTPError as e:
        return None, f"Prediction request failed: {e}"

    if response.status_code >= 400:
        return None, _error_detail(response)

    return response.json(), None
