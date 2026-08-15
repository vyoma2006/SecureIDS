"""
Metrics service — the ONLY place in the API that touches results/metrics/*.json.

Per docs/architecture.md this layer never recomputes metrics — it only reads
what the Defender / Attacker have already written to disk. It DOES adapt
their raw file shapes into the canonical shape api/schemas/metrics_schema.py
expects, because the raw files are sklearn/torch tool output (e.g.
classification_report()), not something we should ask teammates to
hand-reformat. See docs/metrics_contract.md for both shapes side by side.

If a file doesn't exist yet, every public function here raises
FileNotFoundError with a friendly message rather than crashing —
api/routers/metrics_routes.py turns that into a clean 404.
"""

import json
from pathlib import Path
from typing import Any

RESULTS_METRICS_DIR = Path(__file__).resolve().parents[2] / "results" / "metrics"
RESULTS_REPORTS_DIR = Path(__file__).resolve().parents[2] / "results" / "reports"

BASELINE_FILE = "baseline_metrics.json"
COMPARISON_FILES = {
    "all_eps": "comparison_all_eps.json",
    "high_eps_only": "comparison_high_eps_only.json",
}
DEFAULT_EXPERIMENT = "all_eps"
EVASION_FILE = "evasion_metrics.json"
BASELINE_FINDINGS_FILE = "baseline_findings.md"

# sklearn's classification_report() mixes these summary keys in with the
# per-class keys — strip them out when building per-class metrics.
_NON_CLASS_KEYS = {"accuracy", "macro avg", "weighted avg"}

DEFAULT_MODEL_NAME = "IDS_MLP"
DEFAULT_TARGET_MODEL = "IDS_MLP (baseline)"


def _load_json(filename: str) -> Any:
    path = RESULTS_METRICS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"'{filename}' not found in results/metrics/. "
            f"This file hasn't been produced yet — check with whoever owns that pipeline step."
        )
    with path.open("r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"'{filename}' exists but isn't valid JSON: {e}") from e


# --------------------------------------------------------------------------
# baseline_metrics.json
#
# Raw shape on disk (sklearn classification_report + confusion_matrix dump):
#   {
#     "test_accuracy": 0.914...,
#     "classification_report": {
#       "Benign": {"precision":..., "recall":..., "f1-score":..., "support":...},
#       ...,
#       "accuracy": 0.914...,
#       "macro avg": {...}, "weighted avg": {...}
#     },
#     "confusion_matrix": [[...], ...],
#     "class_names": [...]
#   }
# --------------------------------------------------------------------------

def get_baseline_metrics() -> dict[str, Any]:
    """Baseline metrics, adapted from the raw sklearn-report shape to our canonical shape."""
    raw = _load_json(BASELINE_FILE)
    try:
        report = raw["classification_report"]
        classes = raw["class_names"]
        per_class = {
            class_name: {
                "precision": report[class_name]["precision"],
                "recall": report[class_name]["recall"],
                "f1": report[class_name]["f1-score"],
                "support": int(report[class_name]["support"]),
            }
            for class_name in classes
            if class_name not in _NON_CLASS_KEYS
        }
        return {
            "model_name": raw.get("model_name", DEFAULT_MODEL_NAME),
            "accuracy": raw["test_accuracy"],
            "macro_f1": report["macro avg"]["f1-score"],
            "classes": classes,
            "per_class": per_class,
            "confusion_matrix": raw["confusion_matrix"],
            "trained_at": raw.get("trained_at"),
            "notes": raw.get("notes"),
        }
    except KeyError as e:
        raise ValueError(f"'{BASELINE_FILE}' is missing expected key: {e}") from e


def get_baseline_findings() -> str:
    """Raw markdown text of results/reports/baseline_findings.md, if the Defender wrote one."""
    path = RESULTS_REPORTS_DIR / BASELINE_FINDINGS_FILE
    if not path.exists():
        raise FileNotFoundError(f"'{BASELINE_FINDINGS_FILE}' not found in results/reports/.")
    return path.read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# comparison_<experiment>.json
#
# Two ablation runs actually landed on disk instead of a single
# comparison.json: "all_eps" (trained on adversarial samples across every
# epsilon) and "high_eps_only" (trained only on high-epsilon samples).
#
# Raw shape on disk (from src/defender/adversarial_training.py):
#   {
#     "experiment_name": "all_eps",
#     "baseline": {"clean_test_accuracy":..., "macro_f1":..., "adversarial_evasion_rate":...},
#     "robust": {"clean_test_accuracy":..., "macro_f1":..., "adversarial_evasion_rate":...,
#                "classification_report": {...}},
#     "improvement": {...}, "class_names": [...], ...
#   }
#
# There's no classification-report-based recall in the baseline block, so
# "detection_rate" is defined here as (1 - adversarial_evasion_rate): the
# fraction of adversarial traffic the model still catches. That's the
# number the before/after comparison is actually trying to answer.
# --------------------------------------------------------------------------

def get_comparison_metrics(experiment: str = DEFAULT_EXPERIMENT) -> dict[str, Any]:
    """Before/after comparison for one adversarial-training ablation run."""
    if experiment not in COMPARISON_FILES:
        raise ValueError(
            f"Unknown experiment '{experiment}'. Choose one of: {', '.join(COMPARISON_FILES)}"
        )
    raw = _load_json(COMPARISON_FILES[experiment])

    def _snapshot(block: dict[str, Any]) -> dict[str, Any]:
        return {
            "accuracy": block["clean_test_accuracy"],
            "macro_f1": block["macro_f1"],
            "detection_rate": 1 - block["adversarial_evasion_rate"],
            "evasion_rate": block["adversarial_evasion_rate"],
        }

    try:
        return {
            "experiment_name": raw.get("experiment_name", experiment),
            "baseline": _snapshot(raw["baseline"]),
            "robust": _snapshot(raw["robust"]),
            "evaluated_at": raw.get("evaluated_at"),
        }
    except KeyError as e:
        raise ValueError(f"comparison file for '{experiment}' is missing expected key: {e}") from e


# --------------------------------------------------------------------------
# evasion_metrics.json
#
# Raw shape on disk is a bare JSON array (not an object):
#   [{"class": "Bot", "epsilon": 0.01, "evasion_rate": ..., "confidence_drop": ..., "n_samples": ...}, ...]
# --------------------------------------------------------------------------

def get_evasion_metrics() -> dict[str, Any]:
    """Evasion metrics, adapted from the raw bare-array shape to our canonical shape."""
    raw = _load_json(EVASION_FILE)
    if not isinstance(raw, list):
        raise ValueError(f"'{EVASION_FILE}' was expected to be a JSON array, got {type(raw).__name__}.")
    try:
        results = [
            {
                "attack_class": record["class"],
                "epsilon": record["epsilon"],
                "evasion_rate": record["evasion_rate"],
                "confidence_drop": record.get("confidence_drop"),
                "n_samples": record.get("n_samples"),
            }
            for record in raw
        ]
    except KeyError as e:
        raise ValueError(f"'{EVASION_FILE}' record is missing expected key: {e}") from e

    return {
        "attack_type": "FGSM",
        "target_model": DEFAULT_TARGET_MODEL,
        "results": results,
        "generated_at": None,
    }


# --------------------------------------------------------------------------
# detection-rate summary
# --------------------------------------------------------------------------

def get_detection_rate() -> dict[str, Any]:
    """
    Convenience summary: the *current best* model's detection rate.

    Prefers the robust model from comparison.json (adversarial training has
    run); falls back to the baseline model's recall on attack traffic if
    adversarial training hasn't happened yet.
    """
    try:
        comparison = get_comparison_metrics()
        return {
            "model_name": "robust",
            "detection_rate": comparison["robust"]["detection_rate"],
            "source": "robust",
        }
    except FileNotFoundError:
        pass  # fall through to baseline

    baseline = get_baseline_metrics()
    detection_rate = _detection_rate_from_baseline(baseline)
    return {
        "model_name": baseline.get("model_name", DEFAULT_MODEL_NAME),
        "detection_rate": detection_rate,
        "source": "baseline",
    }


def _detection_rate_from_baseline(baseline: dict[str, Any]) -> float:
    """Recall averaged across all classes except 'Benign', weighted by support."""
    per_class = baseline.get("per_class", {})
    attack_classes = {c: m for c, m in per_class.items() if c.lower() != "benign"}
    if not attack_classes:
        return 0.0
    total_support = sum(m["support"] for m in attack_classes.values())
    if total_support == 0:
        return 0.0
    weighted_recall = sum(m["recall"] * m["support"] for m in attack_classes.values())
    return weighted_recall / total_support