"""
Defender routes — owned by Team Lead.
Wraps src/defender/ logic (train_baseline, evaluate, adversarial_training)
via api/services/defender_service.py.
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Query

from api.schemas.prediction_schema import PredictResponse, RetrainResponse, TrainResponse
from api.services import defender_service
from api.services.defender_service import ArtifactsNotFoundError

router = APIRouter()


@router.post("/train", response_model=TrainResponse)
def train_model():
    """
    Train the baseline IDS model (MLP) from scratch on data/processed/.

    SLOW — this blocks the API while it runs (multiple training epochs over
    the full dataset). Treat as an admin-only operation with a loading
    state, not an instant end-user button.
    """
    try:
        result = defender_service.train_baseline_model()
        return {"status": result["status"], "test_accuracy": result["test_accuracy"]}
    except ArtifactsNotFoundError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {e}")


@router.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    """
    Classify uploaded traffic data using the trained IDS.

    Upload a CSV of RAW (unscaled) feature values — one row per traffic
    sample, containing at least the 78 columns listed in
    src/defender/saved_models/feature_columns.json. Scaling is handled
    automatically server-side; do not pre-scale.

    Returns one prediction per row: predicted class, confidence, and the
    full probability distribution across all 8 classes.
    """
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a .csv file.")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        return defender_service.predict(contents)
    except ArtifactsNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


@router.post("/retrain", response_model=RetrainResponse)
def retrain_with_adversarial_data(
    experiment: str = Query(
        default="all_eps",
        description="'all_eps' (recommended) or 'high_eps_only' — which adversarial-training ablation to run.",
    )
):
    """
    Retrain the model using original + adversarial samples (adversarial
    training). Requires POST /attacker/generate to have already produced
    data/adversarial/ samples.

    SLOW — this blocks the API while it runs. Treat as an admin-only
    operation with a loading state, not an instant end-user button.
    """
    try:
        result = defender_service.adversarial_retrain(experiment)
        return {
            "status": result["status"],
            "experiment_name": result["experiment_name"],
            "robust_clean_accuracy": result["robust_clean_accuracy"],
            "robust_evasion_rate": result["robust_evasion_rate"],
        }
    except ArtifactsNotFoundError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retraining failed: {e}")
