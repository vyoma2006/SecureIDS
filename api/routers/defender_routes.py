"""
Defender routes — owned by Team Lead.
Wraps src/defender/ logic (train_baseline, evaluate, adversarial_training)
via api/services/defender_service.py.
"""

from fastapi import APIRouter, UploadFile, File

router = APIRouter()


@router.post("/train")
def train_model():
    """Train the baseline IDS model (Random Forest / XGBoost)."""
    # TODO: call api.services.defender_service.train_baseline_model()
    return {"message": "TODO: implement training endpoint"}


@router.post("/predict")
def predict(file: UploadFile = File(...)):
    """Classify uploaded traffic data using the trained IDS."""
    # TODO: call api.services.defender_service.predict(file)
    return {"message": "TODO: implement prediction endpoint"}


@router.post("/retrain")
def retrain_with_adversarial_data():
    """Retrain the model using original + adversarial samples."""
    # TODO: call api.services.defender_service.adversarial_retrain()
    return {"message": "TODO: implement retraining endpoint"}
