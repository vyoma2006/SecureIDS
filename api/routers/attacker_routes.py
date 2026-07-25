"""
Attacker routes.
Wraps src/attacker/ logic (fgsm_attack, generate_adversarial, evasion_metrics)
via api/services/attacker_service.py.
"""

from fastapi import APIRouter

router = APIRouter()


@router.post("/generate")
def generate_adversarial_samples():
    """Generate adversarial traffic samples via FGSM."""
    # TODO: call api.services.attacker_service.generate_fgsm_samples()
    return {"message": "TODO: implement adversarial generation endpoint"}


@router.get("/evasion-rate")
def get_evasion_rate():
    """Return the evasion rate of the last adversarial batch against the current model."""
    # TODO: call api.services.attacker_service.compute_evasion_rate()
    return {"message": "TODO: implement evasion rate endpoint"}
