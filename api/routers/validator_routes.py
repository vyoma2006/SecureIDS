"""
Validator routes — owned by the cybersecurity expert.
Wraps src/validator/ logic to sanity-check the system end-to-end.
"""

from fastapi import APIRouter

router = APIRouter()


@router.post("/validate")
def validate_system():
    """Run the validation checklist against the current model/pipeline."""
    # TODO: call api.services.validator_service.run_validation_checklist()
    return {"message": "TODO: implement validation endpoint"}


@router.get("/threat-report")
def get_threat_report():
    """Return the MITRE ATT&CK / CVE mapped threat report."""
    # TODO: implement
    return {"message": "TODO: implement threat report endpoint"}
