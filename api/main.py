"""
SecureIDS API — entry point.

Run locally with:
    uvicorn api.main:app --reload

Interactive docs available at http://127.0.0.1:8000/docs
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import defender_routes, attacker_routes, metrics_routes, validator_routes
from api.services import defender_service

logger = logging.getLogger("secureids")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model / scaler / encoder once at startup so the first live
    # prediction doesn't pay the loading cost. If artifacts are missing we
    # still start; /defender/predict will report a clear 503 instead.
    try:
        info = defender_service.load_inference_artifacts()
        logger.warning("Inference model ready: %s", info["model_used"])
    except Exception as e:  # noqa: BLE001
        logger.warning("Inference model NOT loaded at startup: %s", e)
    yield


app = FastAPI(
    lifespan=lifespan,
    title="SecureIDS API",
    description="Network Intrusion Detection System with Adversarial Attack Simulation and Defense",
    version="0.1.0",
)

# Allow the dashboard (Streamlit/React, likely on a different port) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this before deploying anywhere real
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(defender_routes.router, prefix="/defender", tags=["Defender"])
app.include_router(attacker_routes.router, prefix="/attacker", tags=["Attacker"])
app.include_router(metrics_routes.router, prefix="/metrics", tags=["Metrics"])
app.include_router(validator_routes.router, prefix="/validator", tags=["Validator"])


@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "service": "SecureIDS API"}