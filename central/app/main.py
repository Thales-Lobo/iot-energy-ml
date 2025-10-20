"""
central/app/main.py

FastAPI application that implements the central server for the IoT energy
anomaly detection system.

This temporary version runs without an actual ML model — instead, it returns
random predictions with equal probability. All structure is preserved so that
the real model can be plugged back in later without code refactoring.
"""

from __future__ import annotations

import os
from dotenv import load_dotenv
import time
import logging
import random
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field, validator

# Application-local imports
# (Commented out for now since the ML model is not being used)
# from app.model_loader import ModelService
from app.metrics_store import MetricsStore

# Load variables from .env
load_dotenv()

# Read threshold (default 0.5 if not defined)
THRESHOLD_ATTACK = float(os.getenv("THRESHOLD_ATTACK", 0.5))

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("central.main")

app = FastAPI(title="IoT Energy Central - ML Inference", version="0.1")

# Global singletons (loaded on startup)
# MODEL_SERVICE: Optional[ModelService] = None
METRICS: Optional[MetricsStore] = None


# -------------------------
# Pydantic request / response models
# -------------------------
class PredictIn(BaseModel):
    node_id: str = Field(..., description="Unique node identifier")
    timestamps: List[float] = Field(
        ..., description="List of timestamps (unix epoch seconds) corresponding to each sample"
    )
    currents: List[float] = Field(
        ..., description="List of current measurements (same length/order as timestamps)"
    )
    is_attack: bool = Field(..., description="Ground-truth: whether this sample window is under attack")

    @validator("timestamps")
    def timestamps_non_empty(cls, v: List[float]):
        if not v:
            raise ValueError("timestamps must be a non-empty list")
        return v

    @validator("currents")
    def currents_non_empty(cls, v: List[float]):
        if not v:
            raise ValueError("currents must be a non-empty list")
        return v

    @validator("currents")
    def lengths_match(cls, currents: List[float], values: Dict[str, Any]):
        timestamps = values.get("timestamps")
        if timestamps is not None and len(timestamps) != len(currents):
            raise ValueError("timestamps and currents must have the same length")
        return currents


class PredictOut(BaseModel):
    node_id: str
    pred_label: int = Field(..., description="Predicted label: 1=attack, 0=normal")
    prob_attack: float = Field(..., ge=0.0, le=1.0, description="Model probability for attack class")
    correct: bool = Field(..., description="Whether prediction matched provided ground-truth")


# -------------------------
# Classification core (temporary mock version)
# -------------------------
def classify_window(timestamps: List[float], currents: List[float]) -> Dict[str, Any]:
    """
    Classify a single window of measurements.

    Currently, no ML model is loaded. Instead, this mock implementation
    generates random predictions to simulate a balanced model output.

    Returns
    -------
    dict
        {
            "pred_label": int (0 or 1),
            "prob_attack": float in [0,1],
            "meta": dict (optional metadata)
        }

    Notes
    -----
    - The commented block below shows where real inference would normally occur.
    - When integrating the ML model, simply uncomment those lines.
    """

    # --- MOCK INFERENCE (for testing without ML model) ---
    prob_attack = random.random()
    pred_label = prob_attack > THRESHOLD_ATTACK
    return {"pred_label": pred_label, "prob_attack": prob_attack, "meta": {}}

    # --- REAL INFERENCE (to be re-enabled later) ---
    # if MODEL_SERVICE is None:
    #     logger.error("Model service is not initialized")
    #     raise RuntimeError("Model service is not available")
    #
    # try:
    #     pred_label, prob_attack = MODEL_SERVICE.predict(timestamps=timestamps, currents=currents)
    # except Exception as exc:
    #     logger.exception("ModelService.predict raised an exception")
    #     raise
    #
    # return {"pred_label": int(pred_label), "prob_attack": float(prob_attack), "meta": {}}


# -------------------------
# FastAPI lifecycle events
# -------------------------
@app.on_event("startup")
async def on_startup():
    """
    Initialize the metrics store on application startup.

    The model loading section is temporarily disabled since no ML model
    is currently in use. Only the metrics store is initialized.
    """
    global METRICS

    logger.info("Starting up central server (mock mode: no ML model loaded)")

    # --- Disabled model loading section ---
    # global MODEL_SERVICE
    # MODEL_SERVICE = ModelService(model_dir="saved_models")
    # try:
    #     MODEL_SERVICE.load()
    # except Exception as exc:
    #     logger.exception("Failed to load model service during startup")
    #     raise RuntimeError("Model loading failed") from exc

    # --- Metrics store initialization ---
    METRICS = MetricsStore(csv_path="data/results.csv")
    try:
        METRICS.load()
    except Exception:
        logger.warning("Could not load previous metrics; starting with empty store")
        METRICS.reset()


@app.on_event("shutdown")
async def on_shutdown():
    """
    Flush/persist metrics at shutdown if necessary.
    """
    logger.info("Shutting down: persisting metrics store")
    try:
        if METRICS is not None:
            METRICS.flush()
    except Exception:
        logger.exception("Failed to flush metrics on shutdown")


# -------------------------
# Endpoints
# -------------------------
@app.post("/predict", response_model=PredictOut)
async def predict_endpoint(payload: PredictIn, background_tasks: BackgroundTasks):
    """
    Receive a window (timestamps + currents) from a node, run classification,
    and return prediction + correctness boolean.

    The metrics recording is done in background to keep request latency minimal.
    """
    received_at = time.time()
    logger.debug("Received /predict from node=%s samples=%d", payload.node_id, len(payload.currents))

    # Perform mock classification
    try:
        result = classify_window(payload.timestamps, payload.currents)
    except Exception as exc:
        logger.exception("Classification failed for node=%s", payload.node_id)
        raise HTTPException(status_code=500, detail=f"classification error: {exc}")

    pred_label = int(result["pred_label"])
    prob_attack = float(result["prob_attack"])
    correct = (pred_label == int(payload.is_attack))

    record = {
        "node_id": payload.node_id,
        "sent_ts": payload.timestamps[-1] if payload.timestamps else None,
        "received_at": received_at,
        "pred_label": pred_label,
        "prob_attack": prob_attack,
        "is_attack": int(payload.is_attack),
        "correct": int(correct),
    }

    if METRICS is not None:
        background_tasks.add_task(_background_record_metrics, record)
    else:
        logger.warning("Metrics store not initialized; skipping metrics record")

    logger.info("Node=%s pred=%d prob=%.4f correct=%s", payload.node_id, pred_label, prob_attack, correct)
    return PredictOut(node_id=payload.node_id, pred_label=pred_label, prob_attack=prob_attack, correct=correct)


def _background_record_metrics(record: Dict[str, Any]) -> None:
    """Helper to record metrics from background tasks."""
    try:
        if METRICS is None:
            logger.error("Metrics store is not available in background task")
            return
        METRICS.record(record)
    except Exception:
        logger.exception("Failed to record metrics in background task")


@app.get("/metrics")
async def get_metrics():
    """Return a JSON summary of metrics."""
    if METRICS is None:
        raise HTTPException(status_code=500, detail="metrics store not available")
    try:
        return METRICS.summary()
    except Exception:
        logger.exception("Failed to produce metrics summary")
        raise HTTPException(status_code=500, detail="failed to produce metrics summary")


@app.get("/results.csv")
async def download_results():
    """Return the results CSV file for offline analysis."""
    if METRICS is None:
        raise HTTPException(status_code=500, detail="metrics store not available")
    try:
        return METRICS.get_csv_response()
    except AttributeError:
        logger.warning("MetricsStore has no get_csv_response; returning JSON summary instead")
        return METRICS.summary()
    except Exception:
        logger.exception("Failed to retrieve results CSV")
        raise HTTPException(status_code=500, detail="failed to retrieve results CSV")


if __name__ == "__main__":  # Dev mode helper
    import uvicorn
    logger.info("Running central FastAPI app (development mode, mock predictions)")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
