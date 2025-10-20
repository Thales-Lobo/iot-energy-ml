"""
central/app/model_loader.py

This module defines the ModelService class, responsible for:
- Loading the trained ML model and associated scaler/preprocessing artifacts.
- Providing a clean, modular interface for predictions on incoming IoT data.
- Applying preprocessing to convert raw time-series data (timestamps + currents)
  into the same feature representation used during training.

This structure ensures the model logic is isolated from the FastAPI layer,
promoting modularity and testability.

Expected artifacts in model_dir:
  - model.keras (or model.h5): Trained TensorFlow/Keras model file.
  - scaler.joblib: Scaler object (e.g., StandardScaler) used during training.

Dependencies:
  - numpy
  - joblib
  - tensorflow (keras)
  - local module `preprocess.py` for decimation and feature extraction
"""

from __future__ import annotations

import os
import logging
import numpy as np
from typing import Tuple, List

from tensorflow.keras.models import load_model
from joblib import load as joblib_load

from app.preprocess import decimate_if_needed, extract_features

logger = logging.getLogger("central.model_loader")


class ModelService:
    """
    Service class responsible for managing model loading and inference.

    Attributes
    ----------
    model_dir : str
        Directory containing the model and scaler artifacts.
    model : keras.Model or None
        The loaded TensorFlow/Keras model used for inference.
    scaler : object or None
        The loaded Scaler (e.g., StandardScaler) used to normalize feature vectors.

    Methods
    -------
    load():
        Loads the model and scaler artifacts from disk.
    predict(timestamps, currents):
        Runs preprocessing and model inference on the given time-series window.
    """

    def __init__(self, model_dir: str = "saved_models") -> None:
        """
        Initialize the service with the given model directory.

        Parameters
        ----------
        model_dir : str, optional
            Path to directory containing model.keras and scaler.joblib (default: 'saved_models')
        """
        self.model_dir = model_dir
        self.model = None
        self.scaler = None

    # ----------------------------------------------------------------------
    def load(self) -> None:
        """
        Load model and scaler artifacts from disk.

        Raises
        ------
        FileNotFoundError
            If model or scaler files are missing.
        Exception
            If loading fails due to corruption or version mismatch.
        """
        model_path = os.path.join(self.model_dir, "model.keras")
        scaler_path = os.path.join(self.model_dir, "scaler.joblib")

        logger.info(f"Loading ML model from {model_path}")
        logger.info(f"Loading scaler from {scaler_path}")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler file not found: {scaler_path}")

        try:
            self.model = load_model(model_path)
            self.scaler = joblib_load(scaler_path)
        except Exception as exc:
            logger.exception("Failed to load model or scaler")
            raise RuntimeError("Model loading failed") from exc

        logger.info("Model and scaler successfully loaded")

    # ----------------------------------------------------------------------
    def _preprocess(self, timestamps: List[float], currents: List[float]) -> np.ndarray:
        """
        Internal helper to preprocess a time-series window into a feature vector.

        The preprocessing steps must mirror those used during model training.
        By default, this includes:
        - Decimation to match training sample rate.
        - Feature extraction (mean, std, FFT energy, etc.).
        - Feature scaling via the loaded scaler.

        Parameters
        ----------
        timestamps : list[float]
            Unix timestamps (seconds) for each sample.
        currents : list[float]
            Current measurements corresponding to timestamps.

        Returns
        -------
        np.ndarray
            2D array of shape (1, n_features), ready for model prediction.
        """
        if len(timestamps) != len(currents):
            raise ValueError("Timestamps and currents must have equal length")

        # Convert to numpy array
        arr = np.array(currents, dtype=np.float32)

        # Apply decimation (if implemented)
        arr_decimated = decimate_if_needed(arr)

        # Extract features (customizable via preprocess.py)
        feats = extract_features(arr_decimated)

        # Scale features using the fitted scaler (if available)
        if self.scaler is not None:
            feats_scaled = self.scaler.transform(feats.reshape(1, -1))
        else:
            feats_scaled = feats.reshape(1, -1)

        return feats_scaled

    # ----------------------------------------------------------------------
    def predict(self, timestamps: List[float], currents: List[float]) -> Tuple[int, float]:
        """
        Run model inference for a single time window.

        Parameters
        ----------
        timestamps : list[float]
            List of timestamps corresponding to the current measurements.
        currents : list[float]
            List of current measurements in amperes (same length as timestamps).

        Returns
        -------
        (int, float)
            Tuple (pred_label, prob_attack)
            - pred_label: 1 if model predicts attack, else 0.
            - prob_attack: probability/confidence for attack class (0.0–1.0).

        Raises
        ------
        RuntimeError
            If model has not been loaded before prediction.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() before predict().")

        try:
            # Preprocess input
            feats_scaled = self._preprocess(timestamps, currents)

            # Run inference
            probs = self.model.predict(feats_scaled, verbose=0)
            prob_attack = float(np.squeeze(probs))

            # Convert probability to binary label (threshold = 0.5)
            pred_label = int(prob_attack >= 0.5)

            logger.debug("Inference -> label=%d prob=%.4f", pred_label, prob_attack)
            return pred_label, prob_attack

        except Exception as exc:
            logger.exception("Prediction failed")
            raise RuntimeError("Prediction error") from exc
