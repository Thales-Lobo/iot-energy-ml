# simulator/sim/node.py
import time
import random
import logging
import requests
from pydantic import BaseModel, Field, validator
from typing import List
from sim.config import CENTRAL_HOST

logger = logging.getLogger("simulator.node")

class NodeWindow(BaseModel):
    node_id: str
    timestamps: List[float]
    currents: List[float]
    is_attack: bool = False

    @validator("timestamps")
    def timestamps_non_empty(cls, v):
        if not v:
            raise ValueError("timestamps must not be empty")
        return v

    @validator("currents")
    def currents_non_empty(cls, v):
        if not v:
            raise ValueError("currents must not be empty")
        return v

    @validator("currents")
    def lengths_match(cls, currents, values):
        timestamps = values.get("timestamps")
        if timestamps is not None and len(timestamps) != len(currents):
            raise ValueError("timestamps and currents must have the same length")
        return currents

class Node:
    """Represents a single simulated IoT node."""
    
    def __init__(self, node_id: str):
        self.node_id = node_id

    def generate_window(self) -> NodeWindow:
        """Generate synthetic measurement window."""
        n_samples = random.randint(2, 5)
        now = time.time()
        timestamps = [now + i for i in range(n_samples)]
        currents = [round(random.uniform(0, 10), 2) for _ in range(n_samples)]
        is_attack = random.choice([False, False, True])
        return NodeWindow(node_id=self.node_id, timestamps=timestamps, currents=currents, is_attack=is_attack)

    def send_window(self, window: NodeWindow):
        """Send window to central server."""
        try:
            payload = window.dict()
            resp = requests.post(f"{CENTRAL_HOST}/predict", json=payload)
            resp.raise_for_status()
            resp_json = resp.json()
            logger.info("Node=%s sent window. Response pred=%s prob=%.4f correct=%s",
                        self.node_id, resp_json.get("pred_label"),
                        resp_json.get("prob_attack"),
                        resp_json.get("correct"))
        except Exception as exc:
            logger.error("Failed to send data for node=%s: %s", self.node_id, exc)
