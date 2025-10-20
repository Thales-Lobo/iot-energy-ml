# simulator/sim/config.py
from dotenv import load_dotenv
import os

# Read threshold (default 0.5 if not defined)
load_dotenv()

CENTRAL_HOST = os.getenv("CENTRAL_HOST", "http://localhost:8000")
NUM_NODES = int(os.getenv("NUM_NODES", 5))
MIN_DELAY = float(os.getenv("MIN_DELAY", 0))
MAX_DELAY = float(os.getenv("MAX_DELAY", 10))
