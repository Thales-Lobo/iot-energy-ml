# simulator/sim/sim_main.py
import logging
import time
import random
from sim.config import NUM_NODES, MIN_DELAY, MAX_DELAY
from sim.node import Node

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("simulator.sim_main")

def main_loop():
    """Main loop: generate and send windows for all nodes."""
    node_ids = [f"node_{i+1}" for i in range(NUM_NODES)]
    nodes = [Node(node_id) for node_id in node_ids]

    logger.info("Starting simulator with %d nodes, delay range %.1f-%.1f sec", NUM_NODES, MIN_DELAY, MAX_DELAY)

    try:
        while True:
            for node in nodes:
                window = node.generate_window()
                node.send_window(window)
                delay = random.uniform(MIN_DELAY, MAX_DELAY)
                time.sleep(delay)
    except KeyboardInterrupt:
        logger.info("Simulator stopped by user.")


if __name__ == "__main__":
    main_loop()
