# ========= IoT Simulator Dockerfile =========
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /sim

# Copy simulator code
COPY sim/ ./sim/

# Install Python dependencies from sim/requirements.txt
RUN pip install --no-cache-dir -r sim/requirements.txt

# Default environment variables (can be overridden at runtime)
ENV CENTRAL_HOST=http://central:8000
ENV NUM_NODES=10
ENV MIN_DELAY=0
ENV MAX_DELAY=60

# Run the main simulation script
CMD ["python", "sim/main.py"]
