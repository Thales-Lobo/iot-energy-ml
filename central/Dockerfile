# ========= Central Node Dockerfile =========
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy code and requirements
COPY app/ ./app/
COPY model/ ./model/

# Install dependencies
RUN pip install --no-cache-dir -r app/requirements.txt

# Expose API port
EXPOSE 8000

# Environment defaults
ENV MODEL_PATH=/app/model/model.h5
ENV SCALER_PATH=/app/model/scaler.pkl
ENV METRICS_CSV_PATH=/app/data/results.csv

# Create data folder
RUN mkdir -p /app/data

# Run FastAPI with Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
