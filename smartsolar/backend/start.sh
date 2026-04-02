#!/bin/bash
# start.sh — Render Web Service start command
# Downloads models from Google Drive, then launches FastAPI

set -e

echo "=== SmartSolar Backend Startup ==="

# Step 1: Download models if not already present
python download_models.py

# Step 2: Launch FastAPI (Render sets PORT env var)
echo "=== Starting FastAPI on port ${PORT:-8000} ==="
uvicorn main:app --host 0.0.0.0 --port "${PORT:-8000}"
