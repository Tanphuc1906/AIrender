#!/bin/bash
# AI Image Studio - Linux/Mac start script

set -e

echo "============================================"
echo "   AI Image Studio - Starting Server"
echo "============================================"

# Create venv if not exists
if [ ! -d "venv" ]; then
    echo "[INFO] Creating virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate

echo "[INFO] Installing/checking dependencies..."
pip install -r requirements.txt -q

mkdir -p models outputs

echo ""
echo "[OK] Ready!"
echo "--------------------------------------------"
echo "  Server: http://localhost:8000"
echo "  API docs: http://localhost:8000/docs"
echo "--------------------------------------------"
echo ""

python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
