"""
run_api.py — FastAPI Server Entry Point
========================================

Usage:
    python run_api.py

Or with uvicorn directly:
    uvicorn run_api:app --host 0.0.0.0 --port 8000 --reload
"""

import logging
import uvicorn
from api.app import create_app

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)

app = create_app()

if __name__ == "__main__":
    uvicorn.run(
        "run_api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,          # Set True during development
        log_level="info",
    )
