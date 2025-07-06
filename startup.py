#!/usr/bin/env python3
"""
Startup script for Marathon Time Prediction API
This script provides better error handling and logging for deployment.
"""

import os
import sys
import logging
import threading
import time
from marathon_prediction import MarathonPrediction

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global model instance
model_instance = None
model_ready = False


def initialize_model():
    """Initialize the marathon prediction model."""
    global model_instance, model_ready

    try:
        logger.info("Initializing MarathonPrediction model...")

        # Try deployment model first (smaller, faster to load)
        deployment_model_path = "deployment_model.pkl"
        if os.path.exists(deployment_model_path):
            logger.info(
                "Found deployment model, using it for faster startup...")
            model_instance = MarathonPrediction(
                model_path=deployment_model_path)
        else:
            logger.info(
                "No deployment model found, using default model path...")
            model_instance = MarathonPrediction()

        # Try to load existing model
        if model_instance.load_model():
            logger.info("Model loaded successfully from disk")
            model_ready = True
            return model_instance
        else:
            logger.info("No saved model found, will train in background...")
            # Start training in background thread
            training_thread = threading.Thread(target=train_model_background)
            training_thread.daemon = True
            training_thread.start()
            return model_instance

    except Exception as e:
        logger.error(f"Error initializing model: {e}")
        # Return a model instance even if initialization fails
        # The API will handle this gracefully
        model_instance = MarathonPrediction()
        return model_instance


def train_model_background():
    """Train the model in a background thread."""
    global model_instance, model_ready

    try:
        logger.info("Starting background model training...")
        metrics = model_instance.train_model()
        model_ready = True
        logger.info(
            f"Background model training completed with metrics: {metrics}")
    except Exception as e:
        logger.error(f"Background model training failed: {e}")


if __name__ == "__main__":
    # Initialize model (non-blocking)
    model = initialize_model()

    # Import and run the FastAPI app
    from marathon_api import app
    import uvicorn

    # Get configuration from environment
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    log_level = os.environ.get("LOG_LEVEL", "info")

    logger.info(f"Starting FastAPI server on {host}:{port}")
    if not model_ready:
        logger.info(
            "API will start without trained model - predictions will be available once training completes")

    uvicorn.run(
        "marathon_api:app",
        host=host,
        port=port,
        reload=False,  # Disable reload in production
        log_level=log_level
    )
