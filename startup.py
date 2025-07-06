#!/usr/bin/env python3
"""
Startup script for Marathon Time Prediction API
This script provides better error handling and logging for deployment.
"""

import os
import sys
import logging
from marathon_prediction import MarathonPrediction

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def initialize_model():
    """Initialize the marathon prediction model."""
    try:
        logger.info("Initializing MarathonPrediction model...")
        model = MarathonPrediction()

        # Try to load existing model
        if model.load_model():
            logger.info("Model loaded successfully from disk")
            return model
        else:
            logger.info("No saved model found, training new model...")
            # Train new model
            metrics = model.train_model()
            logger.info(f"Model trained successfully with metrics: {metrics}")
            return model

    except Exception as e:
        logger.error(f"Error initializing model: {e}")
        # Return a model instance even if initialization fails
        # The API will handle this gracefully
        return MarathonPrediction()


if __name__ == "__main__":
    # Initialize model
    model = initialize_model()

    # Import and run the FastAPI app
    from marathon_api import app
    import uvicorn

    # Get configuration from environment
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    log_level = os.environ.get("LOG_LEVEL", "info")

    logger.info(f"Starting FastAPI server on {host}:{port}")

    uvicorn.run(
        "marathon_api:app",
        host=host,
        port=port,
        reload=False,  # Disable reload in production
        log_level=log_level
    )
