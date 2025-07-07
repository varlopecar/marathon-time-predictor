"""
Marathon Prediction FastAPI Application
Production-ready API for marathon time predictions.

Endpoints:
- POST /predict - Get marathon time prediction with model info
- GET /health - Health check
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import uvicorn
from marathon_prediction import MarathonPrediction
import os
from contextlib import asynccontextmanager
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model instance
model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the model when the application starts."""
    global model

    try:
        logger.info("Starting model initialization...")

        # Try deployment model first (smaller, faster to load)
        deployment_model_path = "deployment_model.pkl"
        if os.path.exists(deployment_model_path):
            logger.info(
                "Found deployment model, using it for faster startup...")
            model = MarathonPrediction(model_path=deployment_model_path)
        else:
            logger.info(
                "No deployment model found, using default model path...")
            model = MarathonPrediction()

        # Try to load existing model
        if model.load_model():
            logger.info("Model loaded successfully from disk")
        else:
            logger.info("No saved model found, training new model...")
            # Train new model
            metrics = model.train_model()
            logger.info(f"Model trained successfully with metrics: {metrics}")

        logger.info("Model initialization completed successfully")
    except Exception as e:
        logger.error(f"Error during model initialization: {e}")
        # Continue without model - endpoints will handle this gracefully
        model = MarathonPrediction()

    yield

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Marathon Time Prediction API",
    description="Predict marathon finish times based on training data and race conditions",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware with more comprehensive configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://marathon-predictor.lovable.app",
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:8080",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:8080",
        # Add wildcard for development (remove in production)
        "*"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD"],
    allow_headers=[
        "Accept",
        "Accept-Language",
        "Content-Language",
        "Content-Type",
        "Authorization",
        "X-Requested-With",
        "Origin",
        "Access-Control-Request-Method",
        "Access-Control-Request-Headers",
        "Cache-Control",
        "Pragma"
    ],
    expose_headers=["Content-Length", "Content-Type"],
    max_age=86400,  # Cache preflight requests for 24 hours
)

# Pydantic models for request/response validation


class PredictionRequest(BaseModel):
    distance_km: float = Field(..., ge=0, le=500,
                               description="Race distance in kilometers")
    elevation_gain_m: float = Field(..., ge=0, le=10000,
                                    description="Total elevation gain in meters")
    mean_km_per_week: float = Field(..., ge=0, le=200,
                                    description="Average training volume in km per week")
    mean_training_days_per_week: float = Field(
        ..., ge=0, le=7, description="Average training days per week")
    gender: str = Field(..., description="Gender: 'male' or 'female'")
    level: int = Field(..., ge=1, le=3,
                       description="Experience level: 1=beginner, 2=intermediate, 3=advanced")


class PredictionResponse(BaseModel):
    success: bool
    prediction: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    model_info: Optional[Dict[str, Any]] = None
    cross_validation: Optional[Dict[str, Any]] = None
    feature_importance: Optional[Dict[str, float]] = None
    data_insights: Optional[Dict[str, Any]] = None

    model_config = {"protected_namespaces": ()}


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_type: str

    model_config = {"protected_namespaces": ()}


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    # Ensure model is not None
    if model is None:
        model_ready = "false"
    else:
        model_ready = "true" if (
            model.is_trained and model.model is not None) else "false"

    return {
        "message": "Marathon Time Prediction API",
        "version": "1.0.0",
        "status": "running",
        "model_ready": model_ready,
        "endpoints": "predict: POST /predict - Get marathon time prediction with model info; health: GET /health - Health check"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        # Check if model is loaded and working
        if model is not None and model.is_trained and model.model is not None:
            # Simple health check without making a prediction to avoid delays
            model_working = True
        else:
            model_working = False

        return HealthResponse(
            status="healthy" if model_working else "degraded",
            model_loaded=model.is_trained if model is not None else False,
            model_type="Random Forest"
        )
    except Exception as e:
        return HealthResponse(
            status="unhealthy",
            model_loaded=False,
            model_type="Random Forest"
        )


@app.post("/predict", response_model=PredictionResponse)
async def predict_marathon_time(request: PredictionRequest):
    """
    Predict marathon finish time based on user input.
    Includes model information, cross-validation results, and feature importance.

    Returns:
        PredictionResponse with predicted time and comprehensive model information
    """
    try:
        # Check if model is loaded
        if model is None or not model.is_trained or model.model is None:
            raise HTTPException(
                status_code=503,
                detail="Model is not ready. Please try again in a few moments."
            )

        # Convert request to dictionary
        user_data = {
            'distance_km': request.distance_km,
            'elevation_gain_m': request.elevation_gain_m,
            'mean_km_per_week': request.mean_km_per_week,
            'mean_training_days_per_week': request.mean_training_days_per_week,
            'gender': request.gender,
            'level': request.level
        }

        # Make prediction
        result = model.predict_time(user_data)

        if result['success']:
            # Get comprehensive metrics and insights (without cross-validation for speed)
            comprehensive_metrics = model.get_comprehensive_metrics(include_cv=False)

            # Enhanced model info
            enhanced_model_info = {
                'model_type': 'Random Forest',
                'features_used': model.feature_names,
                'training_samples': comprehensive_metrics.get('model_performance_summary', {}).get('training_samples', '~30,000'),
                'model_performance': comprehensive_metrics.get('model_performance_summary', {})
            }

            return PredictionResponse(
                success=True,
                prediction=result['prediction'],
                model_info=enhanced_model_info,
                cross_validation=comprehensive_metrics.get('cross_validation'),
                feature_importance=comprehensive_metrics.get('feature_importance'),
                data_insights=comprehensive_metrics.get('data_insights')
            )
        else:
            raise HTTPException(status_code=400, detail=result['error'])

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Prediction failed: {str(e)}")





if __name__ == "__main__":
    # Run the FastAPI application
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    log_level = os.environ.get("LOG_LEVEL", "info")

    uvicorn.run(
        "marathon_api:app",
        host=host,
        port=port,
        reload=False,  # Disable reload in production
        log_level=log_level
    )
