"""
Marathon Prediction FastAPI Application
Production-ready API for marathon time predictions.

Endpoints:
- POST /predict - Get marathon time prediction with model info
- GET /health - Health check
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import uvicorn
from marathon_prediction import MarathonPrediction
import os
from contextlib import asynccontextmanager

# Import the global model instance from startup
try:
    from startup import model_instance as model
    if model is None:
        model = MarathonPrediction()
except ImportError:
    # Fallback if startup module is not available
    model = MarathonPrediction()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan - model is already initialized in startup.py."""
    print("FastAPI application starting - model should be ready from startup.py")
    yield

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Marathon Time Prediction API",
    description="Predict marathon finish times based on training data and race conditions",
    version="1.0.0",
    lifespan=lifespan
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
        model_ready = False
    else:
        model_ready = model.is_trained and model.model is not None

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
            # Get additional model information
            feature_importance = model.get_feature_importance()

            # Get cross-validation results
            try:
                cv_results = model.perform_cross_validation()
            except:
                cv_results = None

            # Enhanced model info
            enhanced_model_info = {
                'model_type': 'Random Forest',
                'features_used': model.feature_names,
                'training_samples': '~30,000',
                'model_performance': {
                    'r2_score': cv_results['r2_mean'] if cv_results else 'N/A',
                    'mae_minutes': cv_results['mae_mean'] / 60 if cv_results else 'N/A',
                    'cross_validation_folds': 5
                }
            }

            return PredictionResponse(
                success=True,
                prediction=result['prediction'],
                model_info=enhanced_model_info,
                cross_validation=cv_results,
                feature_importance=feature_importance
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
