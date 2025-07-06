# Marathon Time Predictor API Documentation

## Overview

The Marathon Time Predictor API is a RESTful web service that provides marathon finish time predictions based on training data, race conditions, and athlete characteristics. The API is built with FastAPI and provides automatic documentation, input validation, and comprehensive error handling.

## Base URL

**Production (Live):**

```
https://marathon-time-predictor.osc-fr1.scalingo.io
```

**Local Development:**

```
http://localhost:8000
```

## Authentication

Currently, the API does not require authentication. All endpoints are publicly accessible.

## Endpoints

### 1. Root Endpoint

**GET** `/`

Returns basic information about the API.

#### Response

```json
{
  "message": "Marathon Time Prediction API",
  "version": "1.0.0",
  "status": "running",
  "model_ready": "true",
  "endpoints": "predict: POST /predict - Get marathon time prediction with model info; health: GET /health - Health check"
}
```

#### Example

```bash
# Production
curl -X GET "https://marathon-time-predictor.osc-fr1.scalingo.io/"

# Local development
curl -X GET "http://localhost:8000/"
```

### 2. Health Check

**GET** `/health`

Returns the health status of the API and model.

#### Response

```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_type": "Random Forest"
}
```

#### Response Fields

- `status`: Always "healthy" if the API is running
- `model_loaded`: Boolean indicating if the prediction model is loaded
- `model_type`: Type of machine learning model used

#### Example

```bash
# Production
curl -X GET "https://marathon-time-predictor.osc-fr1.scalingo.io/health"

# Local development
curl -X GET "http://localhost:8000/health"
```

### 3. Prediction Endpoint

**POST** `/predict`

Predicts marathon finish time based on input parameters.

#### Request Body

```json
{
  "distance_km": 42.2,
  "elevation_gain_m": 200,
  "mean_km_per_week": 60,
  "mean_training_days_per_week": 5,
  "gender": "male",
  "level": 2
}
```

#### Request Parameters

| Parameter                     | Type    | Range              | Description                                               |
| ----------------------------- | ------- | ------------------ | --------------------------------------------------------- |
| `distance_km`                 | float   | 0-500              | Race distance in kilometers                               |
| `elevation_gain_m`            | float   | 0-10000            | Total elevation gain in meters                            |
| `mean_km_per_week`            | float   | 0-200              | Average training volume in km per week                    |
| `mean_training_days_per_week` | float   | 0-7                | Average training days per week                            |
| `gender`                      | string  | "male" or "female" | Athlete gender                                            |
| `level`                       | integer | 1-3                | Experience level (1=beginner, 2=intermediate, 3=advanced) |

#### Response

**Success (200)**

```json
{
  "success": true,
  "prediction": {
    "time_seconds": 14400.0,
    "time_minutes": 240.0,
    "time_hours": 4.0,
    "time_string": "04:00:00",
    "pace_minutes_per_km": 5.7,
    "distance_km": 42.2
  },
  "model_info": {
    "model_type": "Random Forest",
    "features_used": [
      "distance_m",
      "elevation_gain_m",
      "mean_km_per_week",
      "mean_training_days_per_week",
      "gender_binary",
      "level"
    ],
    "training_samples": "~30,000",
    "model_performance": {
      "r2_score": 0.85,
      "mae_minutes": 15.0,
      "cross_validation_folds": 5
    }
  },
  "cross_validation": {
    "r2_mean": 0.85,
    "r2_std": 0.05,
    "mae_mean": 900,
    "mae_std": 50,
    "mse_mean": 1000000,
    "mse_std": 50000
  },
  "feature_importance": {
    "mean_km_per_week": 0.45,
    "level": 0.25,
    "distance_m": 0.15,
    "mean_training_days_per_week": 0.1,
    "elevation_gain_m": 0.03,
    "gender_binary": 0.02
  }
}
```

**Error (400)**

```json
{
  "detail": "Invalid input: Distance must be between 0 and 500 km"
}
```

**Error (500)**

```json
{
  "detail": "Prediction failed: Unexpected error occurred"
}
```

**Error (503)**

```json
{
  "detail": "Model is not ready. Please try again in a few moments."
}
```

#### Response Fields

**Prediction Object**

- `time_seconds`: Predicted finish time in seconds
- `time_minutes`: Predicted finish time in minutes
- `time_hours`: Predicted finish time in hours
- `time_string`: Formatted time string (HH:MM:SS)
- `pace_minutes_per_km`: Predicted pace in minutes per kilometer
- `distance_km`: Race distance in kilometers

**Model Info Object**

- `model_type`: Type of machine learning model
- `features_used`: List of features used for prediction
- `training_samples`: Number of training samples used
- `model_performance`: Model performance metrics

**Cross Validation Object**

- `r2_mean`: Mean R² score across cross-validation folds
- `r2_std`: Standard deviation of R² scores
- `mae_mean`: Mean absolute error in seconds
- `mae_std`: Standard deviation of MAE
- `mse_mean`: Mean squared error
- `mse_std`: Standard deviation of MSE

**Feature Importance Object**

- Key-value pairs where keys are feature names and values are importance scores (0-1)

#### Examples

**Basic Prediction**

```bash
# Production
curl -X POST "https://marathon-time-predictor.osc-fr1.scalingo.io/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "distance_km": 42.2,
    "elevation_gain_m": 200,
    "mean_km_per_week": 60,
    "mean_training_days_per_week": 5,
    "gender": "male",
    "level": 2
  }'

# Local development
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "distance_km": 42.2,
    "elevation_gain_m": 200,
    "mean_km_per_week": 60,
    "mean_training_days_per_week": 5,
    "gender": "male",
    "level": 2
  }'
```

## Error Codes

| Status Code | Description                                   |
| ----------- | --------------------------------------------- |
| 200         | Success                                       |
| 400         | Bad Request - Invalid input parameters        |
| 422         | Unprocessable Entity - Validation error       |
| 500         | Internal Server Error - Model or server error |
| 503         | Service Unavailable - Model not ready         |

## Rate Limiting

Currently, there are no rate limits implemented. However, please be respectful of the service and avoid making excessive requests.

## Model Information

### Algorithm

- **Type**: Random Forest Regressor
- **Estimators**: 100 decision trees
- **Cross-validation**: 5-fold
- **Training data**: ~30,000 marathon records

### Features

1. **Distance (m)**: Race distance converted to meters
2. **Elevation Gain (m)**: Total elevation gain during the race
3. **Training Volume (km/week)**: Average weekly training distance
4. **Training Frequency (days/week)**: Average training days per week
5. **Gender**: Binary encoding (0=male, 1=female)
6. **Experience Level**: Categorical (1=beginner, 2=intermediate, 3=advanced)

### Performance Metrics

- **R² Score**: ~0.85 (85% variance explained)
- **Mean Absolute Error**: ~15 minutes
- **Cross-validation**: 5-fold with consistent performance

## Usage Examples

### Python

```python
import requests

# Make prediction
response = requests.post(
    "http://localhost:8000/predict",
    json={
        "distance_km": 42.2,
        "elevation_gain_m": 200,
        "mean_km_per_week": 60,
        "mean_training_days_per_week": 5,
        "gender": "male",
        "level": 2
    }
)

if response.status_code == 200:
    result = response.json()
    prediction = result["prediction"]
    print(f"Predicted time: {prediction['time_string']}")
    print(f"Pace: {prediction['pace_minutes_per_km']} min/km")
else:
    print(f"Error: {response.json()['detail']}")
```

### JavaScript

```javascript
// Make prediction
const response = await fetch("http://localhost:8000/predict", {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
  },
  body: JSON.stringify({
    distance_km: 42.2,
    elevation_gain_m: 200,
    mean_km_per_week: 60,
    mean_training_days_per_week: 5,
    gender: "male",
    level: 2,
  }),
});

if (response.ok) {
  const result = await response.json();
  const prediction = result.prediction;
  console.log(`Predicted time: ${prediction.time_string}`);
  console.log(`Pace: ${prediction.pace_minutes_per_km} min/km`);
} else {
  const error = await response.json();
  console.error(`Error: ${error.detail}`);
}
```

## Interactive Documentation

The API provides automatic interactive documentation:

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`
- **OpenAPI Schema**: `http://localhost:8000/openapi.json`

## Support

For issues, questions, or feature requests:

- **GitHub Issues**: [Create an issue](https://github.com/yourusername/marathon-time-predictor/issues)
- **GitHub Discussions**: [Start a discussion](https://github.com/yourusername/marathon-time-predictor/discussions)
- **Email**: your.email@example.com
