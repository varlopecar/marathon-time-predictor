"""
API tests for the Marathon Time Predictor FastAPI application.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import json

from marathon_api import app


class TestMarathonAPI:
    """Test cases for Marathon API endpoints."""

    def setup_method(self):
        """Set up test fixtures."""
        self.client = TestClient(app)
        self.valid_prediction_request = {
            "distance_km": 42.2,
            "elevation_gain_m": 200,
            "mean_km_per_week": 60,
            "mean_training_days_per_week": 5,
            "gender": "male",
            "level": 2
        }

    def test_root_endpoint(self):
        """Test the root endpoint."""
        response = self.client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "endpoints" in data
        assert data["message"] == "Marathon Time Prediction API"

    @patch('marathon_api.model.load_model')
    @patch('marathon_api.model.is_trained')
    def test_health_check_model_loaded(self, mock_is_trained, mock_load_model):
        """Test health check when model is loaded."""
        mock_is_trained.__get__ = MagicMock(return_value=True)

        response = self.client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert data["model_type"] == "Random Forest"

    @patch('marathon_api.model.load_model')
    @patch('marathon_api.model.is_trained')
    def test_health_check_model_not_loaded(self, mock_is_trained, mock_load_model):
        """Test health check when model is not loaded."""
        mock_is_trained.__get__ = MagicMock(return_value=False)

        response = self.client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is False
        assert data["model_type"] == "Random Forest"

    @patch('marathon_api.model.predict_time')
    @patch('marathon_api.model.load_model')
    @patch('marathon_api.model.is_trained')
    def test_predict_endpoint_success(self, mock_is_trained, mock_load_model, mock_predict):
        """Test successful prediction endpoint."""
        mock_is_trained.__get__ = MagicMock(return_value=True)

        # Mock successful prediction
        mock_predict.return_value = {
            'success': True,
            'prediction': {
                'time_seconds': 14400.0,
                'time_minutes': 240.0,
                'time_hours': 4.0,
                'time_string': '04:00:00',
                'pace_minutes_per_km': 5.7,
                'distance_km': 42.2
            },
            'input_data': {
                'distance_m': 42200,
                'elevation_gain_m': 200,
                'mean_km_per_week': 60,
                'mean_training_days_per_week': 5,
                'gender_binary': 0,
                'level': 2
            },
            'model_info': {
                'model_type': 'Random Forest',
                'features_used': ['distance_m', 'elevation_gain_m', 'mean_km_per_week', 'mean_training_days_per_week', 'gender_binary', 'level'],
                'training_samples': '~30,000'
            }
        }

        response = self.client.post(
            "/predict", json=self.valid_prediction_request)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "prediction" in data
        assert "model_info" in data
        assert data["prediction"]["time_string"] == "04:00:00"
        assert data["prediction"]["pace_minutes_per_km"] == 5.7

    @patch('marathon_api.model.predict_time')
    @patch('marathon_api.model.load_model')
    @patch('marathon_api.model.is_trained')
    def test_predict_endpoint_validation_error(self, mock_is_trained, mock_load_model, mock_predict):
        """Test prediction endpoint with validation error."""
        mock_is_trained.__get__ = MagicMock(return_value=True)

        # Mock validation error
        mock_predict.return_value = {
            'success': False,
            'error': 'Invalid input: Distance must be between 0 and 500 km'
        }

        invalid_request = self.valid_prediction_request.copy()
        invalid_request['distance_km'] = -1

        response = self.client.post("/predict", json=invalid_request)

        assert response.status_code == 400
        data = response.json()
        assert "detail" in data
        assert "Invalid input" in data["detail"]

    @patch('marathon_api.model.predict_time')
    @patch('marathon_api.model.load_model')
    @patch('marathon_api.model.is_trained')
    def test_predict_endpoint_model_error(self, mock_is_trained, mock_load_model, mock_predict):
        """Test prediction endpoint with model error."""
        mock_is_trained.__get__ = MagicMock(return_value=True)

        # Mock model error
        mock_predict.return_value = {
            'success': False,
            'error': 'Model not trained. Please train the model first.'
        }

        response = self.client.post(
            "/predict", json=self.valid_prediction_request)

        assert response.status_code == 400
        data = response.json()
        assert "detail" in data
        assert "Model not trained" in data["detail"]

    @patch('marathon_api.model.predict_time')
    @patch('marathon_api.model.load_model')
    @patch('marathon_api.model.is_trained')
    def test_predict_endpoint_exception(self, mock_is_trained, mock_load_model, mock_predict):
        """Test prediction endpoint with exception."""
        mock_is_trained.__get__ = MagicMock(return_value=True)

        # Mock exception
        mock_predict.side_effect = Exception("Unexpected error")

        response = self.client.post(
            "/predict", json=self.valid_prediction_request)

        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
        assert "Prediction failed" in data["detail"]

    def test_predict_endpoint_missing_fields(self):
        """Test prediction endpoint with missing required fields."""
        incomplete_request = {
            "distance_km": 42.2,
            "gender": "male"
            # Missing other required fields
        }

        response = self.client.post("/predict", json=incomplete_request)

        # Should return 422 (Unprocessable Entity) for validation errors
        assert response.status_code == 422

    def test_predict_endpoint_invalid_data_types(self):
        """Test prediction endpoint with invalid data types."""
        invalid_request = {
            "distance_km": "not_a_number",
            "elevation_gain_m": 200,
            "mean_km_per_week": 60,
            "mean_training_days_per_week": 5,
            "gender": "male",
            "level": 2
        }

        response = self.client.post("/predict", json=invalid_request)

        assert response.status_code == 422

    def test_predict_endpoint_out_of_range_values(self):
        """Test prediction endpoint with out-of-range values."""
        invalid_request = self.valid_prediction_request.copy()
        invalid_request['distance_km'] = 1000  # Too high
        invalid_request['level'] = 5  # Invalid level

        response = self.client.post("/predict", json=invalid_request)

        assert response.status_code == 422

    @patch('marathon_api.model.predict_time')
    @patch('marathon_api.model.get_feature_importance')
    @patch('marathon_api.model.perform_cross_validation')
    @patch('marathon_api.model.load_model')
    @patch('marathon_api.model.is_trained')
    def test_predict_endpoint_with_model_info(self, mock_is_trained, mock_load_model, mock_cv, mock_importance, mock_predict):
        """Test prediction endpoint includes model information."""
        mock_is_trained.__get__ = MagicMock(return_value=True)

        # Mock feature importance
        mock_importance.return_value = {
            'mean_km_per_week': 0.45,
            'level': 0.25,
            'distance_m': 0.15,
            'mean_training_days_per_week': 0.10,
            'elevation_gain_m': 0.03,
            'gender_binary': 0.02
        }

        # Mock cross validation
        mock_cv.return_value = {
            'r2_mean': 0.85,
            'mae_mean': 900,
            'mse_mean': 1000000,
            'r2_std': 0.05,
            'mae_std': 50,
            'mse_std': 50000
        }

        # Mock successful prediction
        mock_predict.return_value = {
            'success': True,
            'prediction': {
                'time_seconds': 14400.0,
                'time_minutes': 240.0,
                'time_hours': 4.0,
                'time_string': '04:00:00',
                'pace_minutes_per_km': 5.7,
                'distance_km': 42.2
            },
            'input_data': {},
            'model_info': {}
        }

        response = self.client.post(
            "/predict", json=self.valid_prediction_request)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "feature_importance" in data
        assert "cross_validation" in data
        assert "model_info" in data

        # Check feature importance
        importance = data["feature_importance"]
        assert importance['mean_km_per_week'] == 0.45
        assert importance['level'] == 0.25

        # Check cross validation
        cv = data["cross_validation"]
        assert cv['r2_mean'] == 0.85
        assert cv['mae_mean'] == 900

    def test_predict_endpoint_different_genders(self):
        """Test prediction endpoint with different gender values."""
        test_cases = [
            {"gender": "male", "expected_binary": 0},
            {"gender": "female", "expected_binary": 1},
            {"gender": "M", "expected_binary": 0},
            {"gender": "F", "expected_binary": 1},
            {"gender": "1", "expected_binary": 0},
            {"gender": "0", "expected_binary": 1},
        ]

        for test_case in test_cases:
            request = self.valid_prediction_request.copy()
            request['gender'] = test_case['gender']

            # This will fail validation, but we can test the request format
            response = self.client.post("/predict", json=request)

            # Should not be a 422 error (validation error) for valid gender values
            if test_case['gender'] in ['male', 'female', 'M', 'F', '1', '0']:
                assert response.status_code != 422, f"Gender '{test_case['gender']}' should be valid"

    def test_predict_endpoint_different_levels(self):
        """Test prediction endpoint with different experience levels."""
        for level in [1, 2, 3]:
            request = self.valid_prediction_request.copy()
            request['level'] = level

            # This will fail validation, but we can test the request format
            response = self.client.post("/predict", json=request)

            # Should not be a 422 error for valid levels
            assert response.status_code != 422, f"Level {level} should be valid"


class TestMarathonAPIIntegration:
    """Integration tests for Marathon API."""

    def test_api_documentation_available(self):
        """Test that API documentation is available."""
        client = TestClient(app)

        # Test OpenAPI schema
        response = client.get("/openapi.json")
        assert response.status_code == 200

        # Test docs endpoint
        response = client.get("/docs")
        assert response.status_code == 200

    def test_api_schema_validation(self):
        """Test that API schema validation works correctly."""
        client = TestClient(app)

        # Get OpenAPI schema
        response = client.get("/openapi.json")
        schema = response.json()

        # Check that prediction endpoint is defined
        assert "/predict" in schema["paths"]
        assert "post" in schema["paths"]["/predict"]

        # Check that health endpoint is defined
        assert "/health" in schema["paths"]
        assert "get" in schema["paths"]["/health"]

    def test_request_response_schema(self):
        """Test that request and response schemas are properly defined."""
        client = TestClient(app)

        # Get OpenAPI schema
        response = client.get("/openapi.json")
        schema = response.json()

        # Check prediction request schema
        predict_path = schema["paths"]["/predict"]["post"]
        assert "requestBody" in predict_path
        assert "content" in predict_path["requestBody"]
        assert "application/json" in predict_path["requestBody"]["content"]

        # Check prediction response schema
        assert "responses" in predict_path
        assert "200" in predict_path["responses"]
        assert "400" in predict_path["responses"]
        assert "500" in predict_path["responses"]


if __name__ == "__main__":
    pytest.main([__file__])
