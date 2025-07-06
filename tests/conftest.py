"""
Pytest configuration and fixtures for Marathon Time Predictor tests.
"""

import pytest
import tempfile
import os
import pandas as pd
import numpy as np
from unittest.mock import MagicMock


@pytest.fixture
def sample_marathon_data():
    """Sample marathon data for testing."""
    return pd.DataFrame({
        'distance_m': [42200, 21100, 42195],
        'elevation_gain_m': [200, 100, 150],
        'mean_km_per_week': [60, 30, 45],
        'mean_training_days_per_week': [5, 3, 4],
        'gender_binary': [0, 1, 0],
        'level': [2, 1, 3],
        'elapsed_time_s': [14400, 7200, 10800]
    })


@pytest.fixture
def valid_user_input():
    """Valid user input for prediction testing."""
    return {
        'distance_km': 42.2,
        'elevation_gain_m': 200,
        'mean_km_per_week': 60,
        'mean_training_days_per_week': 5,
        'gender': 'male',
        'level': 2
    }


@pytest.fixture
def mock_trained_model():
    """Mock trained model for testing."""
    mock_model = MagicMock()
    mock_model.feature_importances_ = np.array([0.3, 0.2, 0.2, 0.1, 0.1, 0.1])
    mock_model.predict.return_value = np.array([14400])  # 4 hours in seconds
    return mock_model


@pytest.fixture
def temp_model_file():
    """Temporary file for model testing."""
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
        temp_path = f.name

    yield temp_path

    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def temp_data_file():
    """Temporary CSV file for data testing."""
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
        temp_path = f.name

    # Create sample data
    sample_data = pd.DataFrame({
        'distance_m': ['42,2', '21,1'],
        'elevation_gain_m': ['200,0', '100,0'],
        'mean_km_per_week': ['60,0', '30,0'],
        'mean_training_days_per_week': ['5,0', '3,0'],
        'elapsed_time_s': ['14400,0', '7200,0']
    })
    sample_data.to_csv(temp_path, sep=';', index=False)

    yield temp_path

    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def prediction_response():
    """Sample prediction response for testing."""
    return {
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


@pytest.fixture
def cross_validation_results():
    """Sample cross-validation results for testing."""
    return {
        'r2_mean': 0.85,
        'r2_std': 0.05,
        'mae_mean': 900,
        'mae_std': 50,
        'mse_mean': 1000000,
        'mse_std': 50000,
        'r2_scores': [0.82, 0.85, 0.88, 0.84, 0.86],
        'mae_scores': [850, 900, 950, 875, 925],
        'mse_scores': [950000, 1000000, 1050000, 975000, 1025000]
    }


@pytest.fixture
def feature_importance():
    """Sample feature importance for testing."""
    return {
        'mean_km_per_week': 0.45,
        'level': 0.25,
        'distance_m': 0.15,
        'mean_training_days_per_week': 0.10,
        'elevation_gain_m': 0.03,
        'gender_binary': 0.02
    }


@pytest.fixture
def invalid_inputs():
    """Various invalid inputs for testing validation."""
    return {
        'negative_distance': {
            'distance_km': -1,
            'elevation_gain_m': 200,
            'mean_km_per_week': 60,
            'mean_training_days_per_week': 5,
            'gender': 'male',
            'level': 2
        },
        'high_elevation': {
            'distance_km': 42.2,
            'elevation_gain_m': 15000,
            'mean_km_per_week': 60,
            'mean_training_days_per_week': 5,
            'gender': 'male',
            'level': 2
        },
        'invalid_gender': {
            'distance_km': 42.2,
            'elevation_gain_m': 200,
            'mean_km_per_week': 60,
            'mean_training_days_per_week': 5,
            'gender': 'other',
            'level': 2
        },
        'invalid_level': {
            'distance_km': 42.2,
            'elevation_gain_m': 200,
            'mean_km_per_week': 60,
            'mean_training_days_per_week': 5,
            'gender': 'male',
            'level': 5
        },
        'missing_fields': {
            'distance_km': 42.2,
            'gender': 'male'
            # Missing other required fields
        }
    }


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )


def pytest_collection_modifyitems(config, items):
    """Automatically mark tests based on their names."""
    for item in items:
        # Mark tests that contain "integration" in their name
        if "integration" in item.nodeid.lower():
            item.add_marker(pytest.mark.integration)
        # Mark tests that contain "slow" in their name
        elif "slow" in item.nodeid.lower():
            item.add_marker(pytest.mark.slow)
        # Mark all other tests as unit tests
        else:
            item.add_marker(pytest.mark.unit)
