"""
Unit tests for the MarathonPrediction class.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import tempfile
import os
import joblib

from marathon_prediction import MarathonPrediction


class TestMarathonPrediction:
    """Test cases for MarathonPrediction class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.model = MarathonPrediction()
        self.sample_data = {
            'distance_km': 42.2,
            'elevation_gain_m': 200,
            'mean_km_per_week': 60,
            'mean_training_days_per_week': 5,
            'gender': 'male',
            'level': 2
        }

    def test_init(self):
        """Test MarathonPrediction initialization."""
        assert self.model.model_path == "marathon_model.pkl"
        assert self.model.model is None
        assert self.model.is_trained is False
        assert len(self.model.feature_names) == 6
        assert 'distance_m' in self.model.feature_names
        assert 'elevation_gain_m' in self.model.feature_names

    def test_feature_names(self):
        """Test that feature names are correctly defined."""
        expected_features = [
            'distance_m', 'elevation_gain_m', 'mean_km_per_week',
            'mean_training_days_per_week', 'gender_binary', 'level'
        ]
        assert self.model.feature_names == expected_features

    @patch('pandas.read_csv')
    def test_load_and_prepare_data(self, mock_read_csv):
        """Test data loading and preparation."""
        # Mock data with European number format
        mock_data = pd.DataFrame({
            'distance_m': ['42,2', '21,1'],
            'elevation_gain_m': ['200,0', '100,0'],
            'mean_km_per_week': ['60,0', '30,0'],
            'mean_training_days_per_week': ['5,0', '3,0'],
            'elapsed_time_s': ['14400,0', '7200,0']
        })
        mock_read_csv.return_value = mock_data

        result = self.model.load_and_prepare_data("test.csv")

        # Check that European format was converted
        assert result['distance_m'].dtype == float
        assert result['distance_m'].iloc[0] == 42.2
        assert result['elevation_gain_m'].iloc[0] == 200.0

    def test_validate_user_input_valid(self):
        """Test validation with valid user input."""
        result = self.model.validate_user_input(self.sample_data)

        assert result['is_valid'] is True
        assert len(result['errors']) == 0
        assert result['processed_data']['distance_m'] == 42200  # km to m
        assert result['processed_data']['gender_binary'] == 0  # male
        assert result['processed_data']['level'] == 2

    def test_validate_user_input_invalid_distance(self):
        """Test validation with invalid distance."""
        invalid_data = self.sample_data.copy()
        invalid_data['distance_km'] = -1

        result = self.model.validate_user_input(invalid_data)

        assert result['is_valid'] is False
        assert "Distance must be between 0 and 500 km" in result['errors']

    def test_validate_user_input_invalid_elevation(self):
        """Test validation with invalid elevation."""
        invalid_data = self.sample_data.copy()
        invalid_data['elevation_gain_m'] = 15000

        result = self.model.validate_user_input(invalid_data)

        assert result['is_valid'] is False
        assert "Elevation gain must be between 0 and 10,000 meters" in result['errors']

    def test_validate_user_input_invalid_training_volume(self):
        """Test validation with invalid training volume."""
        invalid_data = self.sample_data.copy()
        invalid_data['mean_km_per_week'] = 250

        result = self.model.validate_user_input(invalid_data)

        assert result['is_valid'] is False
        assert "Training volume must be between 0 and 200 km/week" in result['errors']

    def test_validate_user_input_invalid_training_days(self):
        """Test validation with invalid training days."""
        invalid_data = self.sample_data.copy()
        invalid_data['mean_training_days_per_week'] = 8

        result = self.model.validate_user_input(invalid_data)

        assert result['is_valid'] is False
        assert "Training days must be between 0 and 7" in result['errors']

    def test_validate_user_input_invalid_gender(self):
        """Test validation with invalid gender."""
        invalid_data = self.sample_data.copy()
        invalid_data['gender'] = 'other'

        result = self.model.validate_user_input(invalid_data)

        assert result['is_valid'] is False
        assert "Gender must be 'male' or 'female'" in result['errors']

    def test_validate_user_input_invalid_level(self):
        """Test validation with invalid level."""
        invalid_data = self.sample_data.copy()
        invalid_data['level'] = 4

        result = self.model.validate_user_input(invalid_data)

        assert result['is_valid'] is False
        assert "Level must be 1 (beginner), 2 (intermediate), or 3 (advanced)" in result[
            'errors']

    def test_validate_user_input_female_gender(self):
        """Test validation with female gender."""
        female_data = self.sample_data.copy()
        female_data['gender'] = 'female'

        result = self.model.validate_user_input(female_data)

        assert result['is_valid'] is True
        assert result['processed_data']['gender_binary'] == 1

    def test_validate_user_input_missing_values(self):
        """Test validation with missing values."""
        incomplete_data = {
            'distance_km': 42.2,
            'gender': 'male'
            # Missing other required fields
        }

        result = self.model.validate_user_input(incomplete_data)

        assert result['is_valid'] is False
        assert len(result['errors']) > 0

    @patch('joblib.dump')
    def test_save_model(self, mock_dump):
        """Test model saving."""
        # Mock a model
        self.model.model = MagicMock()

        self.model.save_model()

        mock_dump.assert_called_once()

    def test_load_model_not_exists(self):
        """Test loading model when file doesn't exist."""
        # Use a non-existent path
        self.model.model_path = "non_existent_model.pkl"

        result = self.model.load_model()

        assert result is False
        assert self.model.is_trained is False

    @patch('joblib.load')
    def test_load_model_success(self, mock_load):
        """Test successful model loading."""
        # Mock file existence
        with patch('os.path.exists', return_value=True):
            mock_model = MagicMock()
            mock_load.return_value = mock_model

            result = self.model.load_model()

            assert result is True
            assert self.model.model == mock_model
            assert self.model.is_trained is True

    @patch('joblib.load')
    def test_load_model_exception(self, mock_load):
        """Test model loading with exception."""
        with patch('os.path.exists', return_value=True):
            mock_load.side_effect = Exception("Load error")

            result = self.model.load_model()

            assert result is False

    def test_get_feature_importance_no_model(self):
        """Test feature importance when no model is trained."""
        result = self.model.get_feature_importance()

        assert result == {}

    def test_get_feature_importance_with_model(self):
        """Test feature importance with trained model."""
        # Mock a model with feature importance
        mock_model = MagicMock()
        mock_model.feature_importances_ = np.array(
            [0.3, 0.2, 0.2, 0.1, 0.1, 0.1])
        self.model.model = mock_model
        self.model.is_trained = True

        result = self.model.get_feature_importance()

        assert len(result) == 6
        assert result['distance_m'] == 0.3
        assert result['elevation_gain_m'] == 0.2

    @patch('marathon_prediction.MarathonPrediction.load_and_prepare_data')
    @patch('sklearn.model_selection.train_test_split')
    @patch('sklearn.ensemble.RandomForestRegressor')
    def test_train_model(self, mock_rf, mock_split, mock_load_data):
        """Test model training."""
        # Mock data
        mock_data = pd.DataFrame({
            'distance_m': [42200, 21100],
            'elevation_gain_m': [200, 100],
            'mean_km_per_week': [60, 30],
            'mean_training_days_per_week': [5, 3],
            'gender_binary': [0, 1],
            'level': [2, 1],
            'elapsed_time_s': [14400, 7200]
        })
        mock_load_data.return_value = mock_data

        # Mock train/test split
        mock_split.return_value = (
            mock_data[self.model.feature_names][:1],  # X_train
            mock_data[self.model.feature_names][1:],  # X_test
            mock_data['elapsed_time_s'][:1],          # y_train
            mock_data['elapsed_time_s'][1:]           # y_test
        )

        # Mock Random Forest
        mock_rf_instance = MagicMock()
        mock_rf.return_value = mock_rf_instance
        mock_rf_instance.predict.return_value = np.array([7200])

        # Mock cross validation
        with patch.object(self.model, 'perform_cross_validation') as mock_cv:
            mock_cv.return_value = {
                'r2_mean': 0.85,
                'mae_mean': 900,
                'mse_mean': 1000000
            }

            # Mock save
            with patch.object(self.model, 'save_model'):
                result = self.model.train_model()

        assert result['r2_score'] is not None
        assert result['training_samples'] == 1
        assert result['test_samples'] == 1
        assert self.model.is_trained is True

    @patch('marathon_prediction.MarathonPrediction.load_model')
    def test_predict_time_model_not_loaded(self, mock_load):
        """Test prediction when model is not loaded."""
        mock_load.return_value = False

        result = self.model.predict_time(self.sample_data)

        assert result['success'] is False
        assert 'Model not trained' in result['error']

    @patch('marathon_prediction.MarathonPrediction.load_model')
    def test_predict_time_invalid_input(self, mock_load):
        """Test prediction with invalid input."""
        mock_load.return_value = True
        self.model.is_trained = True

        invalid_data = {'distance_km': -1}

        result = self.model.predict_time(invalid_data)

        assert result['success'] is False
        assert 'Invalid input' in result['error']

    @patch('marathon_prediction.MarathonPrediction.load_model')
    def test_predict_time_success(self, mock_load):
        """Test successful prediction."""
        mock_load.return_value = True
        self.model.is_trained = True

        # Mock model prediction
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([14400])  # 4 hours in seconds
        self.model.model = mock_model

        result = self.model.predict_time(self.sample_data)

        assert result['success'] is True
        assert result['prediction']['time_seconds'] == 14400
        assert result['prediction']['time_minutes'] == 240
        assert result['prediction']['time_hours'] == 4
        assert result['prediction']['time_string'] == "04:00:00"
        assert result['prediction']['pace_minutes_per_km'] == pytest.approx(
            5.69, rel=1e-2)

    @patch('marathon_prediction.MarathonPrediction.load_model')
    def test_predict_time_exception(self, mock_load):
        """Test prediction with exception."""
        mock_load.return_value = True
        self.model.is_trained = True

        # Mock model that raises exception
        mock_model = MagicMock()
        mock_model.predict.side_effect = Exception("Prediction error")
        self.model.model = mock_model

        result = self.model.predict_time(self.sample_data)

        assert result['success'] is False
        assert 'Prediction failed' in result['error']


class TestMarathonPredictionIntegration:
    """Integration tests for MarathonPrediction class."""

    def test_full_prediction_workflow(self):
        """Test the complete prediction workflow."""
        model = MarathonPrediction()

        # Test with valid input
        user_data = {
            'distance_km': 42.2,
            'elevation_gain_m': 200,
            'mean_km_per_week': 60,
            'mean_training_days_per_week': 5,
            'gender': 'male',
            'level': 2
        }

        # This will fail without a trained model, but we can test the validation
        validation = model.validate_user_input(user_data)
        assert validation['is_valid'] is True
        assert validation['processed_data']['distance_m'] == 42200

    def test_feature_importance_consistency(self):
        """Test that feature importance values are consistent."""
        model = MarathonPrediction()

        # Mock a model with known feature importance
        mock_model = MagicMock()
        mock_model.feature_importances_ = np.array(
            [0.3, 0.2, 0.2, 0.1, 0.1, 0.1])
        model.model = mock_model
        model.is_trained = True

        importance = model.get_feature_importance()

        # Check that all values sum to approximately 1.0
        total_importance = sum(importance.values())
        assert total_importance == pytest.approx(1.0, rel=1e-10)

        # Check that all values are non-negative
        assert all(value >= 0 for value in importance.values())


if __name__ == "__main__":
    pytest.main([__file__])
