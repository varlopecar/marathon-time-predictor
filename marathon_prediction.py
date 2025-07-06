"""
Marathon Time Prediction API Model
A simplified version designed to accept user input from forms and predict race times.

This model:
1. Trains on the clean dataset
2. Saves the trained model for reuse
3. Provides a simple prediction function for user input
4. Includes input validation and preprocessing
5. Returns predictions in a user-friendly format
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict, Any, Optional


class MarathonPrediction:
    def __init__(self, model_path: str = "marathon_model.pkl"):
        """
        Initialize the marathon prediction API model.

        Args:
            model_path: Path to save/load the trained model
        """
        self.model_path = model_path
        self.model = None
        self.feature_names = [
            'distance_m', 'elevation_gain_m', 'mean_km_per_week',
            'mean_training_days_per_week', 'gender_binary', 'level'
        ]
        self.is_trained = False

    def load_and_prepare_data(self, data_path: str = "clean_dataset.csv") -> pd.DataFrame:
        """
        Load and prepare the training data.

        Args:
            data_path: Path to the clean dataset

        Returns:
            Prepared DataFrame
        """
        # Load the data
        data = pd.read_csv(data_path, sep=';')

        # Convert European number format to standard format
        numeric_columns = ['distance_m', 'elevation_gain_m', 'mean_km_per_week',
                           'mean_training_days_per_week', 'elapsed_time_s']

        for col in numeric_columns:
            if col in data.columns:
                data[col] = data[col].astype(
                    str).str.replace(',', '.').astype(float)

        return data

    def train_model(self, data_path: str = "clean_dataset.csv") -> Dict[str, float]:
        """
        Train the Random Forest model and save it.

        Args:
            data_path: Path to the training data

        Returns:
            Dictionary with training metrics
        """
        # Load and prepare data
        data = self.load_and_prepare_data(data_path)

        # Prepare features and target
        X = data[self.feature_names]
        y = data['elapsed_time_s']

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train model
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)

        # Evaluate model
        y_pred = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Perform cross-validation
        cv_results = self.perform_cross_validation()

        # Save model
        self.save_model()

        self.is_trained = True

        metrics = {
            'mae_seconds': mae,
            'mae_minutes': mae / 60,
            'mse': mse,
            'r2_score': r2,
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'cross_validation': cv_results
        }

        return metrics

    def perform_cross_validation(self, cv_folds: int = 5, data_path: str = "clean_dataset.csv") -> Dict[str, Any]:
        """
        Perform cross-validation to get robust model performance estimates.

        Args:
            cv_folds: Number of cross-validation folds
            data_path: Path to the training data

        Returns:
            Dictionary with cross-validation results
        """
        # Load and prepare data
        data = self.load_and_prepare_data(data_path)

        # Prepare features and target for cross-validation
        X = data[self.feature_names]
        y = data['elapsed_time_s']

        # Perform cross-validation with different metrics
        # R² Score cross-validation
        cv_r2_scores = cross_val_score(
            self.model, X, y, cv=cv_folds, scoring='r2')

        # Mean Absolute Error cross-validation (negative because sklearn maximizes)
        cv_mae_scores = -cross_val_score(self.model,
                                         X, y, cv=cv_folds, scoring='neg_mean_absolute_error')

        # Mean Squared Error cross-validation (negative because sklearn maximizes)
        cv_mse_scores = -cross_val_score(self.model,
                                         X, y, cv=cv_folds, scoring='neg_mean_squared_error')

        # Calculate statistics
        cv_results = {
            'r2_mean': cv_r2_scores.mean(),
            'r2_std': cv_r2_scores.std(),
            'mae_mean': cv_mae_scores.mean(),
            'mae_std': cv_mae_scores.std(),
            'mse_mean': cv_mse_scores.mean(),
            'mse_std': cv_mse_scores.std(),
            'r2_scores': cv_r2_scores.tolist(),
            'mae_scores': cv_mae_scores.tolist(),
            'mse_scores': cv_mse_scores.tolist()
        }

        return cv_results

    def save_model(self):
        """Save the trained model to disk."""
        if self.model is not None:
            with open(self.model_path, 'wb') as f:
                pickle.dump(self.model, f)

    def load_model(self) -> bool:
        """
        Load a previously trained model from disk.

        Returns:
            True if model loaded successfully, False otherwise
        """
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, 'rb') as f:
                    self.model = pickle.load(f)
                self.is_trained = True
                return True
            except Exception as e:
                return False
        else:
            return False

    def validate_user_input(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and preprocess user input from a form.

        Args:
            user_data: Dictionary containing user input

        Returns:
            Dictionary with validated and processed features
        """
        errors = []
        processed_data = {}

        # Validate distance (in kilometers, convert to meters)
        try:
            distance_km = float(user_data.get('distance_km', 0))
            if distance_km <= 0 or distance_km > 500:  # Reasonable range
                errors.append("Distance must be between 0 and 500 km")
            processed_data['distance_m'] = distance_km * 1000
        except (ValueError, TypeError):
            errors.append("Invalid distance value")

        # Validate elevation gain (in meters)
        try:
            elevation = float(user_data.get('elevation_gain_m', 0))
            if elevation < 0 or elevation > 10000:  # Reasonable range
                errors.append(
                    "Elevation gain must be between 0 and 10,000 meters")
            processed_data['elevation_gain_m'] = elevation
        except (ValueError, TypeError):
            errors.append("Invalid elevation gain value")

        # Validate training volume (km per week)
        try:
            km_per_week = float(user_data.get('mean_km_per_week', 0))
            if km_per_week <= 0 or km_per_week > 200:  # Reasonable range
                errors.append(
                    "Training volume must be between 0 and 200 km/week")
            processed_data['mean_km_per_week'] = km_per_week
        except (ValueError, TypeError):
            errors.append("Invalid training volume value")

        # Validate training frequency (days per week)
        try:
            days_per_week = float(user_data.get(
                'mean_training_days_per_week', 0))
            if days_per_week <= 0 or days_per_week > 7:
                errors.append("Training days must be between 0 and 7")
            processed_data['mean_training_days_per_week'] = days_per_week
        except (ValueError, TypeError):
            errors.append("Invalid training days value")

        # Validate gender (convert to binary)
        gender = user_data.get('gender', '').lower()
        if gender in ['male', 'm', '1']:
            processed_data['gender_binary'] = 0
        elif gender in ['female', 'f', '0']:
            processed_data['gender_binary'] = 1
        else:
            errors.append("Gender must be 'male' or 'female'")

        # Validate experience level
        try:
            level = int(user_data.get('level', 0))
            if level not in [1, 2, 3]:  # 1=beginner, 2=intermediate, 3=advanced
                errors.append(
                    "Level must be 1 (beginner), 2 (intermediate), or 3 (advanced)")
            processed_data['level'] = level
        except (ValueError, TypeError):
            errors.append("Invalid level value")

        return {
            'processed_data': processed_data,
            'errors': errors,
            'is_valid': len(errors) == 0
        }

    def predict_time(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict marathon time based on user input.

        Args:
            user_data: Dictionary containing user input from form

        Returns:
            Dictionary with prediction results and metadata
        """
        # Ensure model is loaded
        if not self.is_trained:
            if not self.load_model():
                return {
                    'error': 'Model not trained. Please train the model first.',
                    'success': False
                }

        # Validate user input
        validation = self.validate_user_input(user_data)
        if not validation['is_valid']:
            return {
                'error': f"Invalid input: {'; '.join(validation['errors'])}",
                'success': False
            }

        # Prepare features for prediction
        features = np.array([[
            validation['processed_data']['distance_m'],
            validation['processed_data']['elevation_gain_m'],
            validation['processed_data']['mean_km_per_week'],
            validation['processed_data']['mean_training_days_per_week'],
            validation['processed_data']['gender_binary'],
            validation['processed_data']['level']
        ]])

        # Make prediction
        try:
            predicted_seconds = self.model.predict(features)[0]

            # Convert to different time formats
            predicted_minutes = predicted_seconds / 60
            predicted_hours = predicted_minutes / 60

            # Format time strings
            hours = int(predicted_hours)
            minutes = int(predicted_minutes % 60)
            seconds = int(predicted_seconds % 60)

            time_string = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

            # Calculate pace (minutes per kilometer)
            distance_km = validation['processed_data']['distance_m'] / 1000
            pace_minutes_per_km = predicted_minutes / distance_km

            return {
                'success': True,
                'prediction': {
                    'time_seconds': round(predicted_seconds, 2),
                    'time_minutes': round(predicted_minutes, 2),
                    'time_hours': round(predicted_hours, 2),
                    'time_string': time_string,
                    'pace_minutes_per_km': round(pace_minutes_per_km, 2),
                    'distance_km': round(distance_km, 2)
                },
                'input_data': validation['processed_data'],
                'model_info': {
                    'model_type': 'Random Forest',
                    'features_used': self.feature_names,
                    'training_samples': '~30,000'  # Approximate
                }
            }

        except Exception as e:
            return {
                'error': f"Prediction failed: {str(e)}",
                'success': False
            }

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance from the trained model.

        Returns:
            Dictionary with feature names and their importance scores
        """
        if not self.is_trained or self.model is None:
            return {}

        importance = self.model.feature_importances_
        return dict(zip(self.feature_names, importance))


# Example usage and testing
if __name__ == "__main__":
    # Initialize the API model
    api_model = MarathonPrediction()

    # Train the model (only needed once)
    print("Training model...")
    metrics = api_model.train_model()

    # Example user input (like from a web form)
    example_user_data = {
        'distance_km': 42.2,  # Marathon distance
        'elevation_gain_m': 200,  # Moderate elevation
        'mean_km_per_week': 60,  # 60 km per week training
        'mean_training_days_per_week': 5,  # 5 days per week
        'gender': 'male',
        'level': 2  # Intermediate
    }

    # Make prediction
    result = api_model.predict_time(example_user_data)

    if result['success']:
        pred = result['prediction']

    else:
        print(f"❌ Error: {result['error']}")

    # Show feature importance
    importance = api_model.get_feature_importance()
    for feature, score in sorted(importance.items(), key=lambda x: x[1], reverse=True):
        print(f"{feature}: {score:.3f}")
