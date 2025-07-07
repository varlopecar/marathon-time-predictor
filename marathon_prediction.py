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

    def train_model(self, data_path: str = "clean_dataset.csv", save_model: bool = True) -> Dict[str, float]:
        """
        Train the Random Forest model and save it.

        Args:
            data_path: Path to the training data
            save_model: Whether to save the trained model to disk

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

        # Save model (only if requested)
        if save_model:
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
            try:
                import joblib
                joblib.dump(self.model, self.model_path, compress=3)
                print(f"Model saved successfully with joblib to {self.model_path}")
            except ImportError:
                # Fallback to pickle if joblib is not available
                with open(self.model_path, 'wb') as f:
                    pickle.dump(self.model, f)
                print(f"Model saved successfully with pickle to {self.model_path}")

    def load_model(self) -> bool:
        """
        Load a previously trained model from disk.

        Returns:
            True if model loaded successfully, False otherwise
        """
        if os.path.exists(self.model_path):
            try:
                # Check file size to avoid loading extremely large files
                file_size = os.path.getsize(self.model_path)
                if file_size > 500 * 1024 * 1024:  # 500MB limit
                    print(
                        f"Warning: Model file is very large ({file_size / 1024 / 1024:.1f}MB)")

                # Try joblib first (for compressed models), then pickle
                try:
                    import joblib
                    self.model = joblib.load(self.model_path)
                    print(f"Model loaded successfully with joblib from {self.model_path}")
                except:
                    # Fallback to pickle
                    with open(self.model_path, 'rb') as f:
                        self.model = pickle.load(f)
                    print(f"Model loaded successfully with pickle from {self.model_path}")
                
                self.is_trained = True
                return True
            except Exception as e:
                print(f"Error loading model: {e}")
                return False
        else:
            print(f"Model file not found at {self.model_path}")
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

        # Prepare features for prediction with proper feature names
        features_df = pd.DataFrame([[
            validation['processed_data']['distance_m'],
            validation['processed_data']['elevation_gain_m'],
            validation['processed_data']['mean_km_per_week'],
            validation['processed_data']['mean_training_days_per_week'],
            validation['processed_data']['gender_binary'],
            validation['processed_data']['level']
        ]], columns=self.feature_names)

        # Make prediction
        try:
            predicted_seconds = self.model.predict(features_df)[0]

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

    def get_comprehensive_metrics(self, include_cv: bool = False) -> Dict[str, Any]:
        """
        Get comprehensive model metrics and data insights.

        Args:
            include_cv: Whether to include cross-validation results (can be slow)

        Returns:
            Dictionary with model metrics and data insights
        """
        if not self.is_trained or self.model is None:
            return {}

        # Get feature importance
        feature_importance = self.get_feature_importance()
        
        # Get cross-validation results (only if requested)
        cv_results = None
        if include_cv:
            try:
                cv_results = self.perform_cross_validation()
            except:
                cv_results = None

        # Load data for insights
        try:
            data = self.load_and_prepare_data()
            
            # Calculate correlations
            correlations = data[['distance_m', 'elevation_gain_m', 'mean_km_per_week', 
                               'mean_training_days_per_week', 'gender_binary', 'level', 'elapsed_time_s']].corr()['elapsed_time_s']
            
            # Gender analysis
            male_data = data[data['gender_binary'] == 0]
            female_data = data[data['gender_binary'] == 1]
            
            # Training volume analysis
            data['time_hours'] = data['elapsed_time_s'] / 3600
            data['training_group'] = pd.cut(data['mean_km_per_week'], 
                                           bins=[0, 30, 50, 70, 100, 200], 
                                           labels=['0-30', '30-50', '50-70', '70-100', '100+'])
            
            training_performance = data.groupby('training_group', observed=False)['time_hours'].agg(['mean', 'std', 'count']).to_dict()
            
            data_insights = {
                'dataset_size': len(data),
                'gender_distribution': {
                    'male_count': len(male_data),
                    'female_count': len(female_data),
                    'male_percentage': len(male_data) / len(data) * 100,
                    'female_percentage': len(female_data) / len(data) * 100
                },
                'feature_correlations': {
                    'distance_m': correlations.get('distance_m', 0),
                    'elevation_gain_m': correlations.get('elevation_gain_m', 0),
                    'mean_km_per_week': correlations.get('mean_km_per_week', 0),
                    'mean_training_days_per_week': correlations.get('mean_training_days_per_week', 0),
                    'gender_binary': correlations.get('gender_binary', 0),
                    'level': correlations.get('level', 0)
                },
                'training_volume_analysis': {
                    'mean_km_per_week': data['mean_km_per_week'].mean(),
                    'std_km_per_week': data['mean_km_per_week'].std(),
                    'min_km_per_week': data['mean_km_per_week'].min(),
                    'max_km_per_week': data['mean_km_per_week'].max(),
                    'training_performance_by_group': training_performance
                },
                'gender_performance': {
                    'male_mean_time_hours': male_data['elapsed_time_s'].mean() / 3600 if len(male_data) > 0 else 0,
                    'female_mean_time_hours': female_data['elapsed_time_s'].mean() / 3600 if len(female_data) > 0 else 0,
                    'male_mean_km_per_week': male_data['mean_km_per_week'].mean() if len(male_data) > 0 else 0,
                    'female_mean_km_per_week': female_data['mean_km_per_week'].mean() if len(female_data) > 0 else 0
                },
                'data_quality_warnings': []
            }
            
            # Add warnings for data quality issues
            training_correlation = correlations.get('mean_km_per_week', 0)
            if training_correlation > 0:
                data_insights['data_quality_warnings'].append(
                    f"Warning: Positive correlation ({training_correlation:.3f}) between training volume and finish time suggests data quality issues"
                )
            
            gender_correlation = correlations.get('gender_binary', 0)
            if abs(gender_correlation) < 0.01:
                data_insights['data_quality_warnings'].append(
                    f"Warning: Gender has minimal correlation ({gender_correlation:.3f}) with finish time"
                )
                
        except Exception as e:
            data_insights = {
                'error': f"Could not generate data insights: {str(e)}"
            }

        # Get comprehensive test set performance metrics (always available)
        try:
            # Load data and get test set performance
            data = self.load_and_prepare_data()
            X = data[self.feature_names]
            y = data['elapsed_time_s']
            
            # Use the same train/test split as during training
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Get predictions on test set
            y_pred = self.model.predict(X_test)
            
            # Calculate comprehensive test set metrics
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
            import numpy as np
            
            test_mae = mean_absolute_error(y_test, y_pred)
            test_mse = mean_squared_error(y_test, y_pred)
            test_rmse = np.sqrt(test_mse)
            test_r2 = r2_score(y_test, y_pred)
            
            # Calculate coefficient of determination (same as R² for linear models, but we'll show both)
            # For Random Forest, R² is the coefficient of determination
            coefficient_of_determination = test_r2
            
            test_performance = {
                'r2_score': test_r2,
                'coefficient_of_determination': coefficient_of_determination,
                'mae_seconds': test_mae,
                'mae_minutes': test_mae / 60,
                'mse': test_mse,
                'rmse_seconds': test_rmse,
                'rmse_minutes': test_rmse / 60,
                'test_samples': len(X_test)
            }
        except Exception as e:
            test_performance = {
                'r2_score': 'N/A',
                'coefficient_of_determination': 'N/A',
                'mae_seconds': 'N/A',
                'mae_minutes': 'N/A',
                'mse': 'N/A',
                'rmse_seconds': 'N/A',
                'rmse_minutes': 'N/A',
                'test_samples': 'N/A'
            }
        
        return {
            'feature_importance': feature_importance,
            'cross_validation': cv_results,
            'data_insights': data_insights,
            'model_performance_summary': {
                'r2_score': test_performance['r2_score'],
                'coefficient_of_determination': test_performance['coefficient_of_determination'],
                'mae_seconds': test_performance['mae_seconds'],
                'mae_minutes': test_performance['mae_minutes'],
                'mse': test_performance['mse'],
                'rmse_seconds': test_performance['rmse_seconds'],
                'rmse_minutes': test_performance['rmse_minutes'],
                'cv_folds': 5 if cv_results else 0,
                'training_samples': '~30,000',
                'test_samples': test_performance['test_samples']
            }
        }


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
