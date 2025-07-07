#!/usr/bin/env python3
"""
Test script to evaluate marathon prediction model performance
using raw data format with limited features.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class RawDataMarathonPrediction:
    """Simplified marathon prediction model for raw data format."""
    
    def __init__(self):
        self.model = None
        self.feature_names = ['distance_m', 'elevation_gain_m', 'gender_binary']
        self.is_trained = False
    
    def load_raw_data(self, data_path: str = "raw-data-kaggle.csv") -> pd.DataFrame:
        """Load and prepare raw data."""
        print(f"Loading raw data from {data_path}...")
        
        # Load the data
        data = pd.read_csv(data_path, sep=';')
        
        # Convert European number format to standard format
        numeric_columns = ['distance (m)', 'elapsed time (s)', 'elevation gain (m)', 'average heart rate (bpm)']
        
        for col in numeric_columns:
            if col in data.columns:
                data[col] = data[col].astype(str).str.replace(',', '.').astype(float)
        
        # Clean and prepare features
        data_clean = data.copy()
        
        # Convert gender to binary (M=0, F=1)
        data_clean['gender_binary'] = (data_clean['gender'] == 'F').astype(int)
        
        # Rename columns to match expected format
        data_clean = data_clean.rename(columns={
            'distance (m)': 'distance_m',
            'elapsed time (s)': 'elapsed_time_s',
            'elevation gain (m)': 'elevation_gain_m'
        })
        
        # Filter for reasonable marathon distances (35-50 km)
        marathon_data = data_clean[
            (data_clean['distance_m'] >= 35000) & 
            (data_clean['distance_m'] <= 50000)
        ].copy()
        
        print(f"Raw data shape: {data.shape}")
        print(f"Marathon data shape (35-50km): {marathon_data.shape}")
        
        return marathon_data
    
    def analyze_raw_data(self, data: pd.DataFrame):
        """Analyze the raw data distribution."""
        print("\n=== RAW DATA ANALYSIS ===")
        
        print(f"Total records: {len(data)}")
        print(f"Unique athletes: {data['athlete'].nunique()}")
        
        # Gender distribution
        gender_counts = data['gender_binary'].value_counts()
        print(f"\nGender distribution:")
        print(f"Male (0): {gender_counts.get(0, 0)} ({gender_counts.get(0, 0)/len(data)*100:.1f}%)")
        print(f"Female (1): {gender_counts.get(1, 0)} ({gender_counts.get(1, 0)/len(data)*100:.1f}%)")
        
        # Distance analysis
        print(f"\nDistance analysis:")
        print(f"Mean distance: {data['distance_m'].mean()/1000:.2f} km")
        print(f"Std distance: {data['distance_m'].std()/1000:.2f} km")
        print(f"Min distance: {data['distance_m'].min()/1000:.2f} km")
        print(f"Max distance: {data['distance_m'].max()/1000:.2f} km")
        
        # Time analysis
        print(f"\nTime analysis:")
        print(f"Mean time: {data['elapsed_time_s'].mean()/3600:.2f} hours")
        print(f"Std time: {data['elapsed_time_s'].std()/3600:.2f} hours")
        print(f"Min time: {data['elapsed_time_s'].min()/3600:.2f} hours")
        print(f"Max time: {data['elapsed_time_s'].max()/3600:.2f} hours")
        
        # Elevation analysis
        print(f"\nElevation analysis:")
        print(f"Mean elevation: {data['elevation_gain_m'].mean():.1f} m")
        print(f"Std elevation: {data['elevation_gain_m'].std():.1f} m")
        print(f"Min elevation: {data['elevation_gain_m'].min():.1f} m")
        print(f"Max elevation: {data['elevation_gain_m'].max():.1f} m")
        
        # Check correlations
        print(f"\nFeature correlations with finish time:")
        correlations = data[['distance_m', 'elevation_gain_m', 'gender_binary', 'elapsed_time_s']].corr()['elapsed_time_s']
        for feature, corr in correlations.items():
            if feature != 'elapsed_time_s':
                print(f"{feature}: {corr:.3f}")
    
    def train_model(self, data: pd.DataFrame) -> dict:
        """Train the model on raw data."""
        print("\n=== TRAINING MODEL ON RAW DATA ===")
        
        # Prepare features and target
        X = data[self.feature_names]
        y = data['elapsed_time_s']
        
        print(f"Features used: {self.feature_names}")
        print(f"Training samples: {len(X)}")
        
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
        cv_results = self.perform_cross_validation(X, y)
        
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
        
        print(f"\nModel Performance:")
        print(f"MAE: {metrics['mae_minutes']:.2f} minutes")
        print(f"MSE: {metrics['mse']:.2f}")
        print(f"R² Score: {metrics['r2_score']:.3f}")
        print(f"CV R² Score: {cv_results['r2_mean']:.3f} ± {cv_results['r2_std']:.3f}")
        
        return metrics
    
    def perform_cross_validation(self, X: pd.DataFrame, y: pd.Series, cv_folds: int = 5) -> dict:
        """Perform cross-validation."""
        # R² Score cross-validation
        cv_r2_scores = cross_val_score(self.model, X, y, cv=cv_folds, scoring='r2')
        
        # Mean Absolute Error cross-validation
        cv_mae_scores = -cross_val_score(self.model, X, y, cv=cv_folds, scoring='neg_mean_absolute_error')
        
        # Mean Squared Error cross-validation
        cv_mse_scores = -cross_val_score(self.model, X, y, cv=cv_folds, scoring='neg_mean_squared_error')
        
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
    
    def predict_time(self, user_data: dict) -> dict:
        """Predict marathon time based on raw data format."""
        if not self.is_trained:
            return {
                'error': 'Model not trained. Please train the model first.',
                'success': False
            }
        
        # Validate and prepare input
        try:
            # Convert distance from km to meters
            distance_km = float(user_data.get('distance_km', 42.2))
            distance_m = distance_km * 1000
            
            # Get elevation gain
            elevation_m = float(user_data.get('elevation_gain_m', 100))
            
            # Convert gender to binary
            gender = user_data.get('gender', 'male').lower()
            gender_binary = 1 if gender in ['female', 'f'] else 0
            
            # Prepare features
            features = np.array([[distance_m, elevation_m, gender_binary]])
            
            # Make prediction
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
                'input_data': {
                    'distance_m': distance_m,
                    'elevation_gain_m': elevation_m,
                    'gender_binary': gender_binary
                }
            }
            
        except Exception as e:
            return {
                'error': f"Prediction failed: {str(e)}",
                'success': False
            }
    
    def get_feature_importance(self) -> dict:
        """Get feature importance from the trained model."""
        if not self.is_trained or self.model is None:
            return {}
        
        importance = self.model.feature_importances_
        return dict(zip(self.feature_names, importance))

def test_raw_data_predictions():
    """Test predictions with raw data model."""
    print("\n=== TESTING PREDICTIONS WITH RAW DATA MODEL ===")
    
    # Initialize model
    model = RawDataMarathonPrediction()
    
    # Load and analyze data
    data = model.load_raw_data()
    model.analyze_raw_data(data)
    
    # Train model
    metrics = model.train_model(data)
    
    # Test predictions
    test_cases = [
        {
            'name': 'Male Marathon Standard',
            'data': {
                'distance_km': 42.2,
                'elevation_gain_m': 100,
                'gender': 'male'
            }
        },
        {
            'name': 'Female Marathon Standard',
            'data': {
                'distance_km': 42.2,
                'elevation_gain_m': 100,
                'gender': 'female'
            }
        },
        {
            'name': 'Male Marathon Hilly',
            'data': {
                'distance_km': 42.2,
                'elevation_gain_m': 500,
                'gender': 'male'
            }
        },
        {
            'name': 'Female Marathon Hilly',
            'data': {
                'distance_km': 42.2,
                'elevation_gain_m': 500,
                'gender': 'female'
            }
        },
        {
            'name': 'Male Half Marathon',
            'data': {
                'distance_km': 21.1,
                'elevation_gain_m': 50,
                'gender': 'male'
            }
        },
        {
            'name': 'Female Half Marathon',
            'data': {
                'distance_km': 21.1,
                'elevation_gain_m': 50,
                'gender': 'female'
            }
        }
    ]
    
    print("\n--- Prediction Tests ---")
    for test_case in test_cases:
        result = model.predict_time(test_case['data'])
        if result['success']:
            pred = result['prediction']
            print(f"\n{test_case['name']}:")
            print(f"  Predicted time: {pred['time_string']}")
            print(f"  Pace: {pred['pace_minutes_per_km']} min/km")
        else:
            print(f"\n{test_case['name']}: ❌ {result['error']}")
    
    # Show feature importance
    print("\n--- Feature Importance ---")
    importance = model.get_feature_importance()
    for feature, score in sorted(importance.items(), key=lambda x: x[1], reverse=True):
        print(f"{feature}: {score:.4f}")
    
    return model, metrics

def compare_with_clean_data_model():
    """Compare raw data model with clean data model."""
    print("\n=== COMPARING RAW DATA vs CLEAN DATA MODELS ===")
    
    # Test raw data model
    print("\n1. Raw Data Model (limited features):")
    raw_model, raw_metrics = test_raw_data_predictions()
    
    # Test clean data model (from existing script)
    print("\n2. Clean Data Model (full features):")
    try:
        from marathon_prediction import MarathonPrediction
        clean_model = MarathonPrediction()
        clean_metrics = clean_model.train_model(save_model=False)
        
        print(f"Clean model MAE: {clean_metrics['mae_minutes']:.2f} minutes")
        print(f"Clean model R²: {clean_metrics['r2_score']:.3f}")
        print(f"Clean model CV R²: {clean_metrics['cross_validation']['r2_mean']:.3f} ± {clean_metrics['cross_validation']['r2_std']:.3f}")
        
        print(f"\nRaw model MAE: {raw_metrics['mae_minutes']:.2f} minutes")
        print(f"Raw model R²: {raw_metrics['r2_score']:.3f}")
        print(f"Raw model CV R²: {raw_metrics['cross_validation']['r2_mean']:.3f} ± {raw_metrics['cross_validation']['r2_std']:.3f}")
        
        print(f"\nPerformance difference:")
        print(f"MAE difference: {abs(raw_metrics['mae_minutes'] - clean_metrics['mae_minutes']):.2f} minutes")
        print(f"R² difference: {abs(raw_metrics['r2_score'] - clean_metrics['r2_score']):.3f}")
        
    except Exception as e:
        print(f"Could not compare with clean data model: {e}")

def main():
    """Run the raw data analysis."""
    print("🚀 RAW DATA MARATHON PREDICTION ANALYSIS")
    print("=" * 60)
    
    # Test raw data predictions
    test_raw_data_predictions()
    
    # Compare with clean data model
    compare_with_clean_data_model()
    
    print("\n" + "=" * 60)
    print("✅ Raw data analysis complete!")
    print("\nKey findings:")
    print("- Raw data model uses only distance, elevation, and gender")
    print("- Performance may be lower due to fewer features")
    print("- No training volume or experience level information available")

if __name__ == "__main__":
    main() 