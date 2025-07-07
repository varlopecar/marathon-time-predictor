#!/usr/bin/env python3
"""
Create an optimized deployment model with the same performance as the original model.
This script trains a model with the same parameters that gave us:
- MAE: 8.78 minutes
- R² Score: 0.682
- Cross-validation R²: 0.002 ± 0.457
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict, Any
import joblib

def load_and_prepare_data(data_path: str = "clean_dataset.csv") -> pd.DataFrame:
    """Load and prepare the training data."""
    # Load the data
    data = pd.read_csv(data_path, sep=';')

    # Convert European number format to standard format
    numeric_columns = ['distance_m', 'elevation_gain_m', 'mean_km_per_week',
                       'mean_training_days_per_week', 'elapsed_time_s']

    for col in numeric_columns:
        if col in data.columns:
            data[col] = data[col].astype(str).str.replace(',', '.').astype(float)

    return data

def create_optimized_model():
    """Create an optimized deployment model with the same performance."""
    
    print("🚀 Creating optimized deployment model...")
    
    # Load and prepare data
    print("📊 Loading and preparing data...")
    data = load_and_prepare_data()
    
    # Feature names
    feature_names = [
        'distance_m', 'elevation_gain_m', 'mean_km_per_week',
        'mean_training_days_per_week', 'gender_binary', 'level'
    ]
    
    # Prepare features and target
    X = data[feature_names]
    y = data['elapsed_time_s']
    
    print(f"Dataset shape: {data.shape}")
    print(f"Features: {feature_names}")
    
    # Split data (same as original)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Train model with EXACT same parameters as original (no regularization)
    print("🏋️ Training Random Forest model...")
    model = RandomForestRegressor(
        n_estimators=100,  # Same as original
        random_state=42    # Same as original
        # No additional parameters to match original performance
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate model with comprehensive metrics
    print("📈 Evaluating model performance...")
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Test MAE: {mae:.2f} seconds ({mae/60:.2f} minutes)")
    print(f"Test MSE: {mse:.2f}")
    print(f"Test RMSE: {rmse:.2f} seconds ({rmse/60:.2f} minutes)")
    print(f"Test R² Score: {r2:.3f}")
    print(f"Coefficient of Determination: {r2:.3f}")
    
    # Perform cross-validation
    print("🔄 Performing cross-validation...")
    cv_r2_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    cv_mae_scores = -cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error')
    
    print(f"CV R² Score: {cv_r2_scores.mean():.3f} ± {cv_r2_scores.std():.3f}")
    print(f"CV MAE: {cv_mae_scores.mean()/60:.2f} ± {cv_mae_scores.std()/60:.2f} minutes")
    
    # Get feature importance
    importance = model.feature_importances_
    feature_importance = dict(zip(feature_names, importance))
    
    print("\n📊 Feature Importance:")
    for feature, score in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True):
        print(f"  {feature}: {score:.4f}")
    
    # Check model size
    model_size = len(pickle.dumps(model))
    print(f"\n📦 Model size: {model_size / 1024 / 1024:.2f} MB")
    
    # Save optimized model
    print("💾 Saving optimized deployment model...")
    
    # Use joblib for better compression
    deployment_model_path = "deployment_model.pkl"
    joblib.dump(model, deployment_model_path, compress=3)
    
    # Check compressed size
    compressed_size = os.path.getsize(deployment_model_path) / 1024 / 1024
    print(f"Compressed model size: {compressed_size:.2f} MB")
    print(f"Compression ratio: {model_size / (compressed_size * 1024 * 1024):.1f}x")
    
    # Verify the saved model works
    print("✅ Verifying saved model...")
    loaded_model = joblib.load(deployment_model_path)
    test_prediction = loaded_model.predict(X_test[:1])
    print(f"Test prediction: {test_prediction[0]:.2f} seconds")
    
    # Create model metadata with comprehensive metrics
    metadata = {
        'model_type': 'Random Forest',
        'features': feature_names,
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'performance': {
            'mae_seconds': mae,
            'mae_minutes': mae / 60,
            'mse': mse,
            'rmse_seconds': rmse,
            'rmse_minutes': rmse / 60,
            'r2_score': r2,
            'coefficient_of_determination': r2,
            'cv_r2_mean': cv_r2_scores.mean(),
            'cv_r2_std': cv_r2_scores.std(),
            'cv_mae_mean': cv_mae_scores.mean(),
            'cv_mae_std': cv_mae_scores.std()
        },
        'feature_importance': feature_importance,
        'model_size_mb': compressed_size,
        'compression_ratio': model_size / (compressed_size * 1024 * 1024)
    }
    
    # Save metadata
    metadata_path = "deployment_model_metadata.json"
    import json
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"📋 Model metadata saved to {metadata_path}")
    
    print("\n🎉 Optimized deployment model created successfully!")
    print(f"Model file: {deployment_model_path}")
    print(f"Metadata file: {metadata_path}")
    
    return model, metadata

if __name__ == "__main__":
    create_optimized_model() 