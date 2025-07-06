#!/usr/bin/env python3
"""
Script to create a smaller, deployment-friendly model for Scalingo.
This model will be trained with fewer estimators and saved as a smaller file.
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from marathon_prediction import MarathonPrediction


def create_deployment_model():
    """Create a smaller model suitable for deployment."""

    print("Creating deployment-friendly model...")

    # Initialize with smaller model parameters
    model = MarathonPrediction(model_path="deployment_model.pkl")

    # Load and prepare data
    data = model.load_and_prepare_data()

    # Prepare features and target
    X = data[model.feature_names]
    y = data['elapsed_time_s']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model with fewer estimators for smaller file size
    deployment_model = RandomForestRegressor(
        n_estimators=50,  # Reduced from 100
        max_depth=10,     # Limit depth to reduce size
        random_state=42,
        n_jobs=1         # Single thread for deployment
    )

    print("Training deployment model...")
    deployment_model.fit(X_train, y_train)

    # Evaluate model
    y_pred = deployment_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Model performance:")
    print(f"  MAE: {mae:.2f} seconds ({mae/60:.2f} minutes)")
    print(f"  MSE: {mse:.2f}")
    print(f"  R²: {r2:.4f}")

    # Save the deployment model
    model.model = deployment_model
    model.save_model()

    # Check file size
    file_size = os.path.getsize("deployment_model.pkl")
    print(f"Model file size: {file_size / 1024 / 1024:.1f} MB")

    if file_size < 50 * 1024 * 1024:  # Less than 50MB
        print("✅ Deployment model created successfully!")
        print("This model can be included in the git repository.")
    else:
        print("⚠️  Model is still quite large. Consider further optimization.")

    return model


if __name__ == "__main__":
    create_deployment_model()
