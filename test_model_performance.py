#!/usr/bin/env python3
"""
Test script to evaluate marathon prediction model performance
and investigate feature importance issues.
"""

import pandas as pd
import numpy as np
from marathon_prediction import MarathonPrediction
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

def analyze_data_distribution():
    """Analyze the distribution of features in the dataset."""
    print("=== DATA DISTRIBUTION ANALYSIS ===")
    
    # Load the data
    data = pd.read_csv("clean_dataset.csv", sep=';')
    
    # Convert European number format
    numeric_columns = ['distance_m', 'elevation_gain_m', 'mean_km_per_week',
                       'mean_training_days_per_week', 'elapsed_time_s']
    
    for col in numeric_columns:
        if col in data.columns:
            data[col] = data[col].astype(str).str.replace(',', '.').astype(float)
    
    print(f"Dataset shape: {data.shape}")
    print(f"Number of unique distance values: {data['distance_m'].nunique()}")
    print(f"Number of unique gender values: {data['gender_binary'].nunique()}")
    
    # Analyze key features
    print("\n--- Feature Statistics ---")
    print(data[['distance_m', 'elevation_gain_m', 'mean_km_per_week', 
                'mean_training_days_per_week', 'elapsed_time_s']].describe())
    
    # Gender distribution
    print(f"\n--- Gender Distribution ---")
    gender_counts = data['gender_binary'].value_counts()
    print(f"Male (0): {gender_counts.get(0, 0)} ({gender_counts.get(0, 0)/len(data)*100:.1f}%)")
    print(f"Female (1): {gender_counts.get(1, 0)} ({gender_counts.get(1, 0)/len(data)*100:.1f}%)")
    
    # Level distribution
    print(f"\n--- Level Distribution ---")
    level_counts = data['level'].value_counts().sort_index()
    for level, count in level_counts.items():
        print(f"Level {level}: {count} ({count/len(data)*100:.1f}%)")
    
    # Training volume distribution
    print(f"\n--- Training Volume Distribution ---")
    km_per_week_stats = data['mean_km_per_week'].describe()
    print(f"Mean km/week: {km_per_week_stats['mean']:.1f}")
    print(f"Std km/week: {km_per_week_stats['std']:.1f}")
    print(f"Min km/week: {km_per_week_stats['min']:.1f}")
    print(f"Max km/week: {km_per_week_stats['max']:.1f}")
    
    # Check for correlations
    print(f"\n--- Feature Correlations with Target ---")
    correlations = data[['distance_m', 'elevation_gain_m', 'mean_km_per_week', 
                        'mean_training_days_per_week', 'gender_binary', 'level', 'elapsed_time_s']].corr()['elapsed_time_s']
    for feature, corr in correlations.items():
        if feature != 'elapsed_time_s':
            print(f"{feature}: {corr:.3f}")
    
    return data

def test_model_performance():
    """Test the current model performance."""
    print("\n=== MODEL PERFORMANCE TEST ===")
    
    # Initialize model
    model = MarathonPrediction()
    
    # Train model directly (skip loading existing model)
    print("Training new model for testing...")
    metrics = model.train_model(save_model=False)
    print("✅ Model trained successfully")
    
    # Test with various inputs
    test_cases = [
        {
            'name': 'Male Beginner Low Training',
            'data': {
                'distance_km': 42.2,
                'elevation_gain_m': 100,
                'mean_km_per_week': 30,
                'mean_training_days_per_week': 3,
                'gender': 'male',
                'level': 1
            }
        },
        {
            'name': 'Male Advanced High Training',
            'data': {
                'distance_km': 42.2,
                'elevation_gain_m': 100,
                'mean_km_per_week': 100,
                'mean_training_days_per_week': 6,
                'gender': 'male',
                'level': 3
            }
        },
        {
            'name': 'Female Intermediate Medium Training',
            'data': {
                'distance_km': 42.2,
                'elevation_gain_m': 100,
                'mean_km_per_week': 60,
                'mean_training_days_per_week': 5,
                'gender': 'female',
                'level': 2
            }
        },
        {
            'name': 'Female Advanced High Training',
            'data': {
                'distance_km': 42.2,
                'elevation_gain_m': 100,
                'mean_km_per_week': 120,
                'mean_training_days_per_week': 7,
                'gender': 'female',
                'level': 3
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
    
    # Check feature importance
    print("\n--- Feature Importance ---")
    importance = model.get_feature_importance()
    if importance:
        for feature, score in sorted(importance.items(), key=lambda x: x[1], reverse=True):
            print(f"{feature}: {score:.4f}")
    else:
        print("❌ Could not get feature importance")

def analyze_feature_importance_issues():
    """Analyze why mean_km_per_week and gender might not be important."""
    print("\n=== FEATURE IMPORTANCE ANALYSIS ===")
    
    # Load data
    data = pd.read_csv("clean_dataset.csv", sep=';')
    
    # Convert European number format
    numeric_columns = ['distance_m', 'elevation_gain_m', 'mean_km_per_week',
                       'mean_training_days_per_week', 'elapsed_time_s']
    
    for col in numeric_columns:
        if col in data.columns:
            data[col] = data[col].astype(str).str.replace(',', '.').astype(float)
    
    # Check for data quality issues
    print("--- Data Quality Check ---")
    print(f"Missing values in mean_km_per_week: {data['mean_km_per_week'].isnull().sum()}")
    print(f"Missing values in gender_binary: {data['gender_binary'].isnull().sum()}")
    
    # Check for extreme values
    print(f"\n--- Extreme Values Check ---")
    print(f"mean_km_per_week > 200: {(data['mean_km_per_week'] > 200).sum()}")
    print(f"mean_km_per_week == 0: {(data['mean_km_per_week'] == 0).sum()}")
    
    # Analyze by gender
    print(f"\n--- Gender Analysis ---")
    male_data = data[data['gender_binary'] == 0]
    female_data = data[data['gender_binary'] == 1]
    
    print(f"Male athletes: {len(male_data)}")
    print(f"Female athletes: {len(female_data)}")
    
    if len(male_data) > 0 and len(female_data) > 0:
        print(f"Male mean time: {male_data['elapsed_time_s'].mean() / 3600:.2f} hours")
        print(f"Female mean time: {female_data['elapsed_time_s'].mean() / 3600:.2f} hours")
        print(f"Male mean km/week: {male_data['mean_km_per_week'].mean():.1f}")
        print(f"Female mean km/week: {female_data['mean_km_per_week'].mean():.1f}")
    
    # Analyze training volume vs performance
    print(f"\n--- Training Volume vs Performance ---")
    data['time_hours'] = data['elapsed_time_s'] / 3600
    
    # Group by training volume ranges
    data['training_group'] = pd.cut(data['mean_km_per_week'], 
                                   bins=[0, 30, 50, 70, 100, 200], 
                                   labels=['0-30', '30-50', '50-70', '70-100', '100+'])
    
    training_performance = data.groupby('training_group')['time_hours'].agg(['mean', 'std', 'count'])
    print(training_performance)
    
    # Check correlation between training volume and performance
    correlation = data['mean_km_per_week'].corr(data['elapsed_time_s'])
    print(f"\nCorrelation between training volume and finish time: {correlation:.3f}")
    
    # If correlation is positive, that's bad - more training should mean faster times
    if correlation > 0:
        print("⚠️  WARNING: Positive correlation between training volume and finish time!")
        print("   This suggests data quality issues or confounding factors.")

def retrain_model_with_analysis():
    """Retrain the model and analyze the training process."""
    print("\n=== MODEL RETRAINING AND ANALYSIS ===")
    
    model = MarathonPrediction()
    
    # Train model and get metrics
    print("Training model...")
    metrics = model.train_model(save_model=False)
    
    print("\n--- Training Metrics ---")
    print(f"MAE: {metrics['mae_minutes']:.2f} minutes")
    print(f"MSE: {metrics['mse']:.2f}")
    print(f"R² Score: {metrics['r2_score']:.3f}")
    print(f"Training samples: {metrics['training_samples']}")
    print(f"Test samples: {metrics['test_samples']}")
    
    print("\n--- Cross-Validation Results ---")
    cv = metrics['cross_validation']
    print(f"CV R² Score: {cv['r2_mean']:.3f} ± {cv['r2_std']:.3f}")
    print(f"CV MAE: {cv['mae_mean']/60:.2f} ± {cv['mae_std']/60:.2f} minutes")
    
    return model, metrics

def main():
    """Run all analyses."""
    print("🚀 MARATHON PREDICTION MODEL ANALYSIS")
    print("=" * 50)
    
    # Analyze data distribution
    data = analyze_data_distribution()
    
    # Analyze feature importance issues
    analyze_feature_importance_issues()
    
    # Test current model
    test_model_performance()
    
    # Retrain and analyze
    model, metrics = retrain_model_with_analysis()
    
    print("\n" + "=" * 50)
    print("✅ Analysis complete!")
    print("\nKey findings:")
    print("- Check the feature importance scores above")
    print("- Look for data quality issues in mean_km_per_week")
    print("- Verify gender encoding is correct")
    print("- Check if training volume correlation with finish time is negative")

if __name__ == "__main__":
    main() 