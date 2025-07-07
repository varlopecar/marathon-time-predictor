#!/usr/bin/env python3
"""
Test script to verify the API returns comprehensive metrics.
"""

import requests
import json
import time

def test_api_metrics():
    """Test the API with comprehensive metrics."""
    
    # API endpoint
    url = "http://localhost:8000/predict"
    
    # Test data
    test_data = {
        "distance_km": 42.2,
        "elevation_gain_m": 200,
        "mean_km_per_week": 60,
        "mean_training_days_per_week": 5,
        "gender": "male",
        "level": 2
    }
    
    try:
        print("🚀 Testing API with comprehensive metrics...")
        print(f"Request data: {json.dumps(test_data, indent=2)}")
        print("-" * 50)
        
        # Make request
        response = requests.post(url, json=test_data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            
            print("✅ API Response received successfully!")
            print(f"Success: {result.get('success')}")
            
            # Show prediction
            if result.get('prediction'):
                pred = result['prediction']
                print(f"\n🏃‍♂️ Prediction:")
                print(f"  Time: {pred.get('time_string', 'N/A')}")
                print(f"  Pace: {pred.get('pace_minutes_per_km', 'N/A')} min/km")
            
            # Show feature importance
            if result.get('feature_importance'):
                print(f"\n📊 Feature Importance:")
                importance = result['feature_importance']
                for feature, score in sorted(importance.items(), key=lambda x: x[1], reverse=True):
                    print(f"  {feature}: {score:.4f}")
            
            # Show model metrics
            if result.get('model_metrics'):
                print(f"\n📈 Model Performance:")
                metrics = result['model_metrics']
                for key, value in metrics.items():
                    print(f"  {key}: {value}")
            
            # Show data insights
            if result.get('data_insights'):
                print(f"\n🔍 Data Insights:")
                insights = result['data_insights']
                
                if 'dataset_size' in insights:
                    print(f"  Dataset size: {insights['dataset_size']}")
                
                if 'gender_distribution' in insights:
                    gender = insights['gender_distribution']
                    print(f"  Gender distribution: {gender['male_percentage']:.1f}% male, {gender['female_percentage']:.1f}% female")
                
                if 'feature_correlations' in insights:
                    print(f"  Feature correlations with finish time:")
                    corr = insights['feature_correlations']
                    for feature, value in corr.items():
                        print(f"    {feature}: {value:.3f}")
                
                if 'data_quality_warnings' in insights and insights['data_quality_warnings']:
                    print(f"  ⚠️  Data Quality Warnings:")
                    for warning in insights['data_quality_warnings']:
                        print(f"    {warning}")
            
            # Show cross-validation results
            if result.get('cross_validation'):
                print(f"\n🔄 Cross-Validation Results:")
                cv = result['cross_validation']
                print(f"  R² Score: {cv.get('r2_mean', 'N/A'):.3f} ± {cv.get('r2_std', 'N/A'):.3f}")
                print(f"  MAE: {cv.get('mae_mean', 'N/A')/60:.2f} ± {cv.get('mae_std', 'N/A')/60:.2f} minutes")
            
        else:
            print(f"❌ API Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API. Make sure the server is running on localhost:8000")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_api_metrics() 