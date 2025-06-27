#!/usr/bin/env python3
"""
Example usage of the Marathon Time Prediction Model
This script demonstrates how to use the MarathonPredictor class
"""

import pandas as pd
import numpy as np
from marathon_model import MarathonPredictor


def create_sample_data():
    """
    Create sample marathon data for demonstration
    """
    print("📊 Creating sample marathon data...")

    # Generate realistic sample data
    np.random.seed(42)
    n_samples = 1000

    # Sample data structure
    data = {
        'gender': np.random.choice(['Male', 'Female'], n_samples),
        'elevation_gain': np.random.uniform(0, 2000, n_samples),  # meters
        # km (marathon distance)
        'distance': np.random.uniform(42.0, 42.5, n_samples),
        # minutes (3-5 hours)
        'race_time': np.random.uniform(180, 300, n_samples)
    }

    # Add some realistic relationships
    df = pd.DataFrame(data)

    # Adjust race time based on features
    # Women tend to be slightly slower on average
    df.loc[df['gender'] == 'Female', 'race_time'] += 15

    # Higher elevation = slower times
    df['race_time'] += df['elevation_gain'] * 0.02

    # Add some noise
    df['race_time'] += np.random.normal(0, 10, n_samples)

    # Ensure reasonable bounds
    df['race_time'] = np.clip(df['race_time'], 150, 360)

    # Save sample data
    df.to_csv('sample_marathon_data.csv', index=False)
    print(f"✅ Created sample data with {n_samples} records")
    print(f"📁 Saved as 'sample_marathon_data.csv'")

    return df


def demonstrate_predictions(predictor):
    """
    Demonstrate making predictions for different types of runners
    """
    print("\n🎯 DEMONSTRATION: Making Predictions")
    print("=" * 50)

    # Example runners
    runners = [
        {
            'name': 'Fast Male Runner',
            'gender': 'MALE',
            'elevation_gain': 100,  # Flat course
            'distance': 42.2
        },
        {
            'name': 'Female Runner',
            'gender': 'FEMALE',
            'elevation_gain': 500,  # Hilly course
            'distance': 42.2
        },
        {
            'name': 'Mountain Runner',
            'gender': 'MALE',
            'elevation_gain': 1500,  # Very hilly
            'distance': 42.2
        }
    ]

    for runner in runners:
        print(f"\n🏃‍♂️ {runner['name']}:")
        print(f"   Gender: {runner['gender']}")
        print(f"   Elevation Gain: {runner['elevation_gain']} meters")
        print(f"   Distance: {runner['distance']} km")

        # Make prediction
        try:
            lr_pred, rf_pred = predictor.predict_for_user(
                runner['gender'],
                runner['elevation_gain'],
                runner['distance']
            )

            # Convert to hours and minutes
            def format_time(minutes):
                hours = int(minutes // 60)
                mins = int(minutes % 60)
                return f"{hours}h {mins}m"

            print(f"   📊 Predictions:")
            print(f"      Linear Regression: {format_time(lr_pred)}")
            print(f"      Random Forest: {format_time(rf_pred)}")

        except Exception as e:
            print(f"   ❌ Error making prediction: {e}")


def main():
    """
    Main demonstration function
    """
    print("🏃‍♂️ MARATHON PREDICTION MODEL - EXAMPLE USAGE")
    print("=" * 60)

    # Step 1: Create sample data if no real data exists
    try:
        # Try to load existing data
        df = pd.read_csv('raw-data-kaggle.csv')
        print("✅ Found existing data file: 'raw-data-kaggle.csv'")
    except FileNotFoundError:
        print("📊 No existing data found. Creating sample data...")
        df = create_sample_data()

        # Update the predictor to use sample data
        predictor = MarathonPredictor('sample_marathon_data.csv')
    else:
        # Use existing data
        predictor = MarathonPredictor('raw-data-kaggle.csv')

    # Step 2: Run the complete pipeline
    print("\n🚀 Running complete prediction pipeline...")
    success = predictor.run_complete_pipeline()

    if success:
        print("\n🎉 Pipeline completed successfully!")

        # Step 3: Demonstrate predictions
        demonstrate_predictions(predictor)

        print("\n📋 SUMMARY:")
        print("✅ Model trained and evaluated")
        print("✅ Feature importance analyzed")
        print("✅ Visualizations generated")
        print("✅ Predictions demonstrated")

        print("\n💡 NEXT STEPS:")
        print("1. Check the generated PNG files for insights")
        print("2. Use the trained models for new predictions")
        print("3. Consider adding more features (age, experience, etc.)")
        print("4. Experiment with different algorithms")

    else:
        print("\n❌ Pipeline failed. Please check your data and try again.")


if __name__ == "__main__":
    main()
