"""
Marathon Time Prediction Model - Random Forest Only
Predicts race time from gender, elevation, and distance using Random Forest.

This implementation:
1. Uses clean_dataset.csv as the dataset
2. Trains Random Forest on 80% of the data
3. Predicts elapsed time for the remaining 20%
4. Compares actual vs predicted elapsed times
5. Outputs predictions alongside original features to CSV for evaluation
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class MarathonModel:
    def __init__(self):
        """
        Initialize the marathon prediction model using only Random Forest.
        """
        self.random_forest = RandomForestRegressor(
            n_estimators=100, random_state=42)
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None

    def load_and_explore_data(self, data_path="clean_dataset.csv"):
        """
        Load and explore the clean dataset
        """
        print("📊 Loading and exploring clean dataset...")

        # Load the data with semicolon separator
        self.data = pd.read_csv(data_path, sep=';')

        # Convert European number format (comma as decimal separator) to standard format
        print("Converting European number format to standard format...")

        # Columns that need conversion (numeric columns)
        numeric_columns = ['distance_m', 'elevation_gain_m', 'mean_km_per_week',
                           'mean_training_days_per_week', 'elapsed_time_s']

        for col in numeric_columns:
            if col in self.data.columns:
                # Replace comma with period and convert to float
                self.data[col] = self.data[col].astype(
                    str).str.replace(',', '.').astype(float)
                print(f"Converted {col} to numeric format")

        # Show basic info about our data
        print(f"Dataset shape: {self.data.shape}")
        print(f"Columns: {list(self.data.columns)}")
        print("\nFirst few rows:")
        print(self.data.head())

        # Check for missing values
        print(f"\nMissing values:\n{self.data.isnull().sum()}")

        # Show basic statistics
        print(f"\nData statistics:\n{self.data.describe()}")

        return self.data

    def prepare_data(self):
        """
        Prepare the data for training - 80/20 split
        """
        print("\n🔧 Preparing data for training...")

        # Select features (everything except the target)
        features = ['distance_m', 'elevation_gain_m', 'mean_km_per_week',
                    'mean_training_days_per_week', 'gender_binary', 'level']

        X = self.data[features]  # Our input features
        # What we want to predict (time in seconds)
        y = self.data['elapsed_time_s']

        # Store feature names for later use
        self.feature_names = features

        # Split data into training (80%) and testing (20%) sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        print(f"Training set size: {len(self.X_train)}")
        print(f"Testing set size: {len(self.X_test)}")
        print(f"Features used: {features}")

        return X, y

    def train_model(self):
        """
        Train the Random Forest model
        """
        print("\n🚀 Training Random Forest model...")

        # Train Random Forest
        self.random_forest.fit(self.X_train, self.y_train)

        print("✅ Random Forest model trained successfully!")

    def evaluate_model(self):
        """
        Evaluate how well the model performs
        """
        print("\n📈 Evaluating model performance...")

        # Make predictions
        rf_pred = self.random_forest.predict(self.X_test)

        # Calculate metrics
        rf_mae = mean_absolute_error(self.y_test, rf_pred)
        rf_mse = mean_squared_error(self.y_test, rf_pred)
        rf_r2 = r2_score(self.y_test, rf_pred)

        print("📊 Random Forest Performance:")
        print("=" * 50)
        print(
            f"Mean Absolute Error: {rf_mae:.2f} seconds ({rf_mae/60:.1f} minutes)")
        print(f"Mean Squared Error: {rf_mse:.2f}")
        print(f"R² Score: {rf_r2:.3f}")

        return {'mae': rf_mae, 'mse': rf_mse, 'r2': rf_r2}

    def analyze_feature_importance(self):
        """
        Analyze which features are most important
        """
        print("\n🔍 Analyzing feature importance...")

        # Random Forest feature importance
        rf_importance = self.random_forest.feature_importances_

        print("Random Forest Feature Importance:")
        for feature, importance in zip(self.feature_names, rf_importance):
            print(f"  {feature}: {importance:.3f}")

        return rf_importance

    def generate_predictions_csv(self):
        """
        Generate CSV with actual vs predicted times and all original features
        """
        print("\n📋 Generating predictions CSV...")

        # Make predictions on test set
        rf_pred = self.random_forest.predict(self.X_test)

        # Create results dataframe with all original features
        results = self.X_test.copy()
        results['actual_elapsed_time_s'] = self.y_test
        results['predicted_elapsed_time_s'] = rf_pred
        results['actual_elapsed_time_min'] = self.y_test / 60
        results['predicted_elapsed_time_min'] = rf_pred / 60
        results['absolute_error_s'] = np.abs(self.y_test - rf_pred)
        results['absolute_error_min'] = np.abs(self.y_test - rf_pred) / 60
        results['percentage_error'] = (
            np.abs(self.y_test - rf_pred) / self.y_test) * 100

        # Save to CSV
        output_file = 'marathon_predictions_comparison.csv'
        results.to_csv(output_file, index=False)

        print(f"✅ Predictions saved to '{output_file}'")
        print(f"Total predictions: {len(results)}")

        # Show summary statistics
        print("\n📈 Prediction Summary:")
        print("=" * 40)
        print(
            f"Average predicted time: {results['predicted_elapsed_time_min'].mean():.1f} minutes")
        print(
            f"Average actual time: {results['actual_elapsed_time_min'].mean():.1f} minutes")
        print(
            f"Average absolute error: {results['absolute_error_min'].mean():.1f} minutes")
        print(
            f"Average percentage error: {results['percentage_error'].mean():.1f}%")
        print(
            f"Min predicted time: {results['predicted_elapsed_time_min'].min():.1f} minutes")
        print(
            f"Max predicted time: {results['predicted_elapsed_time_min'].max():.1f} minutes")

        # Show first few predictions
        print(f"\n📋 First 10 predictions:")
        print("=" * 80)
        display_cols = ['distance_m', 'elevation_gain_m', 'mean_km_per_week',
                        'mean_training_days_per_week', 'gender_binary', 'level',
                        'actual_elapsed_time_min', 'predicted_elapsed_time_min',
                        'absolute_error_min', 'percentage_error']

        print(results[display_cols].head(10).to_string(index=False))

        return results

    def predict(self, data_path="clean_dataset.csv"):
        """
        Main method to run the complete prediction pipeline
        """
        print("🏃‍♂️ Marathon Time Prediction Model - Random Forest")
        print("=" * 60)

        # Step 1: Load and explore data
        self.load_and_explore_data(data_path)

        # Step 2: Prepare data (80/20 split)
        X, y = self.prepare_data()

        # Step 3: Train model
        self.train_model()

        # Step 4: Evaluate model
        metrics = self.evaluate_model()

        # Step 5: Feature importance
        self.analyze_feature_importance()

        # Step 6: Generate predictions CSV
        results = self.generate_predictions_csv()

        print("\n🎉 Model training and evaluation complete!")
        print("\n💡 Key Insights:")
        print("- Random Forest model trained on 80% of clean dataset")
        print("- Predictions made for remaining 20% of data")
        print("- Results saved to 'marathon_predictions_comparison.csv'")

        return self.random_forest, results


# Example usage
if __name__ == "__main__":
    # Create the model
    model = MarathonModel()

    # Run the complete pipeline
    rf_model, results = model.predict()

    print("\n" + "="*60)
    print("🎉 COMPLETE! Check 'marathon_predictions_comparison.csv' for detailed results")
    print("="*60)
