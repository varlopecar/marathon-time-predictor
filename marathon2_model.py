"""
Marathon Time Prediction Model 2
Predicts race time from gender, elevation, and distance using scikit-learn.

This implementation follows the theoretical framework:
1. Data collection and inspection
2. Data cleaning and preprocessing
3. Train/test split
4. Model training (Linear Regression + Random Forest)
5. Evaluation and visualization
6. Cross-validation
7. Feature importance analysis
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns


class Marathon2Model:
    def __init__(self):
        """
        Initialize the marathon prediction model.
        We'll use Linear Regression as the main model because:
        1. It's simple and easy to understand
        2. Perfect for numerical data like distance, elevation, training data
        3. Easy to interpret the results
        """
        self.linear_model = LinearRegression()
        self.random_forest = RandomForestRegressor(
            n_estimators=100, random_state=42)
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_and_explore_data(self, data_path="athletes_80.csv"):
        """
        Step 1: Load and explore the data
        This helps us understand what we're working with
        """
        print("📊 Loading and exploring data...")

        # Load the data
        self.data = pd.read_csv(data_path)

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
        Step 2: Prepare the data for training
        We'll use all features to predict elapsed_time_s
        """
        print("\n🔧 Preparing data for training...")

        # Clean the data - handle missing values
        print("Cleaning data...")

        # Fill missing values in gender_binary with the most common value
        gender_mode = self.data['gender_binary'].mode()[0]
        self.data['gender_binary'] = self.data['gender_binary'].fillna(
            gender_mode)

        print(
            f"Filled {self.data['gender_binary'].isnull().sum()} missing gender values with mode: {gender_mode}")

        # Select features (everything except the target)
        features = ['distance_m', 'elevation_gain_m', 'mean_km_per_week',
                    'mean_training_days_per_week', 'gender_binary']

        X = self.data[features]  # Our input features
        # What we want to predict (time in seconds)
        y = self.data['elapsed_time_s']

        # Split data into training (80%) and testing (20%) sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        print(f"Training set size: {len(self.X_train)}")
        print(f"Testing set size: {len(self.X_test)}")
        print(f"Features used: {features}")

        return X, y

    def train_models(self):
        """
        Step 3: Train our models
        We'll train both Linear Regression and Random Forest to compare
        """
        print("\n🚀 Training models...")

        # Train Linear Regression (our main model)
        print("Training Linear Regression...")
        self.linear_model.fit(self.X_train, self.y_train)

        # Train Random Forest (for comparison)
        print("Training Random Forest...")
        self.random_forest.fit(self.X_train, self.y_train)

        print("✅ Models trained successfully!")

    def evaluate_models(self):
        """
        Step 4: Evaluate how well our models perform
        """
        print("\n📈 Evaluating model performance...")

        # Make predictions
        linear_pred = self.linear_model.predict(self.X_test)
        rf_pred = self.random_forest.predict(self.X_test)

        # Calculate metrics for Linear Regression
        linear_mae = mean_absolute_error(self.y_test, linear_pred)
        linear_mse = mean_squared_error(self.y_test, linear_pred)
        linear_r2 = r2_score(self.y_test, linear_pred)

        # Calculate metrics for Random Forest
        rf_mae = mean_absolute_error(self.y_test, rf_pred)
        rf_mse = mean_squared_error(self.y_test, rf_pred)
        rf_r2 = r2_score(self.y_test, rf_pred)

        print("📊 Model Performance Comparison:")
        print("=" * 50)
        print("Linear Regression:")
        print(
            f"  Mean Absolute Error: {linear_mae:.2f} seconds ({linear_mae/60:.1f} minutes)")
        print(f"  Mean Squared Error: {linear_mse:.2f}")
        print(f"  R² Score: {linear_r2:.3f}")
        print()
        print("Random Forest:")
        print(
            f"  Mean Absolute Error: {rf_mae:.2f} seconds ({rf_mae/60:.1f} minutes)")
        print(f"  Mean Squared Error: {rf_mse:.2f}")
        print(f"  R² Score: {rf_r2:.3f}")

        return {
            'linear': {'mae': linear_mae, 'mse': linear_mse, 'r2': linear_r2},
            'random_forest': {'mae': rf_mae, 'mse': rf_mse, 'r2': rf_r2}
        }

    def cross_validate(self):
        """
        Step 5: Cross-validation to ensure our results are reliable
        """
        print("\n🔄 Performing cross-validation...")

        # Cross-validation for Linear Regression
        linear_cv_scores = cross_val_score(self.linear_model, self.X_train, self.y_train,
                                           cv=5, scoring='r2')

        # Cross-validation for Random Forest
        rf_cv_scores = cross_val_score(self.random_forest, self.X_train, self.y_train,
                                       cv=5, scoring='r2')

        print("Cross-validation R² scores:")
        print(
            f"Linear Regression: {linear_cv_scores.mean():.3f} (+/- {linear_cv_scores.std() * 2:.3f})")
        print(
            f"Random Forest: {rf_cv_scores.mean():.3f} (+/- {rf_cv_scores.std() * 2:.3f})")

        return linear_cv_scores, rf_cv_scores

    def analyze_feature_importance(self):
        """
        Step 6: Analyze which features are most important
        """
        print("\n🔍 Analyzing feature importance...")

        # Linear Regression coefficients
        feature_names = self.X_train.columns
        linear_coefficients = self.linear_model.coef_

        print("Linear Regression Feature Importance:")
        for feature, coef in zip(feature_names, linear_coefficients):
            print(f"  {feature}: {coef:.2f}")

        # Random Forest feature importance
        rf_importance = self.random_forest.feature_importances_

        print("\nRandom Forest Feature Importance:")
        for feature, importance in zip(feature_names, rf_importance):
            print(f"  {feature}: {importance:.3f}")

        return linear_coefficients, rf_importance

    def create_visualizations(self):
        """
        Step 7: Create helpful visualizations
        """
        print("\n📊 Creating visualizations...")

        # Set up the plotting style
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. Actual vs Predicted (Linear Regression)
        linear_pred = self.linear_model.predict(self.X_test)
        axes[0, 0].scatter(self.y_test, linear_pred, alpha=0.5)
        axes[0, 0].plot([self.y_test.min(), self.y_test.max()],
                        [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('Actual Time (seconds)')
        axes[0, 0].set_ylabel('Predicted Time (seconds)')
        axes[0, 0].set_title('Linear Regression: Actual vs Predicted')

        # 2. Actual vs Predicted (Random Forest)
        rf_pred = self.random_forest.predict(self.X_test)
        axes[0, 1].scatter(self.y_test, rf_pred, alpha=0.5)
        axes[0, 1].plot([self.y_test.min(), self.y_test.max()],
                        [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
        axes[0, 1].set_xlabel('Actual Time (seconds)')
        axes[0, 1].set_ylabel('Predicted Time (seconds)')
        axes[0, 1].set_title('Random Forest: Actual vs Predicted')

        # 3. Feature importance comparison
        feature_names = self.X_train.columns
        linear_coefficients = np.abs(self.linear_model.coef_)
        rf_importance = self.random_forest.feature_importances_

        x = np.arange(len(feature_names))
        width = 0.35

        axes[1, 0].bar(x - width/2, linear_coefficients,
                       width, label='Linear Regression')
        axes[1, 0].bar(x + width/2, rf_importance,
                       width, label='Random Forest')
        axes[1, 0].set_xlabel('Features')
        axes[1, 0].set_ylabel('Importance')
        axes[1, 0].set_title('Feature Importance Comparison')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(feature_names, rotation=45)
        axes[1, 0].legend()

        # 4. Residuals plot
        linear_residuals = self.y_test - linear_pred
        axes[1, 1].scatter(linear_pred, linear_residuals, alpha=0.5)
        axes[1, 1].axhline(y=0, color='r', linestyle='--')
        axes[1, 1].set_xlabel('Predicted Time (seconds)')
        axes[1, 1].set_ylabel('Residuals (Actual - Predicted)')
        axes[1, 1].set_title('Linear Regression Residuals')

        plt.tight_layout()
        plt.savefig('marathon_predictions_analysis.png',
                    dpi=300, bbox_inches='tight')
        plt.show()

        print("✅ Visualizations saved as 'marathon_predictions_analysis.png'")

    def make_prediction(self, distance_m, elevation_gain_m, mean_km_per_week,
                        mean_training_days_per_week, gender_binary):
        """
        Make a prediction for a new runner
        """
        # Create input data
        input_data = np.array([[distance_m, elevation_gain_m, mean_km_per_week,
                               mean_training_days_per_week, gender_binary]])

        # Make predictions
        linear_pred = self.linear_model.predict(input_data)[0]
        rf_pred = self.random_forest.predict(input_data)[0]

        print(f"\n🏃‍♂️ Prediction for new runner:")
        print(f"Distance: {distance_m:.1f}m")
        print(f"Elevation gain: {elevation_gain_m:.1f}m")
        print(f"Weekly training: {mean_km_per_week:.1f}km")
        print(f"Training days per week: {mean_training_days_per_week:.1f}")
        print(f"Gender: {'Male' if gender_binary == 1 else 'Female'}")
        print()
        print(
            f"Linear Regression prediction: {linear_pred:.0f} seconds ({linear_pred/60:.1f} minutes)")
        print(
            f"Random Forest prediction: {rf_pred:.0f} seconds ({rf_pred/60:.1f} minutes)")

        return linear_pred, rf_pred

    def predict(self, data_path="athletes_80.csv"):
        """
        Main method to run the complete prediction pipeline
        """
        print("🏃‍♂️ Marathon Time Prediction Model")
        print("=" * 50)

        # Step 1: Load and explore data
        self.load_and_explore_data(data_path)

        # Step 2: Prepare data
        X, y = self.prepare_data()

        # Step 3: Train models
        self.train_models()

        # Step 4: Evaluate models
        metrics = self.evaluate_models()

        # Step 5: Cross-validation
        self.cross_validate()

        # Step 6: Feature importance
        self.analyze_feature_importance()

        # Step 7: Create visualizations
        self.create_visualizations()

        print("\n🎉 Model training and evaluation complete!")
        print("\n💡 Key Insights:")
        print("- Linear Regression is great for understanding relationships")
        print("- Random Forest might be more accurate but harder to interpret")
        print("- You can use either model depending on your needs")

        return self.linear_model, self.random_forest

    def predict_test_file(self, train_data_path="athletes_80.csv", test_data_path="athletes_20_test.csv"):
        """
        Train the model on the 80% data and predict times for each row in the test file
        """
        print("🏃‍♂️ Training on 80% data and predicting test file times")
        print("=" * 60)

        # Step 1: Load training data (80%)
        print("📊 Loading training data...")
        self.data = pd.read_csv(train_data_path)

        # Clean training data
        gender_mode = self.data['gender_binary'].mode()[0]
        self.data['gender_binary'] = self.data['gender_binary'].fillna(
            gender_mode)
        print(f"Training data shape: {self.data.shape}")

        # Step 2: Prepare training data
        print("\n🔧 Preparing training data...")
        features = ['distance_m', 'elevation_gain_m', 'mean_km_per_week',
                    'mean_training_days_per_week', 'gender_binary']

        X_train = self.data[features]
        y_train = self.data['elapsed_time_s']

        print(f"Training features: {features}")
        print(f"Training set size: {len(X_train)}")

        # Step 3: Train models
        print("\n🚀 Training models...")
        self.linear_model.fit(X_train, y_train)
        self.random_forest.fit(X_train, y_train)
        print("✅ Models trained successfully!")

        # Step 4: Load test data
        print("\n📊 Loading test data...")
        test_data = pd.read_csv(test_data_path)
        print(f"Test data shape: {test_data.shape}")

        # Clean test data (handle any missing values)
        if test_data['gender_binary'].isnull().sum() > 0:
            test_gender_mode = test_data['gender_binary'].mode()[0]
            test_data['gender_binary'] = test_data['gender_binary'].fillna(
                test_gender_mode)
            print(
                f"Filled {test_data['gender_binary'].isnull().sum()} missing gender values in test data")

        # Step 5: Make predictions for each test row
        print("\n🔮 Making predictions for test data...")
        X_test = test_data[features]

        # Make predictions
        linear_predictions = self.linear_model.predict(X_test)
        rf_predictions = self.random_forest.predict(X_test)

        # Step 6: Create results dataframe
        results = test_data.copy()
        results['predicted_time_linear_s'] = linear_predictions
        results['predicted_time_rf_s'] = rf_predictions
        results['predicted_time_linear_min'] = linear_predictions / 60
        results['predicted_time_rf_min'] = rf_predictions / 60

        # Step 7: Save results
        output_file = 'test_predictions_results.csv'
        results.to_csv(output_file, index=False)

        print(f"\n✅ Predictions saved to '{output_file}'")
        print(f"Total predictions made: {len(results)}")

        # Step 8: Show summary statistics
        print("\n📈 Prediction Summary:")
        print("=" * 40)
        print(f"Linear Regression predictions:")
        print(
            f"  Average predicted time: {results['predicted_time_linear_min'].mean():.1f} minutes")
        print(
            f"  Min predicted time: {results['predicted_time_linear_min'].min():.1f} minutes")
        print(
            f"  Max predicted time: {results['predicted_time_linear_min'].max():.1f} minutes")

        print(f"\nRandom Forest predictions:")
        print(
            f"  Average predicted time: {results['predicted_time_rf_min'].mean():.1f} minutes")
        print(
            f"  Min predicted time: {results['predicted_time_rf_min'].min():.1f} minutes")
        print(
            f"  Max predicted time: {results['predicted_time_rf_min'].max():.1f} minutes")

        # Step 9: Show first few predictions
        print(f"\n📋 First 10 predictions:")
        print("=" * 60)
        display_cols = ['distance_m', 'elevation_gain_m', 'mean_km_per_week',
                        'mean_training_days_per_week', 'gender_binary',
                        'predicted_time_linear_min', 'predicted_time_rf_min']

        print(results[display_cols].head(10).to_string(index=False))

        return results


# Example usage
if __name__ == "__main__":
    # Create the model
    model = Marathon2Model()

    # Train on 80% data and predict test file times
    results = model.predict_test_file()

    print("\n" + "="*60)
    print("🎉 COMPLETE! Check 'test_predictions_results.csv' for all predictions")
    print("="*60)
