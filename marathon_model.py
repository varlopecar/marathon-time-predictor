"""
Marathon Time Prediction Model - Random Forest Only
Predicts race time from gender, elevation, and distance using Random Forest.

This implementation:
1. Uses clean_dataset.csv as the dataset
2. Trains Random Forest on 80% of the data
3. Predicts elapsed time for the remaining 20%
4. Compares actual vs predicted elapsed times
5. Outputs predictions alongside original features to CSV for evaluation
6. Includes cross-validation for robust model evaluation
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns


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
        self.file_counter = 1  # Counter for CSV file naming

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

    def perform_cross_validation(self, cv_folds=5):
        """
        Perform cross-validation to get robust model performance estimates
        """
        print(f"\n🔄 Performing {cv_folds}-fold cross-validation...")

        # Prepare features and target for cross-validation
        features = ['distance_m', 'elevation_gain_m', 'mean_km_per_week',
                    'mean_training_days_per_week', 'gender_binary', 'level']
        X = self.data[features]
        y = self.data['elapsed_time_s']

        # Perform cross-validation with different metrics
        print("Cross-validating with different metrics...")

        # R² Score cross-validation
        cv_r2_scores = cross_val_score(
            self.random_forest, X, y, cv=cv_folds, scoring='r2')

        # Mean Absolute Error cross-validation (negative because sklearn maximizes)
        cv_mae_scores = -cross_val_score(self.random_forest,
                                         X, y, cv=cv_folds, scoring='neg_mean_absolute_error')

        # Mean Squared Error cross-validation (negative because sklearn maximizes)
        cv_mse_scores = -cross_val_score(self.random_forest,
                                         X, y, cv=cv_folds, scoring='neg_mean_squared_error')

        # Calculate statistics
        cv_results = {
            'r2_mean': cv_r2_scores.mean(),
            'r2_std': cv_r2_scores.std(),
            'mae_mean': cv_mae_scores.mean(),
            'mae_std': cv_mae_scores.std(),
            'mse_mean': cv_mse_scores.mean(),
            'mse_std': cv_mse_scores.std(),
            'r2_scores': cv_r2_scores,
            'mae_scores': cv_mae_scores,
            'mse_scores': cv_mse_scores
        }

        # Display results
        print("📊 Cross-Validation Results:")
        print("=" * 50)
        print(
            f"R² Score: {cv_results['r2_mean']:.3f} (+/- {cv_results['r2_std'] * 2:.3f})")
        print(
            f"Mean Absolute Error: {cv_results['mae_mean']:.2f} seconds (+/- {cv_results['mae_std'] * 2:.2f})")
        print(
            f"Mean Squared Error: {cv_results['mse_mean']:.2f} (+/- {cv_results['mse_std'] * 2:.2f})")

        print(f"\n📈 Individual Fold Results:")
        print("-" * 40)
        for i in range(cv_folds):
            print(
                f"Fold {i+1}: R²={cv_r2_scores[i]:.3f}, MAE={cv_mae_scores[i]:.1f}s, MSE={cv_mse_scores[i]:.1f}")

        # Store results for plotting
        self.cv_results = cv_results

        return cv_results

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
        output_file = f'marathon_predictions_comparison_{self.file_counter}.csv'
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

        self.file_counter += 1  # Increment the file counter

        return results

    def create_evaluation_plots(self, results):
        """
        Create comprehensive visualization plots for model evaluation
        """
        print("\n📊 Creating evaluation plots...")

        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")

        # Create a figure with multiple subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Marathon Time Prediction Model - Evaluation Results',
                     fontsize=16, fontweight='bold')

        # 1. Actual vs Predicted Scatter Plot
        ax1 = axes[0, 0]
        ax1.scatter(results['actual_elapsed_time_min'], results['predicted_elapsed_time_min'],
                    alpha=0.6, s=20)
        ax1.plot([results['actual_elapsed_time_min'].min(), results['actual_elapsed_time_min'].max()],
                 [results['actual_elapsed_time_min'].min(
                 ), results['actual_elapsed_time_min'].max()],
                 'r--', lw=2, label='Perfect Prediction')
        ax1.set_xlabel('Actual Time (minutes)')
        ax1.set_ylabel('Predicted Time (minutes)')
        ax1.set_title('Actual vs Predicted Times')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Residuals Plot
        ax2 = axes[0, 1]
        residuals = results['actual_elapsed_time_min'] - \
            results['predicted_elapsed_time_min']
        ax2.scatter(results['predicted_elapsed_time_min'],
                    residuals, alpha=0.6, s=20)
        ax2.axhline(y=0, color='r', linestyle='--', lw=2)
        ax2.set_xlabel('Predicted Time (minutes)')
        ax2.set_ylabel('Residuals (minutes)')
        ax2.set_title('Residuals vs Predicted Values')
        ax2.grid(True, alpha=0.3)

        # 3. Error Distribution Histogram
        ax3 = axes[0, 2]
        ax3.hist(results['absolute_error_min'],
                 bins=30, alpha=0.7, edgecolor='black')
        ax3.set_xlabel('Absolute Error (minutes)')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Distribution of Absolute Errors')
        ax3.grid(True, alpha=0.3)

        # 4. Percentage Error Distribution
        ax4 = axes[1, 0]
        ax4.hist(results['percentage_error'], bins=30,
                 alpha=0.7, edgecolor='black')
        ax4.set_xlabel('Percentage Error (%)')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Distribution of Percentage Errors')
        ax4.grid(True, alpha=0.3)

        # 5. Feature Importance Bar Plot
        ax5 = axes[1, 1]
        importance = self.random_forest.feature_importances_
        features = self.feature_names
        bars = ax5.bar(features, importance, alpha=0.7, edgecolor='black')
        ax5.set_xlabel('Features')
        ax5.set_ylabel('Importance')
        ax5.set_title('Feature Importance')
        ax5.tick_params(axis='x', rotation=45)
        ax5.grid(True, alpha=0.3)

        # Add value labels on bars
        for bar, imp in zip(bars, importance):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                     f'{imp:.3f}', ha='center', va='bottom', fontsize=9)

        # 6. Cross-validation Results (if available)
        ax6 = axes[1, 2]
        if hasattr(self, 'cv_results'):
            cv_r2 = self.cv_results['r2_scores']
            ax6.boxplot(cv_r2, patch_artist=True)
            ax6.set_ylabel('R² Score')
            ax6.set_title('Cross-Validation R² Scores')
            ax6.set_xticklabels(['CV Folds'])
            ax6.grid(True, alpha=0.3)
        else:
            # Show prediction accuracy by distance range
            distance_bins = pd.cut(results['distance_m'], bins=5)
            accuracy_by_distance = results.groupby(
                distance_bins)['percentage_error'].mean()
            ax6.bar(range(len(accuracy_by_distance)),
                    accuracy_by_distance.values, alpha=0.7, edgecolor='black')
            ax6.set_xlabel('Distance Range (m)')
            ax6.set_ylabel('Average Percentage Error (%)')
            ax6.set_title('Error by Distance Range')
            ax6.set_xticks(range(len(accuracy_by_distance)))
            ax6.set_xticklabels([f'{int(interval.left/1000)}-{int(interval.right/1000)}k'
                                for interval in accuracy_by_distance.index], rotation=45)
            ax6.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save the plot
        plot_filename = f'marathon_evaluation_plots_{self.file_counter-1}.png'
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"✅ Evaluation plots saved to '{plot_filename}'")

        # Show the plot
        plt.show()

        # Create additional detailed plots
        self._create_detailed_plots(results)

        return fig

    def _create_detailed_plots(self, results):
        """
        Create additional detailed plots for deeper analysis
        """
        # Create a second figure for detailed analysis
        fig2, axes2 = plt.subplots(2, 2, figsize=(15, 10))
        fig2.suptitle('Detailed Model Analysis',
                      fontsize=14, fontweight='bold')

        # 1. Error vs Distance
        ax1 = axes2[0, 0]
        ax1.scatter(results['distance_m'],
                    results['absolute_error_min'], alpha=0.6, s=20)
        ax1.set_xlabel('Distance (meters)')
        ax1.set_ylabel('Absolute Error (minutes)')
        ax1.set_title('Prediction Error vs Distance')
        ax1.grid(True, alpha=0.3)

        # 2. Error vs Elevation
        ax2 = axes2[0, 1]
        ax2.scatter(results['elevation_gain_m'],
                    results['absolute_error_min'], alpha=0.6, s=20)
        ax2.set_xlabel('Elevation Gain (meters)')
        ax2.set_ylabel('Absolute Error (minutes)')
        ax2.set_title('Prediction Error vs Elevation Gain')
        ax2.grid(True, alpha=0.3)

        # 3. Error vs Training Volume
        ax3 = axes2[1, 0]
        ax3.scatter(results['mean_km_per_week'],
                    results['absolute_error_min'], alpha=0.6, s=20)
        ax3.set_xlabel('Mean Training km/week')
        ax3.set_ylabel('Absolute Error (minutes)')
        ax3.set_title('Prediction Error vs Training Volume')
        ax3.grid(True, alpha=0.3)

        # 4. Error by Gender
        ax4 = axes2[1, 1]
        gender_errors = results.groupby('gender_binary')[
            'absolute_error_min'].mean()
        gender_labels = ['Female' if x ==
                         0 else 'Male' for x in gender_errors.index]
        bars = ax4.bar(gender_labels, gender_errors.values,
                       alpha=0.7, edgecolor='black')
        ax4.set_ylabel('Average Absolute Error (minutes)')
        ax4.set_title('Prediction Error by Gender')
        ax4.grid(True, alpha=0.3)

        # Add value labels on bars
        for bar, error in zip(bars, gender_errors.values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                     f'{error:.1f}', ha='center', va='bottom', fontsize=10)

        plt.tight_layout()

        # Save the detailed plots
        detailed_plot_filename = f'marathon_detailed_analysis_{self.file_counter-1}.png'
        plt.savefig(detailed_plot_filename, dpi=300, bbox_inches='tight')
        print(f"✅ Detailed analysis plots saved to '{detailed_plot_filename}'")

        plt.show()

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

        # Step 3: Perform cross-validation
        self.cv_results = self.perform_cross_validation()

        # Step 4: Train model
        self.train_model()

        # Step 5: Evaluate model
        metrics = self.evaluate_model()

        # Step 6: Feature importance
        self.analyze_feature_importance()

        # Step 7: Generate predictions CSV
        results = self.generate_predictions_csv()

        # Step 8: Create evaluation plots
        self.create_evaluation_plots(results)

        print("\n🎉 Model training and evaluation complete!")
        print("\n💡 Key Insights:")
        print("- Random Forest model trained on 80% of clean dataset")
        print("- Cross-validation performed for robust performance estimation")
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
