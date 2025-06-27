#!/usr/bin/env python3
"""
Marathon Time Prediction Model 1
Predicts race time from gender, elevation, and distance using scikit-learn.

This implementation follows the theoretical framework:
1. Load and inspect data
2. Data preprocessing (feature selection, encoding, scaling)
3. Train/test split
4. Model training (Linear Regression + Random Forest)
5. Evaluation and visualization
6. Cross-validation
7. Feature importance analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class MarathonPredictor:
    """
    A comprehensive marathon time prediction system using scikit-learn.
    """

    def __init__(self, data_path="raw-data-kaggle.csv"):
        """
        Initialize the predictor with data path.

        Args:
            data_path (str): Path to the CSV file containing marathon data
        """
        self.data_path = data_path
        self.df = None
        self.df_model = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.lr_model = None
        self.rf_model = None
        self.scaler = StandardScaler()

    def load_and_inspect_data(self):
        """
        Step 1: Load and inspect the data
        """
        print("🔍 STEP 1: Loading and Inspecting Data")
        print("=" * 50)

        try:
            # Load the CSV file with semicolon separator (Kaggle format)
            self.df = pd.read_csv(self.data_path, sep=';')
            print(f"✅ Data loaded successfully!")
            print(f"📊 Dataset shape: {self.df.shape}")
            print(f"📋 Columns: {list(self.df.columns)}")

            # Display first few rows
            print("\n📋 First 5 rows:")
            print(self.df.head())

            # Basic info about the dataset
            print("\n📈 Dataset Info:")
            print(self.df.info())

            # Check for missing values
            print("\n❓ Missing Values:")
            missing_values = self.df.isnull().sum()
            print(missing_values[missing_values > 0])

            # Basic statistics
            print("\n📊 Basic Statistics:")
            print(self.df.describe())

        except FileNotFoundError:
            print(f"❌ Error: File '{self.data_path}' not found!")
            print("Please make sure the CSV file is in the current directory.")
            return False

        return True

    def preprocess_data(self):
        """
        Step 2: Data preprocessing - feature selection, encoding, scaling
        """
        print("\n🧹 STEP 2: Data Preprocessing")
        print("=" * 50)

        # Map Kaggle data columns to our expected format
        column_mapping = {
            'gender': 'gender',
            'distance (m)': 'distance',
            'elapsed time (s)': 'race_time',
            'elevation gain (m)': 'elevation_gain'
        }

        # Check if required columns exist
        missing_columns = []
        for kaggle_col, our_col in column_mapping.items():
            if kaggle_col not in self.df.columns:
                missing_columns.append(kaggle_col)

        if missing_columns:
            print(f"❌ Missing required columns: {missing_columns}")
            print("Available columns:", list(self.df.columns))
            return False

        # Select and rename relevant columns
        self.df_model = self.df[[
            'gender',
            'distance (m)',
            'elapsed time (s)',
            'elevation gain (m)'
        ]].copy()

        # Rename columns for consistency
        self.df_model.columns = [
            'gender', 'distance', 'race_time', 'elevation_gain']

        print(f"✅ Selected columns: {list(self.df_model.columns)}")
        print(f"📊 Shape after selection: {self.df_model.shape}")

        # Convert distance from meters to kilometers
        self.df_model['distance'] = self.df_model['distance'] / 1000
        print("✅ Converted distance from meters to kilometers")

        # Filter for marathon distances (40-45 km) for better prediction accuracy
        initial_rows = len(self.df_model)
        self.df_model = self.df_model[(self.df_model['distance'] >= 40) & (
            self.df_model['distance'] <= 45)]
        final_rows = len(self.df_model)
        print(
            f"✅ Filtered for marathon distances (40-45 km): {initial_rows - final_rows} rows removed")
        print(f"📊 Remaining marathon records: {final_rows}")

        # Convert race time from seconds to minutes
        self.df_model['race_time'] = self.df_model['race_time'] / 60
        print("✅ Converted race time from seconds to minutes")

        # Map gender values: M -> MALE, F -> FEMALE
        gender_mapping = {'M': 'MALE', 'F': 'FEMALE'}
        self.df_model['gender'] = self.df_model['gender'].map(gender_mapping)
        print(f"✅ Mapped gender values: {gender_mapping}")

        # Drop rows with missing values
        initial_rows = len(self.df_model)
        self.df_model = self.df_model.dropna()
        final_rows = len(self.df_model)

        print(
            f"🧹 Removed {initial_rows - final_rows} rows with missing values")
        print(f"📊 Final shape: {self.df_model.shape}")

        # Encode gender for machine learning
        print("\n🔤 Encoding categorical variables:")
        print("Gender values:", self.df_model['gender'].unique())

        # Create gender encoding mapping
        unique_genders = self.df_model['gender'].unique()
        self.gender_mapping = {gender: idx for idx,
                               gender in enumerate(unique_genders)}
        self.df_model['gender_encoded'] = self.df_model['gender'].map(
            self.gender_mapping)
        print(f"✅ Gender encoding: {self.gender_mapping}")

        # Define features and target
        self.X = self.df_model[['gender_encoded',
                                'elevation_gain', 'distance']]
        self.y = self.df_model['race_time']

        print(f"\n🎯 Features shape: {self.X.shape}")
        print(f"🎯 Target shape: {self.y.shape}")

        # Display feature statistics
        print("\n📊 Feature Statistics:")
        print(self.X.describe())

        return True

    def split_data(self):
        """
        Step 3: Split data into training and test sets with stratified sampling
        """
        print("\n📊 STEP 3: Data Splitting with Stratified Sampling")
        print("=" * 50)

        # Create performance bins for stratified sampling
        # We'll create 5 performance levels based on race time percentiles
        time_percentiles = [20, 40, 60, 80, 100]
        performance_bins = []

        for i, percentile in enumerate(time_percentiles):
            if i == 0:
                threshold = self.y.quantile(percentile / 100)
                performance_bins.append(f"Level_{i+1}_Fast")
                self.df_model.loc[self.y <= threshold,
                                  'performance_level'] = f"Level_{i+1}_Fast"
            else:
                prev_threshold = self.y.quantile((time_percentiles[i-1]) / 100)
                threshold = self.y.quantile(percentile / 100)
                performance_bins.append(f"Level_{i+1}")
                self.df_model.loc[(self.y > prev_threshold) & (
                    self.y <= threshold), 'performance_level'] = f"Level_{i+1}"

        # Update X and y to include performance level
        self.X = self.df_model[['gender_encoded',
                                'elevation_gain', 'distance']]
        self.y = self.df_model['race_time']

        print("📊 Performance Level Distribution:")
        level_counts = self.df_model['performance_level'].value_counts(
        ).sort_index()
        for level, count in level_counts.items():
            print(f"   {level}: {count} runners")

        # Stratified split based on performance levels
        from sklearn.model_selection import train_test_split

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y,
            test_size=0.3,
            random_state=42,
            stratify=self.df_model['performance_level']
        )

        print(f"\n✅ Training set: {self.X_train.shape[0]} samples")
        print(f"✅ Test set: {self.X_test.shape[0]} samples")

        # Verify stratification worked
        print("\n📊 Verifying Stratification:")
        train_indices = self.X_train.index
        test_indices = self.X_test.index

        train_levels = self.df_model.loc[train_indices,
                                         'performance_level'].value_counts().sort_index()
        test_levels = self.df_model.loc[test_indices,
                                        'performance_level'].value_counts().sort_index()

        print("Training set performance levels:")
        for level in sorted(train_levels.index):
            print(f"   {level}: {train_levels[level]} runners")

        print("Test set performance levels:")
        for level in sorted(test_levels.index):
            print(f"   {level}: {test_levels[level]} runners")

        # Scale features (optional but recommended for linear regression)
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)

        print("✅ Features scaled using StandardScaler")

        return True

    def train_models(self):
        """
        Step 4: Train Linear Regression and Random Forest models
        """
        print("\n🧪 STEP 4: Model Training")
        print("=" * 50)

        # Linear Regression
        print("🔵 Training Linear Regression...")
        self.lr_model = LinearRegression()
        self.lr_model.fit(self.X_train_scaled, self.y_train)
        print("✅ Linear Regression trained!")

        # Random Forest Regressor
        print("🟢 Training Random Forest...")
        self.rf_model = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
        # RF doesn't need scaling
        self.rf_model.fit(self.X_train, self.y_train)
        print("✅ Random Forest trained!")

        return True

    def evaluate_models(self):
        """
        Step 5: Evaluate model performance
        """
        print("\n📈 STEP 5: Model Evaluation")
        print("=" * 50)

        def evaluate_model(model, X, y, name, scaled=False):
            """Helper function to evaluate a model"""
            if scaled:
                preds = model.predict(X)
            else:
                preds = model.predict(X)

            mae = mean_absolute_error(y, preds)
            rmse = np.sqrt(mean_squared_error(y, preds))
            r2 = r2_score(y, preds)

            print(f"\n📊 {name} Results:")
            print(f"   MAE  : {mae:.2f} minutes")
            print(f"   RMSE : {rmse:.2f} minutes")
            print(f"   R²   : {r2:.4f}")

            return preds, mae, rmse, r2

        # Evaluate on test set
        lr_preds, lr_mae, lr_rmse, lr_r2 = evaluate_model(
            self.lr_model, self.X_test_scaled, self.y_test, "Linear Regression", scaled=True
        )

        rf_preds, rf_mae, rf_rmse, rf_r2 = evaluate_model(
            self.rf_model, self.X_test, self.y_test, "Random Forest", scaled=False
        )

        # Store results for comparison
        self.results = {
            'Linear Regression': {'MAE': lr_mae, 'RMSE': lr_rmse, 'R2': lr_r2, 'preds': lr_preds},
            'Random Forest': {'MAE': rf_mae, 'RMSE': rf_rmse, 'R2': rf_r2, 'preds': rf_preds}
        }

        return True

    def cross_validation(self):
        """
        Step 6: Cross-validation for more robust evaluation
        """
        print("\n🔄 STEP 6: Cross-Validation")
        print("=" * 50)

        # 5-fold cross-validation
        cv_scores_lr = cross_val_score(
            self.lr_model, self.X_train_scaled, self.y_train,
            cv=5, scoring='r2'
        )

        cv_scores_rf = cross_val_score(
            self.rf_model, self.X_train, self.y_train,
            cv=5, scoring='r2'
        )

        print("📊 Cross-Validation R² Scores (5-fold):")
        print(
            f"   Linear Regression: {cv_scores_lr.mean():.4f} (+/- {cv_scores_lr.std() * 2:.4f})")
        print(
            f"   Random Forest: {cv_scores_rf.mean():.4f} (+/- {cv_scores_rf.std() * 2:.4f})")

        return True

    def analyze_feature_importance(self):
        """
        Step 7: Analyze feature importance (Random Forest)
        """
        print("\n🧠 STEP 7: Feature Importance Analysis")
        print("=" * 50)

        # Get feature importance from Random Forest
        feature_names = self.X.columns
        importances = self.rf_model.feature_importances_

        # Create feature importance DataFrame
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)

        print("📊 Feature Importance (Random Forest):")
        for idx, row in importance_df.iterrows():
            print(f"   {row['Feature']}: {row['Importance']:.4f}")

        # Plot feature importance
        plt.figure(figsize=(10, 6))
        sns.barplot(data=importance_df, x='Importance', y='Feature')
        plt.title('Feature Importance - Random Forest',
                  fontsize=16, fontweight='bold')
        plt.xlabel('Importance Score', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()

        return importance_df

    def visualize_predictions(self):
        """
        Step 8: Visualize model predictions
        """
        print("\n📉 STEP 8: Prediction Visualizations")
        print("=" * 50)

        # Create subplots for both models
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Linear Regression predictions
        lr_preds = self.results['Linear Regression']['preds']
        axes[0].scatter(self.y_test, lr_preds, alpha=0.6, color='blue')
        axes[0].plot([self.y_test.min(), self.y_test.max()],
                     [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
        axes[0].set_xlabel('Actual Race Time (minutes)', fontsize=12)
        axes[0].set_ylabel('Predicted Race Time (minutes)', fontsize=12)
        axes[0].set_title('Linear Regression: Actual vs Predicted',
                          fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)

        # Random Forest predictions
        rf_preds = self.results['Random Forest']['preds']
        axes[1].scatter(self.y_test, rf_preds, alpha=0.6, color='green')
        axes[1].plot([self.y_test.min(), self.y_test.max()],
                     [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
        axes[1].set_xlabel('Actual Race Time (minutes)', fontsize=12)
        axes[1].set_ylabel('Predicted Race Time (minutes)', fontsize=12)
        axes[1].set_title('Random Forest: Actual vs Predicted',
                          fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('predictions_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

        # Error distribution
        plt.figure(figsize=(12, 5))

        lr_errors = self.y_test - lr_preds
        rf_errors = self.y_test - rf_preds

        plt.subplot(1, 2, 1)
        plt.hist(lr_errors, bins=30, alpha=0.7,
                 color='blue', edgecolor='black')
        plt.xlabel('Prediction Error (minutes)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('Linear Regression Error Distribution',
                  fontsize=14, fontweight='bold')
        plt.axvline(0, color='red', linestyle='--', alpha=0.7)
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        plt.hist(rf_errors, bins=30, alpha=0.7,
                 color='green', edgecolor='black')
        plt.xlabel('Prediction Error (minutes)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('Random Forest Error Distribution',
                  fontsize=14, fontweight='bold')
        plt.axvline(0, color='red', linestyle='--', alpha=0.7)
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('error_distributions.png', dpi=300, bbox_inches='tight')
        plt.show()

        return True

    def predict_for_user(self, gender, elevation_gain, distance):
        """
        Make predictions for a specific user
        """
        print(f"\n🎯 Making Prediction for User:")
        print(f"   Gender: {gender}")
        print(f"   Elevation Gain: {elevation_gain} meters")
        print(f"   Distance: {distance} km")

        # Encode gender using the mapping from training
        if gender not in self.gender_mapping:
            print(f"❌ Error: Gender '{gender}' not found in training data")
            print(f"Available genders: {list(self.gender_mapping.keys())}")
            return None, None

        gender_encoded = self.gender_mapping[gender]

        # Create feature array
        features = np.array([[gender_encoded, elevation_gain, distance]])

        # Make predictions
        lr_pred = self.lr_model.predict(self.scaler.transform(features))[0]
        rf_pred = self.rf_model.predict(features)[0]

        print(f"\n📊 Predictions:")
        print(f"   Linear Regression: {lr_pred:.1f} minutes")
        print(f"   Random Forest: {rf_pred:.1f} minutes")

        return lr_pred, rf_pred

    def analyze_performance_distribution(self):
        """
        Analyze the performance distribution across training and test sets
        """
        print("\n📊 Performance Distribution Analysis")
        print("=" * 50)

        # Get indices for training and test sets
        train_indices = self.X_train.index
        test_indices = self.X_test.index

        # Analyze performance statistics
        train_stats = self.y_train.describe()
        test_stats = self.y_test.describe()

        print("📈 Performance Statistics (minutes):")
        print("Training Set:")
        print(f"   Mean: {train_stats['mean']:.1f}")
        print(f"   Std:  {train_stats['std']:.1f}")
        print(f"   Min:  {train_stats['min']:.1f}")
        print(f"   Max:  {train_stats['max']:.1f}")

        print("\nTest Set:")
        print(f"   Mean: {test_stats['mean']:.1f}")
        print(f"   Std:  {test_stats['std']:.1f}")
        print(f"   Min:  {test_stats['min']:.1f}")
        print(f"   Max:  {test_stats['max']:.1f}")

        # Calculate performance level distribution
        train_levels = self.df_model.loc[train_indices,
                                         'performance_level'].value_counts().sort_index()
        test_levels = self.df_model.loc[test_indices,
                                        'performance_level'].value_counts().sort_index()

        print("\n📊 Performance Level Distribution:")
        print("Level | Training | Test | % Training | % Test")
        print("-" * 45)

        total_train = len(train_indices)
        total_test = len(test_indices)

        for level in sorted(train_levels.index):
            train_count = train_levels[level]
            test_count = test_levels[level]
            train_pct = (train_count / total_train) * 100
            test_pct = (test_count / total_test) * 100

            print(
                f"{level:6} | {train_count:8} | {test_count:4} | {train_pct:10.1f}% | {test_pct:6.1f}%")

        # Visualize performance distribution
        plt.figure(figsize=(15, 5))

        # Histogram comparison
        plt.subplot(1, 3, 1)
        plt.hist(self.y_train, bins=20, alpha=0.7,
                 label='Training', color='blue', edgecolor='black')
        plt.hist(self.y_test, bins=20, alpha=0.7,
                 label='Test', color='red', edgecolor='black')
        plt.xlabel('Race Time (minutes)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('Performance Distribution', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Performance level distribution
        plt.subplot(1, 3, 2)
        levels = sorted(train_levels.index)
        train_counts = [train_levels[level] for level in levels]
        test_counts = [test_levels[level] for level in levels]

        x = np.arange(len(levels))
        width = 0.35

        plt.bar(x - width/2, train_counts, width, label='Training', alpha=0.8)
        plt.bar(x + width/2, test_counts, width, label='Test', alpha=0.8)
        plt.xlabel('Performance Level', fontsize=12)
        plt.ylabel('Number of Runners', fontsize=12)
        plt.title('Performance Level Distribution',
                  fontsize=14, fontweight='bold')
        plt.xticks(x, levels, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Box plot comparison
        plt.subplot(1, 3, 3)
        plt.boxplot([self.y_train, self.y_test], labels=['Training', 'Test'])
        plt.ylabel('Race Time (minutes)', fontsize=12)
        plt.title('Performance Comparison', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('performance_distribution.png',
                    dpi=300, bbox_inches='tight')
        plt.show()

        return True

    def run_complete_pipeline(self):
        """
        Run the complete marathon prediction pipeline
        """
        print("🏃‍♂️ MARATHON TIME PREDICTION PIPELINE")
        print("=" * 60)

        # Step 1: Load and inspect data
        if not self.load_and_inspect_data():
            return False

        # Step 2: Preprocess data
        if not self.preprocess_data():
            return False

        # Step 3: Split data
        if not self.split_data():
            return False

        # Step 4: Train models
        if not self.train_models():
            return False

        # Step 5: Evaluate models
        if not self.evaluate_models():
            return False

        # Step 6: Cross-validation
        if not self.cross_validation():
            return False

        # Step 7: Feature importance
        importance_df = self.analyze_feature_importance()

        # Step 8: Visualizations
        self.visualize_predictions()

        # Step 9: Performance Distribution Analysis
        if not self.analyze_performance_distribution():
            return False

        print("\n✅ Pipeline completed successfully!")
        print("📁 Generated files:")
        print("   - feature_importance.png")
        print("   - predictions_comparison.png")
        print("   - error_distributions.png")
        print("   - performance_distribution.png")

        return True


def main():
    """
    Main function to run the marathon prediction pipeline
    """
    # Initialize the predictor
    predictor = MarathonPredictor()

    # Run the complete pipeline
    success = predictor.run_complete_pipeline()

    if success:
        print("\n🎉 All done! Your marathon prediction model is ready!")
        print("\n💡 Next steps:")
        print("   1. Check the generated plots for model insights")
        print("   2. Use the models to make predictions for new runners")
        print("   3. Consider adding more features for better performance")
    else:
        print("\n❌ Pipeline failed. Please check your data and try again.")


if __name__ == "__main__":
    main()
