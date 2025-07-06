# Marathon Time Predictor

A machine learning API that predicts marathon finish times based on training data, race conditions, and athlete characteristics. Built with FastAPI and scikit-learn.

## 🏃‍♂️ Features

- **Accurate Predictions**: Random Forest model trained on extensive marathon data
- **RESTful API**: FastAPI-based web service with automatic documentation
- **Input Validation**: Comprehensive validation for all user inputs
- **Model Insights**: Feature importance and cross-validation results
- **Production Ready**: Health checks, error handling, and proper logging

## 📊 Model Performance

- **R² Score**: ~0.85 (85% variance explained)
- **Mean Absolute Error**: ~15 minutes
- **Training Data**: ~30,000 marathon records
- **Features**: Distance, elevation, training volume, frequency, gender, experience level

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip or conda

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/marathon-time-predictor.git
   cd marathon-time-predictor
   ```

2. **Create virtual environment**

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the API**
   ```bash
   python marathon_api.py
   ```

The API will be available at `http://localhost:8000`

## 📖 Usage

### API Endpoints

#### Health Check

```bash
GET /health
```

#### Make Prediction

```bash
POST /predict
```

**Request Body:**

```json
{
  "distance_km": 42.2,
  "elevation_gain_m": 200,
  "mean_km_per_week": 60,
  "mean_training_days_per_week": 5,
  "gender": "male",
  "level": 2
}
```

**Response:**

```json
{
  "success": true,
  "prediction": {
    "time_seconds": 14400.0,
    "time_minutes": 240.0,
    "time_hours": 4.0,
    "time_string": "04:00:00",
    "pace_minutes_per_km": 5.7,
    "distance_km": 42.2
  },
  "model_info": {
    "model_type": "Random Forest",
    "features_used": [
      "distance_m",
      "elevation_gain_m",
      "mean_km_per_week",
      "mean_training_days_per_week",
      "gender_binary",
      "level"
    ],
    "training_samples": "~30,000"
  },
  "feature_importance": {
    "mean_km_per_week": 0.45,
    "level": 0.25,
    "distance_m": 0.15,
    "mean_training_days_per_week": 0.1,
    "elevation_gain_m": 0.03,
    "gender_binary": 0.02
  }
}
```

### Python Usage

```python
from marathon_prediction import MarathonPrediction

# Initialize model
model = MarathonPrediction()

# Load or train model
if not model.load_model():
    model.train_model()

# Make prediction
user_data = {
    'distance_km': 42.2,
    'elevation_gain_m': 200,
    'mean_km_per_week': 60,
    'mean_training_days_per_week': 5,
    'gender': 'male',
    'level': 2
}

result = model.predict_time(user_data)
print(f"Predicted time: {result['prediction']['time_string']}")
```

## 🧪 Testing

Run the test suite:

```bash
pytest
```

Test the API:

```bash
python test_simple_api.py
```

## 📁 Project Structure

```
marathon-time-predictor/
├── marathon_prediction.py    # Core ML model and prediction logic
├── marathon_api.py          # FastAPI web service
├── requirements.txt         # Python dependencies
├── README.md               # This file
├── CONTRIBUTING.md         # Contribution guidelines
├── LICENSE                 # Project license
├── setup.py               # Package setup
├── .gitignore             # Git ignore rules
├── .pre-commit-config.yaml # Pre-commit hooks
├── pyproject.toml         # Project configuration
├── tests/                 # Test files
│   └── test_marathon_prediction.py
├── data/                  # Data files (not in repo)
│   ├── clean_dataset.csv
│   └── marathon_model.pkl
└── docs/                  # Documentation
    └── api.md
```

## 🔧 Development

### Setting up development environment

1. **Install development dependencies**

   ```bash
   pip install -r requirements-dev.txt
   ```

2. **Install pre-commit hooks**

   ```bash
   pre-commit install
   ```

3. **Run linting and formatting**
   ```bash
   black .
   flake8 .
   ```

### Training the Model

```python
from marathon_prediction import MarathonPrediction

model = MarathonPrediction()
metrics = model.train_model()
print(f"Model trained with R² score: {metrics['r2_score']:.3f}")
```

## 📈 Model Details

### Features Used

- **distance_m**: Race distance in meters
- **elevation_gain_m**: Total elevation gain in meters
- **mean_km_per_week**: Average weekly training volume
- **mean_training_days_per_week**: Training frequency
- **gender_binary**: Athlete gender (0=male, 1=female)
- **level**: Experience level (1=beginner, 2=intermediate, 3=advanced)

### Model Architecture

- **Algorithm**: Random Forest Regressor
- **Estimators**: 100 trees
- **Cross-validation**: 5-fold
- **Validation**: Train/test split (80/20)

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### How to contribute

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Dataset provided by [source]
- Built with FastAPI and scikit-learn
- Inspired by the running community

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/marathon-time-predictor/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/marathon-time-predictor/discussions)
- **Email**: your.email@example.com

## 🔄 Changelog

### v1.0.0 (2024-01-XX)

- Initial release
- FastAPI web service
- Random Forest prediction model
- Comprehensive input validation
- Feature importance analysis
