#!/usr/bin/env python3
"""
Deployment test script for Marathon Time Prediction API
This script tests the application startup process to identify deployment issues.
"""

import os
import sys
import time
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_imports():
    """Test if all required modules can be imported."""
    logger.info("Testing imports...")

    try:
        import fastapi
        logger.info("✓ FastAPI imported successfully")
    except ImportError as e:
        logger.error(f"✗ FastAPI import failed: {e}")
        return False

    try:
        import uvicorn
        logger.info("✓ Uvicorn imported successfully")
    except ImportError as e:
        logger.error(f"✗ Uvicorn import failed: {e}")
        return False

    try:
        import pandas
        logger.info("✓ Pandas imported successfully")
    except ImportError as e:
        logger.error(f"✗ Pandas import failed: {e}")
        return False

    try:
        import numpy
        logger.info("✓ NumPy imported successfully")
    except ImportError as e:
        logger.error(f"✗ NumPy import failed: {e}")
        return False

    try:
        import sklearn
        logger.info("✓ Scikit-learn imported successfully")
    except ImportError as e:
        logger.error(f"✗ Scikit-learn import failed: {e}")
        return False

    return True


def test_model_loading():
    """Test model loading process."""
    logger.info("Testing model loading...")

    try:
        from marathon_prediction import MarathonPrediction
        logger.info("✓ MarathonPrediction imported successfully")

        model = MarathonPrediction()
        logger.info("✓ MarathonPrediction instance created")

        # Check if model file exists
        if os.path.exists("marathon_model.pkl"):
            file_size = os.path.getsize("marathon_model.pkl")
            logger.info(
                f"✓ Model file exists ({file_size / 1024 / 1024:.1f}MB)")

            # Try to load model
            if model.load_model():
                logger.info("✓ Model loaded successfully")
                return True
            else:
                logger.warning("⚠ Model file exists but failed to load")
                return False
        else:
            logger.info("ℹ No model file found, would need to train")
            return True

    except Exception as e:
        logger.error(f"✗ Model loading test failed: {e}")
        return False


def test_api_startup():
    """Test API startup process."""
    logger.info("Testing API startup...")

    try:
        from marathon_api import app
        logger.info("✓ FastAPI app imported successfully")

        # Test basic endpoint
        from fastapi.testclient import TestClient
        client = TestClient(app)

        response = client.get("/")
        if response.status_code == 200:
            logger.info("✓ Root endpoint responds successfully")
            return True
        else:
            logger.error(f"✗ Root endpoint failed: {response.status_code}")
            return False

    except Exception as e:
        logger.error(f"✗ API startup test failed: {e}")
        return False


def test_environment():
    """Test environment configuration."""
    logger.info("Testing environment...")

    # Check Python version
    python_version = sys.version_info
    logger.info(
        f"✓ Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")

    # Check required files
    required_files = ["requirements.txt",
                      "marathon_api.py", "marathon_prediction.py"]
    for file in required_files:
        if os.path.exists(file):
            logger.info(f"✓ {file} exists")
        else:
            logger.error(f"✗ {file} missing")
            return False

    # Check optional files
    optional_files = ["clean_dataset.csv", "marathon_model.pkl"]
    for file in optional_files:
        if os.path.exists(file):
            file_size = os.path.getsize(file) / 1024 / 1024
            logger.info(f"✓ {file} exists ({file_size:.1f}MB)")
        else:
            logger.info(f"ℹ {file} not found (optional)")

    return True


def main():
    """Run all deployment tests."""
    logger.info("Starting deployment tests...")

    tests = [
        ("Environment", test_environment),
        ("Imports", test_imports),
        ("Model Loading", test_model_loading),
        ("API Startup", test_api_startup),
    ]

    results = []
    for test_name, test_func in tests:
        logger.info(f"\n--- {test_name} Test ---")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}")
            results.append((test_name, False))

    # Summary
    logger.info("\n--- Test Summary ---")
    passed = 0
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{test_name}: {status}")
        if result:
            passed += 1

    logger.info(f"\nOverall: {passed}/{len(results)} tests passed")

    if passed == len(results):
        logger.info("🎉 All tests passed! Deployment should work.")
        return 0
    else:
        logger.error("❌ Some tests failed. Check the logs above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
