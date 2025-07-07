"""
Setup script for Marathon Time Predictor package.
"""

from setuptools import setup, find_packages
import os

# Read the README file


def read_readme():
    """Read README.md file."""
    readme_path = os.path.join(os.path.dirname(__file__), "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8") as f:
            return f.read()
    return "Marathon Time Predictor - A machine learning API for predicting marathon finish times."

# Read requirements


def read_requirements(filename):
    """Read requirements from file."""
    requirements_path = os.path.join(os.path.dirname(__file__), filename)
    if os.path.exists(requirements_path):
        with open(requirements_path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip() and not line.startswith("#")]
    return []


setup(
    name="marathon-time-predictor",
    version="1.0.0",
    author="Marathon Time Predictor Contributors",
    author_email="your.email@example.com",
    description="A machine learning API for predicting marathon finish times",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/varlopecar/marathon-time-predictor",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Internet :: WWW/HTTP :: HTTP Servers",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements("requirements.txt"),
    extras_require={
        "dev": read_requirements("requirements-dev.txt"),
        "test": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "pytest-asyncio>=0.21.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "marathon-predictor=marathon_prediction:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.csv", "*.pkl"],
    },
    keywords="marathon, prediction, machine learning, api, fastapi, scikit-learn",
    project_urls={
        "Bug Reports": "https://github.com/varlopecar/marathon-time-predictor/issues",
        "Source": "https://github.com/varlopecar/marathon-time-predictor",
        "Documentation": "https://github.com/varlopecar/marathon-time-predictor#readme",
    },
)
