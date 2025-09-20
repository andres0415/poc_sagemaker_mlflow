#!/bin/bash

# Quick start script for running the iris classification experiments
# This script demonstrates the complete ML pipeline with MLflow tracking

echo "🌸 Iris Classification MLflow Demo"
echo "=================================="

# Check if virtual environment exists, create if not
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Create necessary directories
echo "Creating directories..."
mkdir -p data/processed logs

echo ""
echo "🔄 Running Data Preprocessing..."
python src/data/data_loader.py

echo ""
echo "🤖 Training Models with MLflow Tracking..."
python src/models/train.py --model all --experiment iris_classification_demo

echo ""
echo "📊 Analyzing Results..."
python src/models/evaluate.py --experiment iris_classification_demo --register-best --model-name iris_champion_model

echo ""
echo "✅ Demo completed successfully!"
echo ""
echo "📋 What was accomplished:"
echo "  - Loaded and preprocessed iris dataset"
echo "  - Trained 5 different ML models:"
echo "    • Logistic Regression"
echo "    • Random Forest"
echo "    • Support Vector Machine"
echo "    • Naive Bayes"
echo "    • Decision Tree"
echo "  - Tracked all experiments with MLflow"
echo "  - Registered the best model to MLflow Model Registry"
echo ""
echo "🔍 To view results:"
echo "  1. Start MLflow UI: mlflow ui"
echo "  2. Open browser to: http://localhost:5000"
echo "  3. Explore experiments and model registry"
echo ""
echo "📓 For interactive analysis:"
echo "  1. Start Jupyter: jupyter notebook"
echo "  2. Open: notebooks/iris_mlflow_demo.ipynb"
echo ""
echo "🔧 For SageMaker Studio:"
echo "  1. Upload this repository to SageMaker Studio"
echo "  2. Configure MLflow tracking server URI"
echo "  3. Set S3 bucket for artifact storage"
echo "  4. Run the training scripts or notebook"