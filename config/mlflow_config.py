# MLflow Configuration for SageMaker Studio
# This file contains configuration settings for MLflow tracking

import os

# MLflow configuration
MLFLOW_CONFIG = {
    # Default experiment name
    'default_experiment_name': 'iris_classification',
    
    # Tracking URI - can be overridden with environment variable
    'tracking_uri': os.getenv('MLFLOW_TRACKING_URI', './mlruns'),
    
    # Model registry settings
    'model_registry': {
        'default_registered_model_name': 'iris_model',
        'staging_alias': 'staging',
        'production_alias': 'production'
    },
    
    # Artifact storage - for SageMaker, this should be an S3 bucket
    'artifact_location': os.getenv('MLFLOW_ARTIFACT_LOCATION', './mlartifacts'),
    
    # SageMaker specific settings
    'sagemaker': {
        'role': os.getenv('SAGEMAKER_ROLE'),
        'region': os.getenv('AWS_DEFAULT_REGION', 'us-east-1'),
        'bucket': os.getenv('SAGEMAKER_DEFAULT_BUCKET'),
        'instance_type': 'ml.m5.large',
        'framework_version': '1.2-1'
    }
}

# Model configurations for hyperparameter tuning
MODEL_CONFIGS = {
    'logistic_regression': {
        'param_grid': {
            'C': [0.1, 1.0, 10.0],
            'solver': ['liblinear', 'lbfgs'],
            'max_iter': [1000, 2000]
        }
    },
    'random_forest': {
        'param_grid': {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    },
    'svm': {
        'param_grid': {
            'C': [0.1, 1, 10],
            'kernel': ['rbf', 'linear'],
            'gamma': ['scale', 'auto', 0.1, 1]
        }
    }
}

# Metrics to track
TRACKING_METRICS = [
    'accuracy',
    'precision_macro',
    'precision_weighted',
    'recall_macro', 
    'recall_weighted',
    'f1_macro',
    'f1_weighted'
]

# SageMaker Studio MLflow setup instructions
SAGEMAKER_SETUP_INSTRUCTIONS = """
To use this project with SageMaker Studio MLflow:

1. Create an MLflow Tracking Server in SageMaker Studio:
   - Go to SageMaker Console > Studio > MLflow
   - Create a new tracking server
   - Note the tracking server ARN and endpoint URL

2. Set environment variables in your SageMaker Studio notebook:
   export MLFLOW_TRACKING_URI=<your-mlflow-tracking-server-url>
   export MLFLOW_ARTIFACT_LOCATION=s3://<your-s3-bucket>/mlflow-artifacts/
   export SAGEMAKER_ROLE=<your-sagemaker-execution-role-arn>
   export SAGEMAKER_DEFAULT_BUCKET=<your-s3-bucket-name>

3. Install required packages:
   pip install -r requirements.txt

4. Run the training pipeline:
   python src/models/train.py --model all --experiment iris_sagemaker_experiment

5. Analyze results:
   python src/models/evaluate.py --experiment iris_sagemaker_experiment --register-best

6. Access MLflow UI through SageMaker Studio MLflow interface
"""