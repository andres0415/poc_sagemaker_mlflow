"""
Model training script with MLflow integration.
Supports multiple algorithms for comparison.
"""
import argparse
import sys
import os
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data.data_loader import load_iris_dataset, preprocess_data
from utils.mlflow_utils import MLflowTracker

def get_model_config(model_name: str) -> Dict[str, Any]:
    """
    Get model configuration based on model name.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Dictionary with model class and hyperparameters
    """
    configs = {
        'logistic_regression': {
            'model_class': LogisticRegression,
            'params': {
                'random_state': 42,
                'max_iter': 1000,
                'solver': 'liblinear'
            }
        },
        'random_forest': {
            'model_class': RandomForestClassifier,
            'params': {
                'n_estimators': 100,
                'random_state': 42,
                'max_depth': 10
            }
        },
        'svm': {
            'model_class': SVC,
            'params': {
                'random_state': 42,
                'kernel': 'rbf',
                'C': 1.0,
                'gamma': 'scale'
            }
        },
        'naive_bayes': {
            'model_class': GaussianNB,
            'params': {}
        },
        'decision_tree': {
            'model_class': DecisionTreeClassifier,
            'params': {
                'random_state': 42,
                'max_depth': 10,
                'min_samples_split': 5
            }
        }
    }
    
    if model_name not in configs:
        raise ValueError(f"Model '{model_name}' not supported. Available: {list(configs.keys())}")
    
    return configs[model_name]

def train_model(model_name: str, X_train: pd.DataFrame, y_train: pd.Series, 
               hyperparams: Dict[str, Any] = None) -> Any:
    """
    Train a model with given data and hyperparameters.
    
    Args:
        model_name: Name of the model to train
        X_train: Training features
        y_train: Training labels
        hyperparams: Additional hyperparameters to override defaults
        
    Returns:
        Trained model
    """
    config = get_model_config(model_name)
    
    # Merge hyperparameters
    params = config['params'].copy()
    if hyperparams:
        params.update(hyperparams)
    
    # Create and train model
    model = config['model_class'](**params)
    model.fit(X_train, y_train)
    
    return model, params

def run_experiment(model_name: str, experiment_name: str = "iris_classification",
                  hyperparams: Dict[str, Any] = None, tracking_uri: str = None) -> str:
    """
    Run a complete training experiment with MLflow tracking.
    
    Args:
        model_name: Name of the model to train
        experiment_name: MLflow experiment name
        hyperparams: Model hyperparameters
        tracking_uri: MLflow tracking server URI
        
    Returns:
        Run ID
    """
    # Initialize MLflow tracker
    tracker = MLflowTracker(experiment_name, tracking_uri)
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    X, y = load_iris_dataset()
    data = preprocess_data(X, y)
    
    X_train, X_test = data['X_train'], data['X_test']
    y_train, y_test = data['y_train'], data['y_test']
    
    # Start MLflow run
    run_name = f"{model_name}_experiment"
    tracker.start_run(run_name)
    
    try:
        # Train model
        print(f"Training {model_name}...")
        model, params = train_model(model_name, X_train, y_train, hyperparams)
        
        # Make predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Calculate metrics
        train_metrics = tracker.calculate_metrics(y_train, y_train_pred)
        test_metrics = tracker.calculate_metrics(y_test, y_test_pred)
        
        # Log parameters
        log_params = {
            'model_name': model_name,
            'train_size': len(X_train),
            'test_size': len(X_test),
            'features': list(X_train.columns),
            **params
        }
        tracker.log_params(log_params)
        
        # Log metrics
        all_metrics = {}
        for metric, value in train_metrics.items():
            all_metrics[f'train_{metric}'] = value
        for metric, value in test_metrics.items():
            all_metrics[f'test_{metric}'] = value
        
        tracker.log_metrics(all_metrics)
        
        # Log visualizations
        class_names = ['setosa', 'versicolor', 'virginica']
        tracker.log_confusion_matrix(y_test, y_test_pred, class_names)
        tracker.log_classification_report(y_test, y_test_pred, class_names)
        
        # Log feature importance for applicable models
        if hasattr(model, 'feature_importances_'):
            tracker.log_feature_importance(model, X_train.columns.tolist())
        
        # Log model
        tracker.log_model(model, "model")
        
        print(f"Experiment completed successfully!")
        print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"Test F1 (weighted): {test_metrics['f1_weighted']:.4f}")
        
        run_id = mlflow.active_run().info.run_id
        return run_id
        
    finally:
        tracker.end_run()

def run_all_models(experiment_name: str = "iris_classification_comparison",
                  tracking_uri: str = None) -> Dict[str, str]:
    """
    Run experiments for all available models.
    
    Args:
        experiment_name: MLflow experiment name
        tracking_uri: MLflow tracking server URI
        
    Returns:
        Dictionary mapping model names to run IDs
    """
    models = ['logistic_regression', 'random_forest', 'svm', 'naive_bayes', 'decision_tree']
    run_ids = {}
    
    print(f"Running comparison experiment with {len(models)} models...")
    
    for model_name in models:
        print(f"\n{'='*50}")
        print(f"Training {model_name.replace('_', ' ').title()}")
        print(f"{'='*50}")
        
        try:
            run_id = run_experiment(model_name, experiment_name, tracking_uri=tracking_uri)
            run_ids[model_name] = run_id
            print(f"✅ {model_name} completed successfully")
        except Exception as e:
            print(f"❌ {model_name} failed: {str(e)}")
            run_ids[model_name] = None
    
    print(f"\n{'='*50}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*50}")
    for model_name, run_id in run_ids.items():
        status = "✅ SUCCESS" if run_id else "❌ FAILED"
        print(f"{model_name.replace('_', ' ').title()}: {status}")
    
    return run_ids

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ML models with MLflow tracking")
    parser.add_argument("--model", type=str, default="all",
                       help="Model to train (logistic_regression, random_forest, svm, naive_bayes, decision_tree, or 'all')")
    parser.add_argument("--experiment", type=str, default="iris_classification",
                       help="MLflow experiment name")
    parser.add_argument("--tracking-uri", type=str, default=None,
                       help="MLflow tracking server URI")
    
    args = parser.parse_args()
    
    if args.model == "all":
        run_ids = run_all_models(args.experiment, args.tracking_uri)
    else:
        run_id = run_experiment(args.model, args.experiment, tracking_uri=args.tracking_uri)
        print(f"Run ID: {run_id}")