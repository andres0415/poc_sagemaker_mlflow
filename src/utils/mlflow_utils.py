"""
MLflow utilities for experiment tracking and model registry.
"""
import mlflow
import mlflow.sklearn
import mlflow.metrics
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional
import os
import tempfile

class MLflowTracker:
    """
    Utility class for MLflow experiment tracking.
    """
    
    def __init__(self, experiment_name: str = "iris_classification", tracking_uri: Optional[str] = None):
        """
        Initialize MLflow tracker.
        
        Args:
            experiment_name: Name of the MLflow experiment
            tracking_uri: MLflow tracking server URI (defaults to local)
        """
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        
        mlflow.set_experiment(experiment_name)
        self.experiment_name = experiment_name
    
    def start_run(self, run_name: Optional[str] = None) -> None:
        """Start a new MLflow run."""
        mlflow.start_run(run_name=run_name)
    
    def end_run(self) -> None:
        """End the current MLflow run."""
        mlflow.end_run()
    
    def log_params(self, params: Dict[str, Any]) -> None:
        """Log parameters to MLflow."""
        mlflow.log_params(params)
    
    def log_metrics(self, metrics: Dict[str, float]) -> None:
        """Log metrics to MLflow."""
        mlflow.log_metrics(metrics)
    
    def log_model(self, model, artifact_path: str = "model", 
                  registered_model_name: Optional[str] = None) -> None:
        """
        Log model to MLflow.
        
        Args:
            model: Trained model
            artifact_path: Path within the run to store the model
            registered_model_name: Name for model registry
        """
        mlflow.sklearn.log_model(
            model, 
            artifact_path=artifact_path,
            registered_model_name=registered_model_name
        )
    
    def calculate_metrics(self, y_true, y_pred) -> Dict[str, float]:
        """
        Calculate classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Dictionary of metrics
        """
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, average='macro'),
            'recall_macro': recall_score(y_true, y_pred, average='macro'),
            'f1_macro': f1_score(y_true, y_pred, average='macro'),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted'),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted'),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted')
        }
    
    def log_confusion_matrix(self, y_true, y_pred, class_names=None) -> None:
        """
        Create and log confusion matrix plot.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: List of class names for labeling
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names or range(len(cm)),
                   yticklabels=class_names or range(len(cm)))
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Save plot to temporary file and log it
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            plt.savefig(f.name, dpi=150, bbox_inches='tight')
            mlflow.log_artifact(f.name, "plots")
            os.unlink(f.name)
        
        plt.close()
    
    def log_classification_report(self, y_true, y_pred, class_names=None) -> None:
        """
        Log detailed classification report.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: List of class names
        """
        report = classification_report(
            y_true, y_pred, 
            target_names=class_names,
            output_dict=True
        )
        
        # Log per-class metrics
        for class_name, metrics in report.items():
            if isinstance(metrics, dict):
                for metric_name, value in metrics.items():
                    mlflow.log_metric(f"{class_name}_{metric_name}", value)
    
    def log_feature_importance(self, model, feature_names) -> None:
        """
        Log feature importance plot for tree-based models.
        
        Args:
            model: Trained model with feature_importances_ attribute
            feature_names: List of feature names
        """
        if not hasattr(model, 'feature_importances_'):
            return
        
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.title("Feature Importances")
        plt.bar(range(len(importances)), importances[indices])
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
        plt.tight_layout()
        
        # Save and log plot
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            plt.savefig(f.name, dpi=150, bbox_inches='tight')
            mlflow.log_artifact(f.name, "plots")
            os.unlink(f.name)
        
        plt.close()
        
        # Log individual feature importances as metrics
        for i, importance in enumerate(importances):
            # Clean feature name for MLflow metric naming
            clean_name = feature_names[i].replace('(', '').replace(')', '').replace(' ', '_')
            mlflow.log_metric(f"feature_importance_{clean_name}", importance)

def register_best_model(experiment_name: str, metric_name: str = "accuracy", 
                       model_name: str = "iris_best_model") -> str:
    """
    Find the best run and register the model.
    
    Args:
        experiment_name: Name of the experiment
        metric_name: Metric to use for comparison
        model_name: Name for the registered model
        
    Returns:
        Model version URI
    """
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if not experiment:
        raise ValueError(f"Experiment '{experiment_name}' not found")
    
    # Find the best run
    best_run = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric_name} DESC"],
        max_results=1
    ).iloc[0]
    
    # Register the model
    model_uri = f"runs:/{best_run.run_id}/model"
    model_version = mlflow.register_model(model_uri, model_name)
    
    print(f"Registered model '{model_name}' version {model_version.version}")
    print(f"Best run ID: {best_run.run_id}")
    print(f"Best {metric_name}: {best_run[f'metrics.{metric_name}']}")
    
    return model_uri

if __name__ == "__main__":
    # Demo usage
    tracker = MLflowTracker("demo_experiment")
    
    # Example of logging a simple run
    with mlflow.start_run(run_name="demo_run"):
        mlflow.log_params({"algorithm": "demo", "param1": 1.0})
        mlflow.log_metrics({"accuracy": 0.95, "f1": 0.94})
        print("Demo run logged successfully!")