"""
Model evaluation and comparison utilities.
"""
import mlflow
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.mlflow_utils import register_best_model

class ExperimentAnalyzer:
    """
    Utility class for analyzing and comparing MLflow experiments.
    """
    
    def __init__(self, experiment_name: str, tracking_uri: Optional[str] = None):
        """
        Initialize the analyzer.
        
        Args:
            experiment_name: Name of the MLflow experiment
            tracking_uri: MLflow tracking server URI
        """
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        
        self.experiment_name = experiment_name
        self.experiment = mlflow.get_experiment_by_name(experiment_name)
        
        if not self.experiment:
            raise ValueError(f"Experiment '{experiment_name}' not found")
    
    def get_experiment_results(self) -> pd.DataFrame:
        """
        Get all runs from the experiment as a DataFrame.
        
        Returns:
            DataFrame with run information and metrics
        """
        runs = mlflow.search_runs(
            experiment_ids=[self.experiment.experiment_id],
            order_by=["start_time DESC"]
        )
        
        return runs
    
    def compare_models(self, metrics: List[str] = None) -> pd.DataFrame:
        """
        Compare models across specified metrics.
        
        Args:
            metrics: List of metrics to compare (defaults to common classification metrics)
            
        Returns:
            DataFrame with model comparison
        """
        if metrics is None:
            metrics = ['test_accuracy', 'test_f1_weighted', 'test_precision_weighted', 'test_recall_weighted']
        
        runs = self.get_experiment_results()
        
        # Filter to successful runs only
        successful_runs = runs[runs['status'] == 'FINISHED'].copy()
        
        if successful_runs.empty:
            print("No successful runs found in the experiment")
            return pd.DataFrame()
        
        # Create comparison dataframe
        comparison_data = []
        
        for _, run in successful_runs.iterrows():
            model_data = {
                'run_id': run['run_id'],
                'model_name': run.get('params.model_name', 'unknown'),
                'run_name': run.get('tags.mlflow.runName', 'unnamed')
            }
            
            # Add metrics
            for metric in metrics:
                model_data[metric] = run.get(f'metrics.{metric}', np.nan)
            
            comparison_data.append(model_data)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Sort by primary metric (first in the list)
        if metrics and not comparison_df.empty:
            primary_metric = metrics[0]
            comparison_df = comparison_df.sort_values(primary_metric, ascending=False)
        
        return comparison_df
    
    def plot_model_comparison(self, metrics: List[str] = None, figsize: tuple = (12, 8)) -> None:
        """
        Create comparison plots for different models.
        
        Args:
            metrics: List of metrics to plot
            figsize: Figure size for the plot
        """
        if metrics is None:
            metrics = ['test_accuracy', 'test_f1_weighted', 'test_precision_weighted', 'test_recall_weighted']
        
        comparison_df = self.compare_models(metrics)
        
        if comparison_df.empty:
            print("No data to plot")
            return
        
        # Create subplots
        n_metrics = len(metrics)
        n_cols = 2 if n_metrics > 2 else n_metrics
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes
        else:
            axes = axes.flatten()
        
        # Plot each metric
        for i, metric in enumerate(metrics):
            ax = axes[i] if i < len(axes) else None
            if ax is None:
                break
                
            # Create bar plot
            valid_data = comparison_df.dropna(subset=[metric])
            if not valid_data.empty:
                bars = ax.bar(valid_data['model_name'], valid_data[metric])
                ax.set_title(f'{metric.replace("_", " ").title()}')
                ax.set_ylabel('Score')
                ax.tick_params(axis='x', rotation=45)
                
                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.3f}',
                           ha='center', va='bottom')
        
        # Hide empty subplots
        for i in range(len(metrics), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    def get_best_model(self, metric: str = 'test_accuracy') -> Dict:
        """
        Get information about the best performing model.
        
        Args:
            metric: Metric to use for comparison
            
        Returns:
            Dictionary with best model information
        """
        runs = self.get_experiment_results()
        successful_runs = runs[runs['status'] == 'FINISHED'].copy()
        
        if successful_runs.empty:
            raise ValueError("No successful runs found")
        
        # Find best run
        metric_col = f'metrics.{metric}'
        if metric_col not in successful_runs.columns:
            available_metrics = [col for col in successful_runs.columns if col.startswith('metrics.')]
            raise ValueError(f"Metric '{metric}' not found. Available metrics: {available_metrics}")
        
        best_run = successful_runs.loc[successful_runs[metric_col].idxmax()]
        
        return {
            'run_id': best_run['run_id'],
            'model_name': best_run.get('params.model_name', 'unknown'),
            'run_name': best_run.get('tags.mlflow.runName', 'unnamed'),
            f'best_{metric}': best_run[metric_col],
            'start_time': best_run['start_time'],
            'experiment_id': best_run['experiment_id']
        }
    
    def print_summary(self) -> None:
        """Print a summary of the experiment."""
        runs = self.get_experiment_results()
        
        print(f"\n{'='*60}")
        print(f"EXPERIMENT SUMMARY: {self.experiment_name}")
        print(f"{'='*60}")
        
        print(f"Total runs: {len(runs)}")
        print(f"Successful runs: {len(runs[runs['status'] == 'FINISHED'])}")
        print(f"Failed runs: {len(runs[runs['status'] == 'FAILED'])}")
        
        # Show model comparison
        comparison_df = self.compare_models()
        if not comparison_df.empty:
            print(f"\n{'MODEL COMPARISON':^60}")
            print("-" * 60)
            print(comparison_df.to_string(index=False, float_format='%.4f'))
            
            # Show best model
            best_model = self.get_best_model()
            print(f"\n{'BEST MODEL':^60}")
            print("-" * 60)
            print(f"Model: {best_model['model_name']}")
            print(f"Run ID: {best_model['run_id']}")
            print(f"Accuracy: {best_model['best_test_accuracy']:.4f}")

def register_champion_model(experiment_name: str, model_name: str = "iris_champion_model",
                          metric: str = "test_accuracy", tracking_uri: Optional[str] = None) -> str:
    """
    Register the best model from an experiment as the champion model.
    
    Args:
        experiment_name: Name of the experiment
        model_name: Name for the registered model
        metric: Metric to use for selecting the best model
        tracking_uri: MLflow tracking server URI
        
    Returns:
        Model version URI
    """
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    
    analyzer = ExperimentAnalyzer(experiment_name, tracking_uri)
    best_model = analyzer.get_best_model(metric)
    
    print(f"Registering best model...")
    print(f"Best model: {best_model['model_name']}")
    print(f"Best {metric}: {best_model[f'best_{metric}']:.4f}")
    print(f"Run ID: {best_model['run_id']}")
    
    # Register the model
    model_uri = f"runs:/{best_model['run_id']}/model"
    model_version = mlflow.register_model(model_uri, model_name)
    
    # Add simple model description without complex object serialization
    client = mlflow.tracking.MlflowClient()
    try:
        client.update_model_version(
            name=model_name,
            version=model_version.version,
            description=f"Champion model from experiment '{experiment_name}'. "
                       f"Algorithm: {best_model['model_name']}, "
                       f"{metric}: {best_model[f'best_{metric}']:.4f}"
        )
    except Exception as e:
        print(f"Warning: Could not update model description: {e}")
        print("Model registration completed without description update.")
    
    print(f"✅ Model registered successfully!")
    print(f"Model name: {model_name}")
    print(f"Version: {model_version.version}")
    
    return model_uri

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze and compare MLflow experiments")
    parser.add_argument("--experiment", type=str, required=True,
                       help="MLflow experiment name")
    parser.add_argument("--tracking-uri", type=str, default=None,
                       help="MLflow tracking server URI")
    parser.add_argument("--register-best", action="store_true",
                       help="Register the best model")
    parser.add_argument("--model-name", type=str, default="iris_champion_model",
                       help="Name for the registered model")
    
    args = parser.parse_args()
    
    # Analyze experiment
    analyzer = ExperimentAnalyzer(args.experiment, args.tracking_uri)
    analyzer.print_summary()
    analyzer.plot_model_comparison()
    
    # Register best model if requested
    if args.register_best:
        register_champion_model(
            args.experiment,
            args.model_name,
            tracking_uri=args.tracking_uri
        )