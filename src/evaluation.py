"""
Simple model evaluation functions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series, model_name: str = "Model"):
    """Evaluate a single model and print results."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n{model_name} Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return accuracy, y_pred


def plot_confusion_matrix(y_test: pd.Series, y_pred: np.ndarray, model_name: str = "Model"):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()


def compare_model_accuracies(results: dict):
    """Plot comparison of model accuracies."""
    models = list(results.keys())
    accuracies = [results[model]['accuracy'] for model in models if model != 'best_model']
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, accuracies)
    plt.title('Model Accuracy Comparison')
    plt.xlabel('Models')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{acc:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("Evaluation module ready to use!")