"""
Simple machine learning models for basic classification.
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from typing import Dict, Any


def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series, 
                       n_estimators: int = 100, random_state: int = 42):
    """Train a Random Forest classifier."""
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)
    return model


def train_svm(X_train: pd.DataFrame, y_train: pd.Series, 
              C: float = 1.0, random_state: int = 42):
    """Train an SVM classifier."""
    model = SVC(C=C, random_state=random_state)
    model.fit(X_train, y_train)
    return model


def train_logistic_regression(X_train: pd.DataFrame, y_train: pd.Series, 
                             random_state: int = 42):
    """Train a Logistic Regression classifier."""
    model = LogisticRegression(random_state=random_state, max_iter=1000)
    model.fit(X_train, y_train)
    return model


def compare_models(X_train: pd.DataFrame, y_train: pd.Series,
                  X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
    """Train and compare three basic models."""
    
    models = {
        'Random Forest': train_random_forest(X_train, y_train),
        'SVM': train_svm(X_train, y_train),
        'Logistic Regression': train_logistic_regression(X_train, y_train)
    }
    
    results = {}
    
    print("Model Comparison Results:")
    print("=" * 40)
    
    for name, model in models.items():
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        
        # Store results
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'predictions': y_pred
        }
        
        print(f"{name}: {accuracy:.4f}")
    
    # Find best model
    best_model_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
    results['best_model'] = best_model_name
    
    print(f"\nBest Model: {best_model_name}")
    
    return results


if __name__ == "__main__":
    from data_loader import load_iris
    from preprocessing import preprocess_data
    
    # Load and preprocess data
    X, y = load_iris()
    X_train, X_test, y_train, y_test = preprocess_data(X, y)
    
    # Compare models
    results = compare_models(X_train, y_train, X_test, y_test)