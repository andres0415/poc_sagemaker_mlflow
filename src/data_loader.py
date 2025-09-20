"""
Simple data loader for basic ML datasets.
"""

import pandas as pd
from sklearn.datasets import load_iris, load_wine
from typing import Tuple


def load_iris() -> Tuple[pd.DataFrame, pd.Series]:
    """Load Iris dataset as pandas DataFrame and Series."""
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target, name='species')
    return X, y


def load_wine() -> Tuple[pd.DataFrame, pd.Series]:
    """Load Wine dataset as pandas DataFrame and Series."""
    wine = load_wine()
    X = pd.DataFrame(wine.data, columns=wine.feature_names)
    y = pd.Series(wine.target, name='wine_class')
    return X, y


if __name__ == "__main__":
    # Test the functions
    print("Loading Iris dataset...")
    X_iris, y_iris = load_iris()
    print(f"Iris shape: {X_iris.shape}, Classes: {y_iris.unique()}")
    
    print("\nLoading Wine dataset...")
    X_wine, y_wine = load_wine()
    print(f"Wine shape: {X_wine.shape}, Classes: {y_wine.unique()}")