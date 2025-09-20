"""
Simple data preprocessing functions.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple


def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.3, 
               random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split data into train and test sets."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Scale features using StandardScaler."""
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), 
        columns=X_train.columns, 
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), 
        columns=X_test.columns, 
        index=X_test.index
    )
    return X_train_scaled, X_test_scaled


def preprocess_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.3, 
                   scale: bool = True, random_state: int = 42):
    """Complete preprocessing pipeline."""
    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y, test_size, random_state)
    
    # Scale if requested
    if scale:
        X_train, X_test = scale_features(X_train, X_test)
    
    print(f"Data split: {X_train.shape[0]} train, {X_test.shape[0]} test samples")
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    from data_loader import load_iris
    X, y = load_iris()
    X_train, X_test, y_train, y_test = preprocess_data(X, y)
    print(f"Preprocessing completed: {X_train.shape}, {X_test.shape}")