"""
Data loading and preprocessing utilities for the ML pipeline.
"""
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict
import os
import joblib

def load_iris_dataset() -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load the iris dataset and return as pandas DataFrame and Series.
    
    Returns:
        Tuple of (features_df, target_series)
    """
    iris = load_iris()
    
    # Create DataFrame with feature names
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target, name='target')
    
    # Map target numbers to class names for better interpretability
    target_names = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
    y = y.map(target_names)
    
    return X, y

def preprocess_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, 
                   random_state: int = 42, scale_features: bool = True) -> Dict:
    """
    Preprocess the data by splitting and optionally scaling.
    
    Args:
        X: Feature DataFrame
        y: Target Series
        test_size: Proportion of dataset to include in test split
        random_state: Random state for reproducibility
        scale_features: Whether to scale features
        
    Returns:
        Dictionary containing train/test splits and scaler (if used)
    """
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    result = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'scaler': None
    }
    
    if scale_features:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Convert back to DataFrame to preserve column names
        result['X_train'] = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
        result['X_test'] = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
        result['scaler'] = scaler
    
    return result

def save_preprocessed_data(data_dict: Dict, output_dir: str = 'data/processed') -> None:
    """
    Save preprocessed data to disk.
    
    Args:
        data_dict: Dictionary containing preprocessed data
        output_dir: Directory to save the data
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save train/test splits
    data_dict['X_train'].to_csv(f'{output_dir}/X_train.csv', index=False)
    data_dict['X_test'].to_csv(f'{output_dir}/X_test.csv', index=False)
    data_dict['y_train'].to_csv(f'{output_dir}/y_train.csv', index=False)
    data_dict['y_test'].to_csv(f'{output_dir}/y_test.csv', index=False)
    
    # Save scaler if it exists
    if data_dict['scaler'] is not None:
        joblib.dump(data_dict['scaler'], f'{output_dir}/scaler.pkl')

def load_preprocessed_data(data_dir: str = 'data/processed') -> Dict:
    """
    Load preprocessed data from disk.
    
    Args:
        data_dir: Directory containing the preprocessed data
        
    Returns:
        Dictionary containing the loaded data
    """
    result = {
        'X_train': pd.read_csv(f'{data_dir}/X_train.csv'),
        'X_test': pd.read_csv(f'{data_dir}/X_test.csv'),
        'y_train': pd.read_csv(f'{data_dir}/y_train.csv')['target'],
        'y_test': pd.read_csv(f'{data_dir}/y_test.csv')['target'],
        'scaler': None
    }
    
    # Load scaler if it exists
    scaler_path = f'{data_dir}/scaler.pkl'
    if os.path.exists(scaler_path):
        result['scaler'] = joblib.load(scaler_path)
    
    return result

if __name__ == "__main__":
    # Demo: Load and preprocess data
    print("Loading iris dataset...")
    X, y = load_iris_dataset()
    print(f"Dataset shape: {X.shape}")
    print(f"Features: {list(X.columns)}")
    print(f"Target classes: {y.unique()}")
    
    print("\nPreprocessing data...")
    data = preprocess_data(X, y)
    print(f"Training set shape: {data['X_train'].shape}")
    print(f"Test set shape: {data['X_test'].shape}")
    
    print("\nSaving preprocessed data...")
    save_preprocessed_data(data)
    print("Data saved successfully!")