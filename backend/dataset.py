import pandas as pd
import numpy as np
import os

def build_training_sets(path=None, target_type='classification'):
    """
    Loads data from Excel or generates synthetic data if file not found.
    Args:
        path: Path to dataset Excel file
        target_type: 'regression' for Target_Quality, 'classification' for Quality_Label
    Returns: X (DataFrame), y (Series)
    """
    if path is None:
        # Set the default path pointing to our main dataset file relative to this script's location
        path = os.path.join(os.path.dirname(__file__), '../../Explainable_AI_Dataset_10000.xlsx')
        path = os.path.abspath(path)

    # Define the core sensor metrics we expect to find in the dataset
    feature_cols = [
        'Temperature', 'Pressure', 'Speed', 'Vibration', 
        'Humidity', 'Power_Consumption', 'Material_Hardness'
    ]
    target_col = 'Target_Quality' if target_type == 'regression' else 'Quality_Label'

    if os.path.exists(path):
        print(f" Loading dataset from {path}...")
        try:
            df = pd.read_excel(path)
            
            # Verify that our dataset contains all the essential sensor columns before proceeding
            missing = [col for col in feature_cols + [target_col] if col not in df.columns]
            if not missing:
                X = df[feature_cols]
                y = df[target_col]
                print(f"Dataset shape: {X.shape}, Balance: {y.mean():.2%} failures")
                return X, y
            else:
                print(f"Warning: Missing columns in dataset: {missing}. Falling back to synthetic.")
        except Exception as e:
            print(f"Error reading Excel file: {e}. Falling back to synthetic.")
    else:
        print(f"Dataset file not found at {path}. Generating synthetic data (fallback)...")
    
    # Synthetic Generation
    n_samples = 1000
    np.random.seed(42)
    
    X = pd.DataFrame({
        'Temperature': np.random.normal(75, 10, n_samples),
        'Pressure': np.random.normal(30, 5, n_samples),
        'Speed': np.random.uniform(500, 3000, n_samples),
        'Vibration': np.random.exponential(2, n_samples),
        'Humidity': np.random.uniform(20, 80, n_samples),
        'Power_Consumption': np.random.normal(150, 20, n_samples),
        'Material_Hardness': np.random.normal(60, 10, n_samples)
    })
    
    # Simulated risk logic: Combining extreme vibration and high temperature spikes the probability of failure
    prob = (
        (X['Temperature'] > 90).astype(int) * 0.5 + 
        (X['Vibration'] > 4).astype(int) * 0.6 + 
        (X['Power_Consumption'] > 180).astype(int) * 0.4
    )
    prob += np.random.normal(0, 0.1, n_samples)
    y = (prob > 0.5).astype(int)
    y.name = target_col
    
    print(f"Dataset shape: {X.shape}, Balance: {y.mean():.2%} failures")
    return X, y

if __name__ == "__main__":
    X, y = build_training_sets()
    print(X.head())
