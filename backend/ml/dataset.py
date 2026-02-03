import pandas as pd
import numpy as np
import os

def build_training_sets(path=None):
    """
    Loads data from Excel or generates synthetic data if file not found.
    Returns: X (DataFrame), y (Series)
    """
    if path is None:
        # Default path relative to this file
        path = os.path.join(os.path.dirname(__file__), '../../Explainable_AI_Dataset_10000.xlsx')
        path = os.path.abspath(path)

    # Expected features
    feature_cols = [
        'Temperature', 'Pressure', 'Speed', 'Vibration', 
        'Humidity', 'Power_Consumption', 'Material_Hardness'
    ]
    target_col = 'Quality_Label'

    if os.path.exists(path):
        print(f" Loading dataset from {path}...")
        try:
            if path.endswith('.csv'):
                df = pd.read_csv(path)
            else:
                df = pd.read_excel(path)
            
            # Check for missing columns
            missing = [col for col in feature_cols + [target_col] if col not in df.columns]
            if not missing:
                X = df[feature_cols]
                y = df[target_col]
                print(f"Dataset shape: {X.shape}, Balance: {y.mean():.2%} failures")
                return X, y
            else:
                print(f"Warning: Missing columns in dataset: {missing}. Falling back to synthetic.")
        except Exception as e:
            print(f"Error reading file: {e}. Falling back to synthetic.")
    else:
        print(f"Dataset file not found at {path}. Using user-provided generation logic...")
        df = generate_manufacturing_data()
        
        # Verify columns exist
        missing = [col for col in feature_cols if col not in df.columns]
        if missing:
             print(f"Warning: Generated data missing columns: {missing}")

        X = df[feature_cols]
        y = df[target_col]
        
        print(f"Dataset shape: {X.shape}, Balance: {y.mean():.2%} failures")
        return X, y

if __name__ == "__main__":
    X, y = build_training_sets()
    print(X.head())

def generate_manufacturing_data(n_samples=10000):
   # Set seed for reproducibility
   np.random.seed(42)
   
   # 1. Generating Raw Features
   data = {
       'Temperature': np.random.normal(70, 15, n_samples),
       'Pressure': np.random.normal(50, 10, n_samples),
       'Speed': np.random.normal(1500, 300, n_samples),
       'Vibration': np.random.normal(5, 2, n_samples),
       'Humidity': np.random.uniform(30, 90, n_samples),
       'Power_Consumption': np.random.normal(15, 3, n_samples),
       'Material_Hardness': np.random.uniform(20, 100, n_samples),
       'Usage_Hours': np.random.uniform(10, 5000, n_samples),
       'Operator_Skill': np.random.choice([0, 1, 2], n_samples)
   }
   
   df = pd.DataFrame(data)
   
   # 2. Physics-Based Logic
   quality_score = (
       100 
       - (df['Temperature'] * 0.1) 
       - (df['Vibration'] * df['Speed'] * 0.002) 
       - (df['Pressure'] * 0.05)
       + (df['Operator_Skill'] * 5)
   )
   
   quality_score += np.random.normal(0, 2, n_samples)
   
   # 3. Setting the Target Labels
   df['Target_Quality'] = quality_score.round(2)
   threshold = np.percentile(df['Target_Quality'], 20)
   df['Quality_Label'] = (df['Target_Quality'] < threshold).astype(int) # Renamed to match earlier code expectation if needed, or stick to 'Failure'
   
   # 4. Save to CSV
   # Save to the parent directory or current? The original code looked for '../../Explainable_AI_Dataset_10000.xlsx'
   # We will save to a standard location: backend/data/
   os.makedirs(os.path.join(os.path.dirname(__file__), '..', 'data'), exist_ok=True)
   output_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'manufacturing_data.csv')
   df.to_csv(output_path, index=False)
   
   print(f"Dataset generated at {output_path}")
   return df

if __name__ == "__main__":
    generate_manufacturing_data()