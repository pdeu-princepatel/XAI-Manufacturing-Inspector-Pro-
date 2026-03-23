
import pandas as pd
import os

try:
    path = r'd:\Aayush\College\Materials\Sem 6\Project\xai_manufacturing\Explainable_AI_Dataset_10000.xlsx'
    if os.path.exists(path):
        df = pd.read_excel(path)
        print("Columns found:", df.columns.tolist())
        print("First few rows:")
        print(df.head())
    else:
        print(f"File not found: {path}")
except Exception as e:
    print(f"Error: {e}")
