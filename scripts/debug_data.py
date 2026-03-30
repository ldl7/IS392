"""Debug the data loading and feature construction to identify NaN issues."""

import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import os
import glob

# Configuration
SHARD_FOLDER = "./exploring_data"
INTERIM_OUTPUT = "./data/interim/filtered_physical_deliverables.parquet"
FINAL_OUTPUT = "./data/processed/labeled_contracts.parquet"

# Check if processed data exists
if os.path.exists(FINAL_OUTPUT):
    print("Loading processed data...")
    labeled_df = pd.read_parquet(FINAL_OUTPUT)
    print(f"Labeled data shape: {labeled_df.shape}")
    
    # Check structured features
    structured_features = [
        'base_value', 'final_value', 'modifications', 'num_offers'
    ]
    
    structured_df = labeled_df[structured_features].copy()
    print(f"Structured features shape: {structured_df.shape}")
    
    # Check for NaN values
    print("\nNaN values in structured features:")
    for col in structured_df.columns:
        nan_count = structured_df[col].isna().sum()
        print(f"  {col}: {nan_count} NaN values ({nan_count/len(structured_df)*100:.2f}%)")
        
    # Check data types
    print("\nData types:")
    print(structured_df.dtypes)
    
    # Show sample values
    print("\nSample values:")
    print(structured_df.head())
    
else:
    print("Processed data not found. Need to run the full pipeline first.")
