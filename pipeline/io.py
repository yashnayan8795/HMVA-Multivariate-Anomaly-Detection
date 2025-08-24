from __future__ import annotations
import pandas as pd
import numpy as np
from typing import List

def add_output_columns(
    df_original: pd.DataFrame,
    scores: pd.Series,
    top_features: list[list[str]]
) -> pd.DataFrame:
    """
    Add the 8 required output columns to the DataFrame.
    
    Args:
        df_original: Original DataFrame with all columns
        scores: Series containing Abnormality_score values
        top_features: List of lists, each containing up to 7 feature names
    
    Returns:
        DataFrame with 8 new columns added
    """
    out = df_original.copy()
    
    # Add Abnormality_score column (0-100 scale)
    out['Abnormality_score'] = scores.astype(float).round(3)
    
    # Add top_feature_1 through top_feature_7 columns
    # Handle cases where dataset has fewer than 7 features
    max_features = min(7, max(len(row) for row in top_features) if top_features else 0)
    
    for k in range(7):
        col = f'top_feature_{k+1}'
        if k < max_features:
            # Extract feature names, pad with empty string if missing
            vals = [row[k] if k < len(row) else '' for row in top_features]
        else:
            # Fill remaining columns with empty strings
            vals = [''] * len(top_features)
        
        out[col] = vals
    
    return out
