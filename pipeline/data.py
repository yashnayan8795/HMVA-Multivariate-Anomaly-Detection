from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Tuple, List

def load_csv(path: str, time_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if time_col not in df.columns:
        # try case-insensitive match
        matches = [c for c in df.columns if c.lower() == time_col.lower()]
        if matches:
            time_col = matches[0]
        else:
            raise ValueError(f"Timestamp column '{time_col}' not found in CSV.")
    ts = pd.to_datetime(df[time_col], errors='coerce')

    if ts.isna().any():
        bad = ts.isna().sum()
        raise ValueError(f"{bad} bad timestamps in '{time_col}'. Please fix or choose correct column.")
    df = df.set_index(ts).drop(columns=[time_col])
    # Keep only numeric
    num_df = df.select_dtypes(include=[np.number]).copy()
    non_num = [c for c in df.columns if c not in num_df.columns]
    if non_num:
        # Fill non-numeric with last valid value if needed, but we drop them for scoring
        pass
    return num_df.sort_index()

def validate_regular_intervals(df: pd.DataFrame) -> None:
    # Ensure mostly regular intervals; warn if grossly irregular
    diffs = df.index.to_series().diff().dropna()
    if diffs.empty:
        return
    top = diffs.mode()
    if not top.empty:
        ref = top.iloc[0]
        off_ratio = (diffs != ref).mean()
        if off_ratio > 0.1:
            print(f"[WARN] >10% timestamp gaps deviate from modal spacing {ref}. Proceeding.")
    else:
        print("[WARN] Could not infer regular interval; proceeding.")

def impute_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    # Forward fill then linear interpolate
    df = df.copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.ffill().interpolate(method='time', limit_direction='both')
    # Drop constant (zero-variance) features
    nunique = df.nunique(dropna=True)
    keep_cols = [c for c in df.columns if nunique[c] > 1]
    dropped = [c for c in df.columns if c not in keep_cols]
    if dropped:
        print(f"[INFO] Dropping constant features: {dropped}")
    return df[keep_cols]

def split_train_analysis(
    df: pd.DataFrame,
    train_start: str,
    train_end: str,
    analysis_end: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_start_ts = pd.to_datetime(train_start)
    train_end_ts = pd.to_datetime(train_end)
    analysis_end_ts = pd.to_datetime(analysis_end)
    if train_start_ts >= train_end_ts:
        raise ValueError("train_start must be before train_end")
    if analysis_end_ts <= train_end_ts:
        raise ValueError("analysis_end must be after train_end")
    # filter
    train = df.loc[(df.index >= train_start_ts) & (df.index <= train_end_ts)]
    analysis = df.loc[(df.index >= train_start_ts) & (df.index <= analysis_end_ts)]
    if train.shape[0] < 72:  # at least 72 timepoints (assuming hourly ≈ 3 days)
        raise ValueError("Insufficient training data (<72 rows).")
    return train, analysis
