"""
Data cleaning module for crypto regime detection
Handles missing values, duplicates, and creates basic features
"""

import pandas as pd
import numpy as np
from pathlib import Path

def clean_btc_data(df, vol_window=5):
    """
    Clean BTC data and create basic return/volatility features
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw data with columns: date, open, high, low, close, adj_close, volume
    vol_window : int
        Window for rolling volatility calculation
        
    Returns:
    --------
    pd.DataFrame
        Cleaned dataframe with additional features
    """
    
    print("Cleaning data...")
    df = df.copy()
    
    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Sort by date
    df = df.sort_values('date').reset_index(drop=True)
    
    # Remove duplicates
    initial_rows = len(df)
    df = df.drop_duplicates(subset=['date'], keep='last')
    if len(df) < initial_rows:
        print(f"  Removed {initial_rows - len(df)} duplicate rows")
    
    # Forward fill missing values
    missing_before = df.isnull().sum().sum()
    df = df.fillna(method='ffill')
    if missing_before > 0:
        print(f"  Forward-filled {missing_before} missing values")
    
    # Create basic features
    # Simple return
    df['return'] = df['close'].pct_change()
    
    # Log return
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    
    # Rolling volatility (standard deviation of log returns)
    df[f'volatility_{vol_window}d'] = df['log_return'].rolling(window=vol_window).std()
    
    # Remove rows with NaN created by rolling calculations
    initial_len = len(df)
    df = df.dropna().reset_index(drop=True)
    print(f"  Created return and volatility features")
    print(f"  Removed {initial_len - len(df)} rows with NaN from rolling calculations")
    
    # Data quality check
    if df['close'].min() <= 0:
        raise ValueError("Found non-positive close prices")
    
    if (df['volume'] < 0).any():
        raise ValueError("Found negative volume values")
    
    print(f"✓ Cleaned data: {len(df)} rows, {df.columns.size} columns")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    
    return df

def load_and_clean(raw_path='data/raw/btc_raw.csv', vol_window=5):
    """
    Load raw data and clean it
    
    Parameters:
    -----------
    raw_path : str
        Path to raw CSV file
    vol_window : int
        Window for volatility calculation
        
    Returns:
    --------
    pd.DataFrame
        Cleaned dataframe
    """
    
    print(f"Loading data from {raw_path}...")
    df = pd.read_csv(raw_path)
    return clean_btc_data(df, vol_window)

if __name__ == "__main__":
    df_clean = load_and_clean()
    print("\\nCleaned data sample:")
    print(df_clean.head())
    print("\\nData info:")
    print(df_clean.info())
    print("\\nBasic statistics:")
    print(df_clean[['close', 'return', 'log_return', 'volatility_5d']].describe())