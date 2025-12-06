"""
Data loading module for crypto regime detection
Downloads BTC-USD data from Yahoo Finance with retry logic
"""

import pandas as pd
import yfinance as yf
from time import sleep
from pathlib import Path

def load_raw_btc(start="2014-01-01", end=None, interval="1d", save=True, max_retries=3):
    """
    Download BTC-USD OHLCV data from Yahoo Finance
    
    Parameters:
    -----------
    start : str
        Start date (YYYY-MM-DD)
    end : str or None
        End date (YYYY-MM-DD), None for today
    interval : str
        Data interval (1d, 1h, etc.)
    save : bool
        Whether to save to CSV
    max_retries : int
        Maximum number of retry attempts
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with columns: date, open, high, low, close, adj close, volume
    """
    
    for attempt in range(max_retries):
        try:
            print(f"Downloading BTC-USD data (attempt {attempt + 1}/{max_retries})...")
            
            # Download data using yfinance
            ticker = yf.Ticker("BTC-USD")
            df = ticker.history(start=start, end=end, interval=interval)
            
            if df.empty:
                raise ValueError("Downloaded data is empty")
            
            # Reset index to get date as column
            df = df.reset_index()
            
            # Normalize column names
            df.columns = df.columns.str.lower().str.replace(' ', '_')
            
            # Rename 'date' or 'datetime' column
            if 'datetime' in df.columns:
                df.rename(columns={'datetime': 'date'}, inplace=True)
            
            # Ensure we have required columns
            required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
            missing = [col for col in required_cols if col not in df.columns]
            if missing:
                raise ValueError(f"Missing columns: {missing}")
            
            # Add adj close if not present
            if 'adj_close' not in df.columns:
                df['adj_close'] = df['close']
            
            # Select and order columns
            df = df[['date', 'open', 'high', 'low', 'close', 'adj_close', 'volume']]
            
            # Sort by date
            df = df.sort_values('date').reset_index(drop=True)
            
            print(f"✓ Downloaded {len(df)} rows from {df['date'].min()} to {df['date'].max()}")
            
            # Save if requested
            if save:
                Path('data/raw').mkdir(parents=True, exist_ok=True)
                save_path = 'data/raw/btc_raw.csv'
                df.to_csv(save_path, index=False)
                print(f"✓ Saved to {save_path}")
            
            return df
            
        except Exception as e:
            print(f"✗ Attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                sleep(2 ** attempt)  # Exponential backoff
            else:
                raise RuntimeError(f"Failed to download data after {max_retries} attempts")

if __name__ == "__main__":
    df = load_raw_btc()
    print(df.head())
    print(df.info())