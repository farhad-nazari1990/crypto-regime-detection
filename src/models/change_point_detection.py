"""
Change Point Detection using Ruptures or Bayesian methods
"""

import pandas as pd
import numpy as np
import ruptures as rpt
from pathlib import Path

def detect_change_points_ruptures(data, model='rbf', min_size=20, jump=5, penalty=None):
    """
    Detect change points using Ruptures library
    
    Parameters:
    -----------
    data : np.ndarray or pd.DataFrame
        Time series data
    model : str
        Detection model ('rbf', 'l2', 'l1', 'normal', 'ar')
    min_size : int
        Minimum segment size
    jump : int
        Subsample step for faster computation
    penalty : float or None
        Penalty value (higher = fewer change points)
        
    Returns:
    --------
    list
        List of change point indices
    """
    
    print(f"Detecting change points using Ruptures ({model} model)...")
    
    # Convert to numpy array if DataFrame
    if isinstance(data, pd.DataFrame):
        # Use only numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [col for col in numeric_cols if 'date' not in col.lower()]
        signal = data[numeric_cols].values
    else:
        signal = data
    
    # Auto-calculate penalty if not provided
    if penalty is None:
        penalty = 3 * np.log(len(signal))
    
    print(f"  Signal shape: {signal.shape}")
    print(f"  Using penalty: {penalty:.2f}")
    
    # Create model
    if model == 'rbf':
        algo = rpt.Pelt(model=model, min_size=min_size, jump=jump).fit(signal)
    else:
        algo = rpt.Pelt(model=model, min_size=min_size, jump=jump).fit(signal)
    
    # Detect change points
    result = algo.predict(pen=penalty)
    
    # Remove last point (it's always the length of signal)
    change_points = [cp for cp in result if cp < len(signal)]
    
    print(f"✓ Detected {len(change_points)} change points")
    
    return change_points

def detect_change_points(df, config=None):
    """
    Detect change points with configuration
    
    Parameters:
    -----------
    df : pd.DataFrame
        Feature dataframe
    config : dict
        CPD configuration
        
    Returns:
    --------
    list
        Change point indices
    """
    
    if config is None:
        config = {
            'method': 'ruptures',
            'model': 'rbf',
            'min_size': 20,
            'jump': 5
        }
    
    method = config.get('method', 'ruptures')
    
    if method == 'ruptures':
        change_points = detect_change_points_ruptures(
            df,
            model=config.get('model', 'rbf'),
            min_size=config.get('min_size', 20),
            jump=config.get('jump', 5),
            penalty=config.get('penalty', None)
        )
    else:
        raise ValueError(f"Unknown CPD method: {method}")
    
    # Create results dataframe
    if 'date' in df.columns:
        cpd_dates = df.iloc[change_points]['date'].tolist()
        cpd_df = pd.DataFrame({
            'index': change_points,
            'date': cpd_dates
        })
    else:
        cpd_df = pd.DataFrame({
            'index': change_points
        })
    
    # Save results
    Path('results/tables').mkdir(parents=True, exist_ok=True)
    cpd_path = 'results/tables/cpd_points.csv'
    cpd_df.to_csv(cpd_path, index=False)
    print(f"✓ Saved change points to {cpd_path}")
    
    # Print change point summary
    if len(change_points) > 0:
        print("\\n  Change points detected at indices:")
        for i, cp in enumerate(change_points[:10]):  # Show first 10
            if 'date' in df.columns:
                print(f"    {i+1}. Index {cp} ({df.iloc[cp]['date']})")
            else:
                print(f"    {i+1}. Index {cp}")
        if len(change_points) > 10:
            print(f"    ... and {len(change_points) - 10} more")
    
    return change_points

if __name__ == "__main__":
    # Load scaled features
    df = pd.read_csv('data/processed/features_scaled.csv')
    df['date'] = pd.to_datetime(df['date'])
    
    # Detect change points
    config = {
        'method': 'ruptures',
        'model': 'rbf',
        'min_size': 20,
        'jump': 5
    }
    
    change_points = detect_change_points(df, config)
    print(f"\\nTotal change points: {len(change_points)}")