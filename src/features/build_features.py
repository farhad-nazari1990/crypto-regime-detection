"""
Feature engineering module for crypto regime detection
Creates advanced features from cleaned data
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import joblib
from pathlib import Path

def calculate_volatility(df, windows=[5, 20]):
    """Calculate rolling volatility for multiple windows"""
    result = df.copy()
    for window in windows:
        result[f'volatility_{window}d'] = result['log_return'].rolling(window=window).std()
    return result

def calculate_momentum(df, windows=[5, 10, 21]):
    """Calculate momentum (rate of change) for multiple windows"""
    result = df.copy()
    for window in windows:
        result[f'momentum_{window}d'] = df['close'].pct_change(periods=window)
    return result

def calculate_zscores(df, window=20):
    """Calculate z-scores for price and volume"""
    result = df.copy()
    
    # Price z-score
    price_mean = df['close'].rolling(window=window).mean()
    price_std = df['close'].rolling(window=window).std()
    result['price_zscore'] = (df['close'] - price_mean) / price_std
    
    # Volume z-score
    vol_mean = df['volume'].rolling(window=window).mean()
    vol_std = df['volume'].rolling(window=window).std()
    result['volume_zscore'] = (df['volume'] - vol_mean) / vol_std
    
    return result

def calculate_volume_features(df, ma_window=20):
    """Calculate volume-based features"""
    result = df.copy()
    
    # Volume moving average ratio
    vol_ma = df['volume'].rolling(window=ma_window).mean()
    result['volume_ratio'] = df['volume'] / vol_ma
    
    return result

def calculate_rsi(df, window=14):
    """Calculate Relative Strength Index"""
    result = df.copy()
    
    # Calculate price changes
    delta = df['close'].diff()
    
    # Separate gains and losses
    gains = delta.where(delta > 0, 0)
    losses = -delta.where(delta < 0, 0)
    
    # Calculate average gains and losses
    avg_gains = gains.rolling(window=window).mean()
    avg_losses = losses.rolling(window=window).mean()
    
    # Calculate RS and RSI
    rs = avg_gains / avg_losses
    result['rsi'] = 100 - (100 / (1 + rs))
    
    return result

def calculate_bollinger_bands(df, window=20, num_std=2):
    """Calculate Bollinger Bands features"""
    result = df.copy()
    
    # Middle band (SMA)
    sma = df['close'].rolling(window=window).mean()
    std = df['close'].rolling(window=window).std()
    
    # Upper and lower bands
    upper_band = sma + (std * num_std)
    lower_band = sma - (std * num_std)
    
    # Bollinger band position (0 to 1)
    result['bb_position'] = (df['close'] - lower_band) / (upper_band - lower_band)
    
    # Bollinger band width
    result['bb_width'] = (upper_band - lower_band) / sma
    
    return result

def build_features(df, config):
    """
    Build all features based on configuration
    
    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned dataframe
    config : dict
        Feature configuration
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with all features
    """
    
    print("Building features...")
    result = df.copy()
    
    # Volatility features
    result = calculate_volatility(result, config['volatility_windows'])
    print(f"  ✓ Volatility features: {config['volatility_windows']}")
    
    # Momentum features
    result = calculate_momentum(result, config['momentum_windows'])
    print(f"  ✓ Momentum features: {config['momentum_windows']}")
    
    # Z-score features
    result = calculate_zscores(result, config['zscore_window'])
    print(f"  ✓ Z-score features (window={config['zscore_window']})")
    
    # Volume features
    result = calculate_volume_features(result, config['volume_ma_window'])
    print(f"  ✓ Volume features (window={config['volume_ma_window']})")
    
    # Price change features
    for window in config.get('price_change_windows', [1, 5, 10]):
        result[f'price_change_{window}d'] = result['close'].pct_change(periods=window)
    print(f"  ✓ Price change features")
    
    # RSI
    if config.get('use_rsi', True):
        result = calculate_rsi(result, config.get('rsi_window', 14))
        print(f"  ✓ RSI (window={config.get('rsi_window', 14)})")
    
    # Bollinger Bands
    if config.get('use_bollinger', True):
        result = calculate_bollinger_bands(
            result, 
            config.get('bollinger_window', 20),
            config.get('bollinger_std', 2)
        )
        print(f"  ✓ Bollinger Bands")
    
    # Remove NaN values
    initial_len = len(result)
    result = result.dropna().reset_index(drop=True)
    print(f"  Removed {initial_len - len(result)} rows with NaN")
    
    print(f"✓ Built {result.shape[1] - df.shape[1]} new features")
    
    return result

def scale_features(df, method='standard', feature_cols=None, save_scaler=True):
    """
    Scale features for ML models
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with features
    method : str
        Scaling method ('standard', 'minmax', 'robust')
    feature_cols : list or None
        List of columns to scale (None = all numeric except date)
    save_scaler : bool
        Whether to save the scaler
        
    Returns:
    --------
    pd.DataFrame, scaler
        Scaled dataframe and fitted scaler
    """
    
    print(f"Scaling features using {method} scaler...")
    
    # Select feature columns
    if feature_cols is None:
        feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # Remove date-related columns
        feature_cols = [col for col in feature_cols if 'date' not in col.lower()]
    
    # Create scaler
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'robust':
        scaler = RobustScaler()
    else:
        raise ValueError(f"Unknown scaling method: {method}")
    
    # Fit and transform
    df_scaled = df.copy()
    df_scaled[feature_cols] = scaler.fit_transform(df[feature_cols])
    
    print(f"✓ Scaled {len(feature_cols)} features")
    
    # Save scaler
    if save_scaler:
        Path('results/models').mkdir(parents=True, exist_ok=True)
        scaler_path = 'results/models/scaler.pkl'
        joblib.dump(scaler, scaler_path)
        print(f"✓ Saved scaler to {scaler_path}")
    
    return df_scaled, scaler

def build_and_scale_features(df, config):
    """
    Complete feature engineering pipeline
    
    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned dataframe
    config : dict
        Feature configuration
        
    Returns:
    --------
    tuple
        (df_features, df_scaled, scaler)
    """
    
    # Build features
    df_features = build_features(df, config)
    
    # Save unscaled features
    Path('data/processed').mkdir(parents=True, exist_ok=True)
    df_features.to_csv('data/processed/features.csv', index=False)
    print(f"✓ Saved unscaled features to data/processed/features.csv")
    
    # Scale features
    df_scaled, scaler = scale_features(df_features, method=config['scaling_method'])
    
    # Save scaled features
    df_scaled.to_csv('data/processed/features_scaled.csv', index=False)
    print(f"✓ Saved scaled features to data/processed/features_scaled.csv")
    
    return df_features, df_scaled, scaler

if __name__ == "__main__":
    from src.features.feature_config import FEATURE_CONFIG
    from src.data.clean_data import load_and_clean
    
    # Load and clean data
    df_clean = load_and_clean()
    
    # Build and scale features
    df_features, df_scaled, scaler = build_and_scale_features(df_clean, FEATURE_CONFIG)
    
    print("\\nFeature summary:")
    print(df_features.describe())