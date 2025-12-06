"""
Feature engineering configuration
Defines all parameters for feature creation
"""

# Default feature configuration
FEATURE_CONFIG = {
    # Volatility windows (in days)
    'volatility_windows': [5, 20],
    
    # Momentum windows (in days)
    'momentum_windows': [5, 10, 21],
    
    # Z-score calculation window
    'zscore_window': 20,
    
    # Volume moving average window
    'volume_ma_window': 20,
    
    # Scaling method
    'scaling_method': 'standard',  # 'standard', 'minmax', or 'robust'
    
    # Price change windows
    'price_change_windows': [1, 5, 10],
    
    # Additional technical indicators
    'use_rsi': True,
    'rsi_window': 14,
    
    'use_bollinger': True,
    'bollinger_window': 20,
    'bollinger_std': 2,
}

def get_feature_names(config=None):
    """
    Get list of all feature names that will be generated
    
    Parameters:
    -----------
    config : dict or None
        Feature configuration dictionary
        
    Returns:
    --------
    list
        List of feature names
    """
    
    if config is None:
        config = FEATURE_CONFIG
    
    features = ['return', 'log_return']
    
    # Volatility features
    for window in config['volatility_windows']:
        features.append(f'volatility_{window}d')
    
    # Momentum features
    for window in config['momentum_windows']:
        features.append(f'momentum_{window}d')
    
    # Z-score features
    features.append('price_zscore')
    features.append('volume_zscore')
    
    # Volume features
    features.append('volume_ratio')
    
    # Price changes
    for window in config.get('price_change_windows', [1, 5, 10]):
        features.append(f'price_change_{window}d')
    
    # RSI
    if config.get('use_rsi', True):
        features.append('rsi')
    
    # Bollinger bands
    if config.get('use_bollinger', True):
        features.extend(['bb_position', 'bb_width'])
    
    return features

if __name__ == "__main__":
    print("Default Feature Configuration:")
    print("-" * 50)
    for key, value in FEATURE_CONFIG.items():
        print(f"{key:25s}: {value}")
    
    print("\\nGenerated Features:")
    print("-" * 50)
    for i, feature in enumerate(get_feature_names(), 1):
        print(f"{i:2d}. {feature}")