"""
Module for saving processed data
"""

import pandas as pd
from pathlib import Path

def save_processed_data(df, output_path='data/processed/btc_clean.csv'):
    """
    Save processed dataframe to CSV
    
    Parameters:
    -----------
    df : pd.DataFrame
        Processed dataframe
    output_path : str
        Path to save CSV
    """
    
    # Create directory if it doesn't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save without index
    df.to_csv(output_path, index=False)
    print(f"✓ Saved processed data to {output_path}")
    
    return output_path

def save_features(df_features, df_scaled, output_dir='data/processed/'):
    """
    Save feature dataframes
    
    Parameters:
    -----------
    df_features : pd.DataFrame
        Unscaled features
    df_scaled : pd.DataFrame
        Scaled features
    output_dir : str
        Directory to save files
    """
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    features_path = Path(output_dir) / 'features.csv'
    scaled_path = Path(output_dir) / 'features_scaled.csv'
    
    df_features.to_csv(features_path, index=False)
    df_scaled.to_csv(scaled_path, index=False)
    
    print(f"✓ Saved features to {features_path}")
    print(f"✓ Saved scaled features to {scaled_path}")
    
    return features_path, scaled_path

if __name__ == "__main__":
    # Example usage
    import pandas as pd
    df = pd.DataFrame({'date': ['2024-01-01'], 'close': [50000]})
    save_processed_data(df, 'data/processed/test.csv')