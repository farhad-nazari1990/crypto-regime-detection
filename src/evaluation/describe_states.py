"""
Generate descriptive analysis of each regime state
"""

import pandas as pd
import numpy as np

def classify_regime(metrics):
    """
    Classify regime based on return and volatility characteristics
    
    Parameters:
    -----------
    metrics : dict
        Performance metrics for a regime
        
    Returns:
    --------
    str
        Regime classification
    """
    
    avg_return = metrics.get('mean_daily_return', 0)
    volatility = metrics.get('volatility', 0)
    sharpe = metrics.get('sharpe_ratio', 0)
    
    # Classification logic
    if avg_return > 0.003 and volatility < 0.03:
        return "Low Volatility Bull"
    elif avg_return > 0.003 and volatility >= 0.03:
        return "High Volatility Rally"
    elif avg_return < -0.002 and volatility > 0.04:
        return "Panic/Crash"
    elif abs(avg_return) < 0.002 and volatility < 0.03:
        return "Consolidation"
    elif avg_return < 0 and volatility < 0.04:
        return "Bear Market"
    else:
        return "Mixed/Transition"

def describe_regime_characteristics(df, state):
    """
    Generate detailed description of regime characteristics
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with features and state labels
    state : int
        State number
        
    Returns:
    --------
    dict
        Regime characteristics
    """
    
    state_data = df[df['state'] == state]
    
    characteristics = {
        'state': state,
        'n_observations': len(state_data)
    }
    
    # Price characteristics
    if 'close' in state_data.columns:
        characteristics['avg_price'] = state_data['close'].mean()
        characteristics['price_range'] = state_data['close'].max() - state_data['close'].min()
    
    # Return characteristics
    if 'return' in state_data.columns:
        characteristics['return_mean'] = state_data['return'].mean()
        characteristics['return_std'] = state_data['return'].std()
        characteristics['return_skew'] = state_data['return'].skew()
        characteristics['return_kurtosis'] = state_data['return'].kurtosis()
    
    # Volatility characteristics
    vol_cols = [col for col in state_data.columns if 'volatility' in col]
    if vol_cols:
        characteristics['avg_volatility'] = state_data[vol_cols].mean().mean()
    
    # Momentum characteristics
    mom_cols = [col for col in state_data.columns if 'momentum' in col]
    if mom_cols:
        characteristics['avg_momentum'] = state_data[mom_cols].mean().mean()
    
    # Volume characteristics
    if 'volume' in state_data.columns:
        characteristics['avg_volume'] = state_data['volume'].mean()
        characteristics['volume_std'] = state_data['volume'].std()
    
    return characteristics

def generate_state_descriptions(df, states, performance_df=None):
    """
    Generate comprehensive descriptions for all states
    
    Parameters:
    -----------
    df : pd.DataFrame
        Feature dataframe
    states : np.ndarray
        State labels
    performance_df : pd.DataFrame or None
        Pre-calculated performance metrics
        
    Returns:
    --------
    pd.DataFrame
        State descriptions
    """
    
    print("Generating state descriptions...")
    
    df_analysis = df.copy()
    df_analysis['state'] = states
    
    descriptions = []
    
    for state in sorted(df_analysis['state'].unique()):
        print(f"  Describing State {state}...")
        
        # Get characteristics
        chars = describe_regime_characteristics(df_analysis, state)
        
        # Get performance metrics if available
        if performance_df is not None:
            perf = performance_df[performance_df['state'] == state].iloc[0].to_dict()
            chars.update(perf)
        
        # Classify regime
        chars['regime_type'] = classify_regime(chars)
        
        descriptions.append(chars)
    
    desc_df = pd.DataFrame(descriptions)
    
    # Save
    from pathlib import Path
    Path('results/tables').mkdir(parents=True, exist_ok=True)
    output_path = 'results/tables/state_descriptions.csv'
    desc_df.to_csv(output_path, index=False)
    print(f"✓ Saved state descriptions to {output_path}")
    
    # Print summary
    print("\\n  Regime Classifications:")
    for _, row in desc_df.iterrows():
        print(f"    State {row['state']}: {row['regime_type']}")
    
    return desc_df

if __name__ == "__main__":
    # Load data
    df = pd.read_csv('data/processed/features.csv')
    df['date'] = pd.to_datetime(df['date'])
    
    labels = pd.read_csv('results/tables/hmm_labels.csv')
    performance = pd.read_csv('results/tables/performance_per_regime.csv')
    
    # Generate descriptions
    descriptions = generate_state_descriptions(
        df, 
        labels['state'].values,
        performance
    )
    
    print("\\nDescriptions:")
    print(descriptions[['state', 'regime_type', 'return_mean', 'avg_volatility']].to_string(index=False))