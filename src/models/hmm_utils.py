"""
Utility functions for HMM analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def analyze_transition_matrix(transmat, save_plot=True, output_dir='results/plots'):
    """
    Analyze and visualize transition matrix
    
    Parameters:
    -----------
    transmat : np.ndarray
        Transition probability matrix
    save_plot : bool
        Whether to save the plot
    output_dir : str
        Directory to save plot
    """
    
    n_states = transmat.shape[0]
    
    # Create heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        transmat,
        annot=True,
        fmt='.3f',
        cmap='YlOrRd',
        xticklabels=[f'State {i}' for i in range(n_states)],
        yticklabels=[f'State {i}' for i in range(n_states)],
        cbar_kws={'label': 'Transition Probability'}
    )
    plt.title('HMM State Transition Matrix', fontsize=14, fontweight='bold')
    plt.xlabel('To State', fontsize=12)
    plt.ylabel('From State', fontsize=12)
    plt.tight_layout()
    
    if save_plot:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        plot_path = Path(output_dir) / 'hmm_transition_matrix.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved transition matrix plot to {plot_path}")
    
    plt.close()
    
    # Calculate persistence (diagonal)
    persistence = np.diag(transmat)
    print("\nState Persistence:")
    for i, p in enumerate(persistence):
        print(f"  State {i}: {p:.3f}")
    
    return persistence

def extract_regime_profiles(df, states, model):
    """
    Extract statistical profiles for each regime
    
    Parameters:
    -----------
    df : pd.DataFrame
        Original dataframe with features
    states : np.ndarray
        State labels
    model : HMM model
        Trained HMM model
        
    Returns:
    --------
    pd.DataFrame
        Regime profile summary
    """
    
    df_copy = df.copy()
    df_copy['state'] = states
    
    profiles = []
    
    for state in range(model.n_components):
        state_data = df_copy[df_copy['state'] == state]
        
        if len(state_data) == 0:
            continue
        
        profile = {
            'state': state,
            'n_days': len(state_data),
            'pct_days': len(state_data) / len(df_copy) * 100,
            'mean_return': state_data['return'].mean() if 'return' in state_data.columns else np.nan,
            'mean_volatility': state_data[[c for c in state_data.columns if 'volatility' in c]].mean().mean(),
            'mean_volume': state_data['volume'].mean() if 'volume' in state_data.columns else np.nan,
            'avg_duration': calculate_avg_duration(states, state)
        }
        
        profiles.append(profile)
    
    profiles_df = pd.DataFrame(profiles)
    
    # Save profiles
    Path('results/tables').mkdir(parents=True, exist_ok=True)
    profiles_path = 'results/tables/regime_profiles.csv'
    profiles_df.to_csv(profiles_path, index=False)
    print(f"✓ Saved regime profiles to {profiles_path}")
    
    return profiles_df

def calculate_avg_duration(states, target_state):
    """Calculate average duration of stays in a state"""
    durations = []
    current_duration = 0
    
    for state in states:
        if state == target_state:
            current_duration += 1
        else:
            if current_duration > 0:
                durations.append(current_duration)
                current_duration = 0
    
    if current_duration > 0:
        durations.append(current_duration)
    
    return np.mean(durations) if durations else 0

def describe_states(df, states, model):
    """
    Generate comprehensive state descriptions
    
    Parameters:
    -----------
    df : pd.DataFrame
        Feature dataframe
    states : np.ndarray
        State labels  
    model : HMM model
        Trained model
        
    Returns:
    --------
    dict
        State descriptions
    """
    
    descriptions = {}
    df_copy = df.copy()
    df_copy['state'] = states
    
    for state in range(model.n_components):
        state_data = df_copy[df_copy['state'] == state]
        
        if len(state_data) == 0:
            continue
        
        # Calculate key metrics
        avg_return = state_data['return'].mean() if 'return' in state_data.columns else 0
        avg_vol = state_data[[c for c in state_data.columns if 'volatility' in c]].mean().mean()
        
        # Classify regime
        if avg_return > 0.002 and avg_vol < state_data[[c for c in state_data.columns if 'volatility' in c]].mean().median():
            regime_type = "Low Volatility Bull"
        elif avg_return > 0.002 and avg_vol > state_data[[c for c in state_data.columns if 'volatility' in c]].mean().median():
            regime_type = "High Volatility Rally"
        elif abs(avg_return) < 0.001:
            regime_type = "Consolidation"
        else:
            regime_type = "Bear/Panic"
        
        descriptions[state] = {
            'type': regime_type,
            'characteristics': {
                'return': avg_return,
                'volatility': avg_vol,
                'days': len(state_data)
            }
        }
    
    return descriptions

if __name__ == "__main__":
    # Load model and data
    import joblib
    model = joblib.load('results/models/hmm_model.pkl')
    df = pd.read_csv('data/processed/features.csv')
    labels = pd.read_csv('results/tables/hmm_labels.csv')
    
    # Analyze
    analyze_transition_matrix(model.transmat_)
    profiles = extract_regime_profiles(df, labels['state'].values, model)
    print("\nRegime Profiles:")
    print(profiles)