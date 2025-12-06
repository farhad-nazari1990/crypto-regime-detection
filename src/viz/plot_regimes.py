"""
Visualization module for regime detection
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def plot_regimes(df, states, output_dir='results/plots', title='BTC Regime Detection'):
    """
    Plot BTC price with colored regimes
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with date and close price
    states : np.ndarray or pd.Series
        State labels
    output_dir : str
        Directory to save plots
    title : str
        Plot title
    """
    
    print("Generating regime visualization...")
    
    # Convert states to numpy array
    if isinstance(states, pd.Series):
        states = states.values
    
    # ✅ Fix: Adjust df length to match states
    df_plot = df.copy()
    if len(states) != len(df_plot):
        print(f"  ⚠️ Length mismatch detected:")
        print(f"     DataFrame length: {len(df_plot)}")
        print(f"     States length: {len(states)}")
        df_plot = df_plot.tail(len(states)).reset_index(drop=True)
        print(f"  ✓ Adjusted DataFrame to {len(df_plot)} rows")
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), 
                                     gridspec_kw={'height_ratios': [3, 1]})
    
    # Define colors for states
    n_states = len(np.unique(states))
    colors = sns.color_palette('husl', n_states)
    
    # Plot 1: Price with colored regimes
    for state in np.unique(states):
        mask = states == state
        ax1.scatter(df_plot.loc[mask, 'date'], df_plot.loc[mask, 'close'],
                   c=[colors[state]], label=f'State {state}',
                   s=10, alpha=0.7, edgecolors='none')
    
    # Add price line
    ax1.plot(df_plot['date'], df_plot['close'], color='black', alpha=0.2, 
             linewidth=0.5, zorder=0)
    
    ax1.set_ylabel('BTC Price (USD)', fontsize=14, fontweight='bold')
    ax1.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax1.legend(loc='upper left', fontsize=10, framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_yscale('log')
    
    # Plot 2: State timeline
    for state in np.unique(states):
        mask = states == state
        ax2.scatter(df_plot.loc[mask, 'date'], 
                   np.full(np.sum(mask), state),
                   c=[colors[state]], s=20, alpha=0.8)
    
    ax2.set_ylabel('Regime State', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax2.set_yticks(range(n_states))
    ax2.grid(True, alpha=0.3, axis='x', linestyle='--')
    
    plt.tight_layout()
    
    # Save plot
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    plot_path = Path(output_dir) / 'regime_detection_plot.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved regime plot to {plot_path}")
    
    plt.close()
    
    # Create additional scatter plot
    plot_regime_scatter(df_plot, states, output_dir)

def plot_regime_scatter(df, states, output_dir):
    """
    Create scatter plot of returns vs volatility colored by regime
    """
    
    if 'return' not in df.columns or 'volatility_5d' not in df.columns:
        print("  Skipping scatter plot (missing return or volatility columns)")
        return
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    n_states = len(np.unique(states))
    colors = sns.color_palette('husl', n_states)
    
    for state in np.unique(states):
        mask = states == state
        ax.scatter(df.loc[mask, 'volatility_5d'], 
                  df.loc[mask, 'return'],
                  c=[colors[state]], label=f'State {state}',
                  s=20, alpha=0.6, edgecolors='none')
    
    ax.set_xlabel('Volatility (5-day)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Daily Return', fontsize=12, fontweight='bold')
    ax.set_title('Regime Clustering: Return vs Volatility', 
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
    
    plt.tight_layout()
    plot_path = Path(output_dir) / 'regime_scatter_plot.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved scatter plot to {plot_path}")
    plt.close()

if __name__ == "__main__":
    # Load data
    df = pd.read_csv('data/processed/btc_clean.csv')
    df['date'] = pd.to_datetime(df['date'])
    
    labels = pd.read_csv('results/tables/hmm_labels.csv')
    
    # Plot
    plot_regimes(df, labels['state'].values)