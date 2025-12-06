"""
Visualization for feature distributions by regime
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def plot_feature_distributions(df, states, output_dir='results/plots'):
    """
    Plot feature distributions for each regime
    
    Parameters:
    -----------
    df : pd.DataFrame
        Feature dataframe
    states : np.ndarray or pd.Series
        State labels
    output_dir : str
        Directory to save plots
    """
    
    print("Generating feature distribution plots...")
    
    # Convert states
    if isinstance(states, pd.Series):
        states = states.values
    
    # Add states to dataframe
    df_plot = df.copy()
    df_plot['state'] = states
    
    # Select key features to plot
    feature_cols = ['return', 'log_return', 'volatility_5d', 'volatility_20d',
                   'momentum_5d', 'momentum_10d', 'volume_ratio']
    
    # Filter to available features
    feature_cols = [col for col in feature_cols if col in df_plot.columns]
    
    if len(feature_cols) == 0:
        print("  No features available for plotting")
        return
    
    # Create distribution plots
    n_features = len(feature_cols)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
    axes = axes.flatten() if n_features > 1 else [axes]
    
    n_states = len(np.unique(states))
    colors = sns.color_palette('husl', n_states)
    
    for idx, feature in enumerate(feature_cols):
        ax = axes[idx]
        
        for state in np.unique(states):
            state_data = df_plot[df_plot['state'] == state][feature].dropna()
            if len(state_data) > 0:
                ax.hist(state_data, bins=50, alpha=0.5, 
                       label=f'State {state}', color=colors[state],
                       density=True)
        
        ax.set_xlabel(feature, fontsize=11, fontweight='bold')
        ax.set_ylabel('Density', fontsize=11, fontweight='bold')
        ax.set_title(f'{feature} Distribution by Regime', 
                    fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    # Hide extra subplots
    for idx in range(n_features, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    plot_path = Path(output_dir) / 'feature_distributions.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved feature distributions to {plot_path}")
    plt.close()
    
    # Create box plots
    plot_feature_boxplots(df_plot, feature_cols, output_dir)
    
    # Create violin plots for key features
    plot_key_feature_violins(df_plot, output_dir)

def plot_feature_boxplots(df_plot, feature_cols, output_dir):
    """
    Create box plots for features by regime
    """
    
    n_features = min(6, len(feature_cols))  # Limit to 6 features
    selected_features = feature_cols[:n_features]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for idx, feature in enumerate(selected_features):
        ax = axes[idx]
        df_plot.boxplot(column=feature, by='state', ax=ax)
        ax.set_xlabel('Regime State', fontsize=11, fontweight='bold')
        ax.set_ylabel(feature, fontsize=11, fontweight='bold')
        ax.set_title(f'{feature} by Regime', fontsize=12, fontweight='bold')
        plt.sca(ax)
        plt.xticks(rotation=0)
    
    # Hide extra subplots
    for idx in range(n_features, 6):
        axes[idx].axis('off')
    
    plt.suptitle('Feature Box Plots by Regime', fontsize=16, fontweight='bold', y=1.0)
    plt.tight_layout()
    
    plot_path = Path(output_dir) / 'feature_boxplots.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved box plots to {plot_path}")
    plt.close()

def plot_key_feature_violins(df_plot, output_dir):
    """
    Create violin plots for key features
    """
    
    key_features = ['return', 'volatility_5d', 'momentum_10d']
    key_features = [f for f in key_features if f in df_plot.columns]
    
    if len(key_features) == 0:
        return
    
    fig, axes = plt.subplots(1, len(key_features), figsize=(6 * len(key_features), 6))
    if len(key_features) == 1:
        axes = [axes]
    
    for idx, feature in enumerate(key_features):
        sns.violinplot(data=df_plot, x='state', y=feature, ax=axes[idx], 
                      palette='husl')
        axes[idx].set_xlabel('Regime State', fontsize=12, fontweight='bold')
        axes[idx].set_ylabel(feature, fontsize=12, fontweight='bold')
        axes[idx].set_title(f'{feature} Distribution', fontsize=14, fontweight='bold')
        axes[idx].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    plot_path = Path(output_dir) / 'feature_violins.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved violin plots to {plot_path}")
    plt.close()

if __name__ == "__main__":
    # Load data
    df = pd.read_csv('data/processed/features.csv')
    df['date'] = pd.to_datetime(df['date'])
    
    labels = pd.read_csv('results/tables/hmm_labels.csv')
    
    # Plot
    plot_feature_distributions(df, labels['state'].values)