"""
Compare HMM and GMM models
"""

import pandas as pd
import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def compare_label_agreement(hmm_labels, gmm_labels):
    """
    Compare agreement between HMM and GMM labels
    
    Parameters:
    -----------
    hmm_labels : np.ndarray
        HMM state labels
    gmm_labels : np.ndarray
        GMM cluster labels
        
    Returns:
    --------
    dict
        Agreement metrics
    """
    
    metrics = {
        'adjusted_rand_index': adjusted_rand_score(hmm_labels, gmm_labels),
        'normalized_mutual_info': normalized_mutual_info_score(hmm_labels, gmm_labels)
    }
    
    return metrics

def create_confusion_matrix(hmm_labels, gmm_labels, save_plot=True):
    """
    Create confusion matrix between HMM and GMM labels
    
    Parameters:
    -----------
    hmm_labels : np.ndarray
        HMM labels
    gmm_labels : np.ndarray
        GMM labels
    save_plot : bool
        Whether to save plot
        
    Returns:
    --------
    np.ndarray
        Confusion matrix
    """
    
    cm = confusion_matrix(hmm_labels, gmm_labels)
    
    if save_plot:
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=[f'GMM {i}' for i in range(cm.shape[1])],
            yticklabels=[f'HMM {i}' for i in range(cm.shape[0])]
        )
        plt.title('Confusion Matrix: HMM vs GMM', fontsize=14, fontweight='bold')
        plt.xlabel('GMM Clusters', fontsize=12)
        plt.ylabel('HMM States', fontsize=12)
        plt.tight_layout()
        
        Path('results/plots').mkdir(parents=True, exist_ok=True)
        plot_path = 'results/plots/hmm_vs_gmm_confusion.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved confusion matrix to {plot_path}")
        plt.close()
    
    return cm

def compare_regime_stability(hmm_labels, gmm_labels):
    """
    Compare regime stability between models
    
    Parameters:
    -----------
    hmm_labels : np.ndarray
        HMM labels
    gmm_labels : np.ndarray
        GMM labels
        
    Returns:
    --------
    dict
        Stability comparison
    """
    
    def count_transitions(labels):
        return np.sum(labels[:-1] != labels[1:])
    
    def avg_duration(labels):
        durations = []
        current = labels[0]
        dur = 1
        for i in range(1, len(labels)):
            if labels[i] == current:
                dur += 1
            else:
                durations.append(dur)
                current = labels[i]
                dur = 1
        durations.append(dur)
        return np.mean(durations)
    
    comparison = {
        'hmm_transitions': count_transitions(hmm_labels),
        'gmm_transitions': count_transitions(gmm_labels),
        'hmm_avg_duration': avg_duration(hmm_labels),
        'gmm_avg_duration': avg_duration(gmm_labels)
    }
    
    return comparison

def compare_models(hmm_labels, gmm_labels, df):
    """
    Comprehensive model comparison
    
    Parameters:
    -----------
    hmm_labels : np.ndarray or pd.Series
        HMM state labels
    gmm_labels : np.ndarray or pd.Series
        GMM cluster labels
    df : pd.DataFrame
        Original dataframe with dates
        
    Returns:
    --------
    dict
        Comparison results
    """
    
    print("Comparing HMM and GMM models...")
    
    # Convert to numpy arrays
    if isinstance(hmm_labels, pd.Series):
        hmm_labels = hmm_labels.values
    if isinstance(gmm_labels, pd.Series):
        gmm_labels = gmm_labels.values
    
    # ✅ Fix: Adjust df length to match labels
    df_comparison = df.copy()
    if len(hmm_labels) != len(df_comparison):
        print(f"  ⚠️ Adjusting DataFrame length:")
        print(f"     DataFrame: {len(df_comparison)} -> {len(hmm_labels)}")
        df_comparison = df_comparison.tail(len(hmm_labels)).reset_index(drop=True)
    
    # Agreement metrics
    agreement = compare_label_agreement(hmm_labels, gmm_labels)
    print(f"  Adjusted Rand Index: {agreement['adjusted_rand_index']:.4f}")
    print(f"  Normalized Mutual Info: {agreement['normalized_mutual_info']:.4f}")
    
    # Confusion matrix
    cm = create_confusion_matrix(hmm_labels, gmm_labels)
    
    # Stability comparison
    stability = compare_regime_stability(hmm_labels, gmm_labels)
    print(f"  HMM transitions: {stability['hmm_transitions']}")
    print(f"  GMM transitions: {stability['gmm_transitions']}")
    print(f"  HMM avg duration: {stability['hmm_avg_duration']:.1f} days")
    print(f"  GMM avg duration: {stability['gmm_avg_duration']:.1f} days")
    
    # Combine results
    results = {
        **agreement,
        **stability,
        'confusion_matrix': cm.tolist()
    }
    
    # Save comparison
    Path('results/tables').mkdir(parents=True, exist_ok=True)
    comparison_df = pd.DataFrame([{k: v for k, v in results.items() if k != 'confusion_matrix'}])
    comparison_path = 'results/tables/model_comparison.csv'
    comparison_df.to_csv(comparison_path, index=False)
    print(f"✓ Saved comparison to {comparison_path}")
    
    # Create visual comparison plot with adjusted df
    create_timeline_comparison(df_comparison, hmm_labels, gmm_labels)
    
    return results

def create_timeline_comparison(df, hmm_labels, gmm_labels):
    """
    Create timeline plot comparing HMM and GMM regime assignments
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with date and close columns (already length-matched)
    hmm_labels : np.ndarray
        HMM labels (same length as df)
    gmm_labels : np.ndarray
        GMM labels (same length as df)
    """
    
    # ✅ Double check lengths match
    if len(df) != len(hmm_labels) or len(df) != len(gmm_labels):
        print(f"  ⚠️ Length mismatch in timeline plot:")
        print(f"     df: {len(df)}, hmm: {len(hmm_labels)}, gmm: {len(gmm_labels)}")
        min_len = min(len(df), len(hmm_labels), len(gmm_labels))
        df = df.iloc[:min_len].copy()
        hmm_labels = hmm_labels[:min_len]
        gmm_labels = gmm_labels[:min_len]
        print(f"  ✓ Adjusted all to length: {min_len}")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    
    # Plot HMM regimes
    ax1.plot(df['date'], df['close'], color='black', alpha=0.3, linewidth=0.5)
    for state in np.unique(hmm_labels):
        mask = hmm_labels == state
        ax1.scatter(df.loc[mask, 'date'], df.loc[mask, 'close'], 
                   label=f'State {state}', s=1, alpha=0.6)
    ax1.set_ylabel('Price (USD)', fontsize=10)
    ax1.set_title('HMM Regime Detection', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Plot GMM clusters
    ax2.plot(df['date'], df['close'], color='black', alpha=0.3, linewidth=0.5)
    for cluster in np.unique(gmm_labels):
        mask = gmm_labels == cluster
        ax2.scatter(df.loc[mask, 'date'], df.loc[mask, 'close'],
                   label=f'Cluster {cluster}', s=1, alpha=0.6)
    ax2.set_ylabel('Price (USD)', fontsize=10)
    ax2.set_xlabel('Date', fontsize=10)
    ax2.set_title('GMM Clustering', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = 'results/plots/hmm_vs_gmm_timeline.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved timeline comparison to {plot_path}")
    plt.close()

if __name__ == "__main__":
    # Load data
    df = pd.read_csv('data/processed/btc_clean.csv')
    df['date'] = pd.to_datetime(df['date'])
    
    hmm_df = pd.read_csv('results/tables/hmm_labels.csv')
    gmm_df = pd.read_csv('results/tables/gmm_labels.csv')
    
    # Compare
    results = compare_models(hmm_df['state'].values, gmm_df['cluster'].values, df)
    
    print("\nComparison complete!")