"""
Visualization for Change Point Detection
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def plot_cpd(df, change_points, output_dir='results/plots', title='Change Point Detection'):
    """
    Plot price with change point overlays
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with date and close price
    change_points : list
        List of change point indices
    output_dir : str
        Directory to save plot
    title : str
        Plot title
    """
    
    print("Generating change point detection visualization...")
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Plot price
    ax.plot(df['date'], df['close'], color='#2E86AB', linewidth=2, 
           label='BTC Price', zorder=2)
    
    # Plot change points
    if len(change_points) > 0:
        cpd_dates = df.iloc[change_points]['date']
        cpd_prices = df.iloc[change_points]['close']
        
        for date, price in zip(cpd_dates, cpd_prices):
            ax.axvline(x=date, color='red', linestyle='--', 
                      linewidth=1.5, alpha=0.7, zorder=1)
        
        ax.scatter(cpd_dates, cpd_prices, color='red', s=100, 
                  marker='o', edgecolors='darkred', linewidths=2,
                  label=f'Change Points ({len(change_points)})', 
                  zorder=3)
    
    ax.set_xlabel('Date', fontsize=14, fontweight='bold')
    ax.set_ylabel('BTC Price (USD)', fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.legend(fontsize=12, loc='upper left', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_yscale('log')
    
    # Add annotations for first few change points
    if len(change_points) > 0:
        for i, (date, price) in enumerate(zip(cpd_dates[:5], cpd_prices[:5])):
            ax.annotate(f'CP{i+1}', 
                       xy=(date, price), 
                       xytext=(10, 10),
                       textcoords='offset points',
                       fontsize=9,
                       bbox=dict(boxstyle='round,pad=0.3', 
                                facecolor='yellow', alpha=0.7),
                       arrowprops=dict(arrowstyle='->', 
                                     connectionstyle='arc3,rad=0'))
    
    plt.tight_layout()
    
    # Save plot
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    plot_path = Path(output_dir) / 'change_point_detection.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved CPD plot to {plot_path}")
    
    plt.close()
    
    # Create segments plot
    plot_cpd_segments(df, change_points, output_dir)

def plot_cpd_segments(df, change_points, output_dir):
    """
    Plot price segments between change points with different colors
    """
    
    if len(change_points) == 0:
        print("  No change points to segment")
        return
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Add boundaries
    boundaries = [0] + change_points + [len(df)]
    
    # Define colors
    colors = plt.cm.tab10(np.linspace(0, 1, len(boundaries) - 1))
    
    # Plot each segment
    for i in range(len(boundaries) - 1):
        start_idx = boundaries[i]
        end_idx = boundaries[i + 1]
        
        segment = df.iloc[start_idx:end_idx]
        ax.plot(segment['date'], segment['close'], 
               color=colors[i], linewidth=2, 
               label=f'Segment {i+1}', alpha=0.8)
    
    # Add change point markers
    if len(change_points) > 0:
        cpd_dates = df.iloc[change_points]['date']
        cpd_prices = df.iloc[change_points]['close']
        ax.scatter(cpd_dates, cpd_prices, color='red', 
                  s=150, marker='X', edgecolors='darkred', 
                  linewidths=2, zorder=5, label='Change Points')
    
    ax.set_xlabel('Date', fontsize=14, fontweight='bold')
    ax.set_ylabel('BTC Price (USD)', fontsize=14, fontweight='bold')
    ax.set_title('Market Segments (Change Point Detection)', 
                fontsize=16, fontweight='bold', pad=20)
    ax.legend(fontsize=10, loc='upper left', framealpha=0.9, ncol=2)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_yscale('log')
    
    plt.tight_layout()
    
    plot_path = Path(output_dir) / 'cpd_segments.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved CPD segments plot to {plot_path}")
    plt.close()

if __name__ == "__main__":
    # Load data
    df = pd.read_csv('data/processed/btc_clean.csv')
    df['date'] = pd.to_datetime(df['date'])
    
    cpd_df = pd.read_csv('results/tables/cpd_points.csv')
    change_points = cpd_df['index'].tolist()
    
    # Plot
    plot_cpd(df, change_points)