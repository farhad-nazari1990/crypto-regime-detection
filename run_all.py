#!/usr/bin/env python3
"""
Master pipeline runner for Crypto Regime Detection
Executes all steps from data loading to final visualization
"""

import os
import sys
import argparse
import yaml
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.data.load_data import load_raw_btc
from src.data.clean_data import clean_btc_data
from src.features.build_features import build_and_scale_features
from src.models.hmm_model import train_hmm_model
from src.models.gmm_baseline import train_gmm_model
from src.models.change_point_detection import detect_change_points
from src.evaluation.performance_by_regime import evaluate_performance_by_regime
from src.evaluation.compare_models import compare_models
from src.viz.plot_regimes import plot_regimes
from src.viz.plot_cpd import plot_cpd
from src.viz.plot_feature_distributions import plot_feature_distributions

def create_directories(config):
    """Create necessary directories"""
    dirs = [
        'data/raw',
        'data/processed',
        config['paths']['results_dir'],
        config['paths']['models_dir'],
        config['paths']['plots_dir'],
        config['paths']['tables_dir']
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)

def load_config(config_path):
    """Load configuration from YAML"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main(config_path='config.yaml'):
    """Run complete pipeline"""
    
    print("=" * 60)
    print("CRYPTO REGIME DETECTION PIPELINE")
    print("=" * 60)
    
    # Load configuration
    config = load_config(config_path)
    create_directories(config)
    
    # Step 1: Load raw data
    print("\\n[1/9] Loading raw BTC data...")
    df_raw = load_raw_btc(
        start=config['data']['start_date'],
        end=config['data']['end_date'],
        interval=config['data']['interval'],
        save=True
    )
    print(f"✓ Loaded {len(df_raw)} rows")
    
    # Step 2: Clean data
    print("\\n[2/9] Cleaning data...")
    df_clean = clean_btc_data(df_raw, vol_window=5)
    df_clean.to_csv(config['paths']['clean_data'], index=False)
    print(f"✓ Cleaned data saved to {config['paths']['clean_data']}")
    
    # Step 3: Build features
    print("\\n[3/9] Building and scaling features...")
    df_features, df_scaled, scaler = build_and_scale_features(
        df_clean, 
        config['features']
    )
    print(f"✓ Created {df_features.shape[1]} features")
    
    # Step 4: Train HMM
    print("\\n[4/9] Training HMM model...")
    hmm_model, hmm_labels = train_hmm_model(
        df_scaled,
        n_states=config['models']['hmm']['n_states'],
        config=config['models']['hmm']
    )
    print(f"✓ HMM trained with {config['models']['hmm']['n_states']} states")
    
    # Step 5: Train GMM
    print("\\n[5/9] Training GMM baseline...")
    gmm_model, gmm_labels = train_gmm_model(
        df_scaled,
        n_components=config['models']['gmm']['n_components'],
        config=config['models']['gmm']
    )
    print(f"✓ GMM trained with {config['models']['gmm']['n_components']} components")
    
    # Step 6: Change Point Detection
    print("\\n[6/9] Detecting change points...")
    cpd_points = detect_change_points(
        df_scaled,
        config=config['models']['cpd']
    )
    print(f"✓ Detected {len(cpd_points)} change points")
    
    # Step 7: Evaluate performance
    print("\\n[7/9] Evaluating regime performance...")
    performance_df = evaluate_performance_by_regime(
        df_clean,
        hmm_labels,
        config['evaluation']
    )
    print(f"✓ Performance metrics calculated")
    
    # Step 8: Compare models
    print("\\n[8/9] Comparing models...")
    comparison = compare_models(hmm_labels, gmm_labels, df_clean)
    print(f"✓ Model comparison complete")
    
    # Step 9: Generate visualizations
    print("\\n[9/9] Generating visualizations...")
    plot_regimes(df_clean, hmm_labels, config['paths']['plots_dir'])
    plot_cpd(df_clean, cpd_points, config['paths']['plots_dir'])
    plot_feature_distributions(df_features, hmm_labels, config['paths']['plots_dir'])
    print(f"✓ Plots saved to {config['paths']['plots_dir']}")
    
    print("\\n" + "=" * 60)
    print("✓ PIPELINE COMPLETE!")
    print("=" * 60)
    print(f"\\nResults saved to: {config['paths']['results_dir']}")
    print("\\nOutputs:")
    print(f"  - Models: {config['paths']['models_dir']}")
    print(f"  - Tables: {config['paths']['tables_dir']}")
    print(f"  - Plots: {config['paths']['plots_dir']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run crypto regime detection pipeline')
    parser.add_argument('--config', default='config.yaml', help='Path to config file')
    args = parser.parse_args()
    
    main(args.config)