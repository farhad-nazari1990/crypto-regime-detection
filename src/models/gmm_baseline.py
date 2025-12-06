"""
Gaussian Mixture Model baseline for comparison
"""

import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
import joblib
from pathlib import Path

def train_gmm_model(df, n_components=4, config=None):
    """
    Train GMM baseline model
    
    Parameters:
    -----------
    df : pd.DataFrame
        Scaled feature dataframe
    n_components : int
        Number of Gaussian components
    config : dict
        GMM configuration
        
    Returns:
    --------
    tuple
        (trained model, cluster labels)
    """
    
    if config is None:
        config = {
            'n_init': 10,
            'random_state': 42,
            'covariance_type': 'full'
        }
    
    print(f"Training GMM with {n_components} components...")
    
    # Prepare data
    feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [col for col in feature_cols if 'date' not in col.lower()]
    
    X = df[feature_cols].values
    print(f"  Using {len(feature_cols)} features")
    print(f"  Training samples: {len(X)}")
    
    # Create and train model
    model = GaussianMixture(
        n_components=n_components,
        covariance_type=config.get('covariance_type', 'full'),
        n_init=config.get('n_init', 10),
        random_state=config.get('random_state', 42),
        max_iter=100,
        verbose=0
    )
    
    # Fit model
    model.fit(X)
    
    # Predict clusters
    labels = model.predict(X)
    
    print(f"✓ GMM training complete")
    print(f"  BIC: {model.bic(X):.2f}")
    print(f"  AIC: {model.aic(X):.2f}")
    print(f"  Converged: {model.converged_}")
    
    # Save model
    Path('results/models').mkdir(parents=True, exist_ok=True)
    model_path = 'results/models/gmm_model.pkl'
    joblib.dump(model, model_path)
    print(f"✓ Saved model to {model_path}")
    
    # Create labels dataframe
    labels_df = pd.DataFrame({
        'date': df['date'],
        'cluster': labels,
        'close': df['close'] if 'close' in df.columns else None
    })
    
    # Save labels
    Path('results/tables').mkdir(parents=True, exist_ok=True)
    labels_path = 'results/tables/gmm_labels.csv'
    labels_df.to_csv(labels_path, index=False)
    print(f"✓ Saved labels to {labels_path}")
    
    # Print cluster distribution
    unique, counts = np.unique(labels, return_counts=True)
    print("\\n  Cluster distribution:")
    for cluster, count in zip(unique, counts):
        print(f"    Cluster {cluster}: {count} days ({count/len(labels)*100:.1f}%)")
    
    return model, labels

def load_gmm_model(model_path='results/models/gmm_model.pkl'):
    """Load trained GMM model"""
    return joblib.load(model_path)

if __name__ == "__main__":
    # Load scaled features
    df = pd.read_csv('data/processed/features_scaled.csv')
    df['date'] = pd.to_datetime(df['date'])
    
    # Train model
    model, labels = train_gmm_model(df, n_components=4)
    
    print("\\nModel parameters:")
    print(f"  Means shape: {model.means_.shape}")
    print(f"  Covariances shape: {model.covariances_.shape}")
    print(f"  Weights: {model.weights_}")
