"""
Hidden Markov Model for regime detection
"""

import pandas as pd
import numpy as np
from hmmlearn import hmm
import joblib
from pathlib import Path

def prepare_hmm_data(df, feature_cols=None):
    """
    Prepare data for HMM training
    
    Parameters:
    -----------
    df : pd.DataFrame
        Scaled feature dataframe
    feature_cols : list or None
        List of feature columns to use
        
    Returns:
    --------
    np.ndarray
        Feature matrix for HMM
    """
    
    if feature_cols is None:
        # Use all numeric columns except date
        feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in feature_cols if 'date' not in col.lower()]
    
    X = df[feature_cols].values
    
    # Check for NaN or inf
    if np.isnan(X).any() or np.isinf(X).any():
        raise ValueError("Data contains NaN or inf values")
    
    return X, feature_cols

def train_hmm_model(df, n_states=4, config=None):
    """
    Train Gaussian HMM model
    
    Parameters:
    -----------
    df : pd.DataFrame
        Scaled feature dataframe
    n_states : int
        Number of hidden states
    config : dict
        HMM configuration
        
    Returns:
    --------
    tuple
        (trained model, state labels)
    """
    
    if config is None:
        config = {
            'n_iter': 100,
            'random_state': 42,
            'covariance_type': 'full'
        }
    
    print(f"Training HMM with {n_states} states...")
    
    # Prepare data
    X, feature_cols = prepare_hmm_data(df)
    print(f"  Using {len(feature_cols)} features")
    print(f"  Training samples: {len(X)}")
    
    # Create and train model
    model = hmm.GaussianHMM(
        n_components=n_states,
        covariance_type=config.get('covariance_type', 'full'),
        n_iter=config.get('n_iter', 100),
        random_state=config.get('random_state', 42),
        verbose=False
    )
    
    # Fit model
    model.fit(X)
    
    # Predict states
    states = model.predict(X)
    
    print(f"✓ HMM training complete")
    print(f"  Log-likelihood: {model.score(X):.2f}")
    print(f"  Converged: {model.monitor_.converged}")
    
    # Save model
    Path('results/models').mkdir(parents=True, exist_ok=True)
    model_path = 'results/models/hmm_model.pkl'
    joblib.dump(model, model_path)
    print(f"✓ Saved model to {model_path}")
    
    # Create labels dataframe
    labels_df = pd.DataFrame({
        'date': df['date'],
        'state': states,
        'close': df['close'] if 'close' in df.columns else None
    })
    
    # Save labels
    Path('results/tables').mkdir(parents=True, exist_ok=True)
    labels_path = 'results/tables/hmm_labels.csv'
    labels_df.to_csv(labels_path, index=False)
    print(f"✓ Saved labels to {labels_path}")
    
    # Save transition matrix
    transition_df = pd.DataFrame(
        model.transmat_,
        columns=[f'To_State_{i}' for i in range(n_states)],
        index=[f'From_State_{i}' for i in range(n_states)]
    )
    trans_path = 'results/tables/hmm_transition_matrix.csv'
    transition_df.to_csv(trans_path)
    print(f"✓ Saved transition matrix to {trans_path}")
    
    # Print state distribution
    unique, counts = np.unique(states, return_counts=True)
    print("\\n  State distribution:")
    for state, count in zip(unique, counts):
        print(f"    State {state}: {count} days ({count/len(states)*100:.1f}%)")
    
    return model, states

def load_hmm_model(model_path='results/models/hmm_model.pkl'):
    """Load trained HMM model"""
    return joblib.load(model_path)

if __name__ == "__main__":
    # Load scaled features
    df = pd.read_csv('data/processed/features_scaled.csv')
    df['date'] = pd.to_datetime(df['date'])
    
    # Train model
    model, states = train_hmm_model(df, n_states=4)
    
    print("\\nModel parameters:")
    print(f"  Means shape: {model.means_.shape}")
    print(f"  Covariances shape: {model.covars_.shape}")