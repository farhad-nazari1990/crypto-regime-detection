"""
Additional evaluation metrics
"""

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

def calculate_clustering_metrics(X, labels):
    """
    Calculate clustering quality metrics
    
    Parameters:
    -----------
    X : np.ndarray
        Feature matrix
    labels : np.ndarray
        Cluster/state labels
        
    Returns:
    --------
    dict
        Clustering metrics
    """
    
    metrics = {}
    
    # Silhouette score
    try:
        metrics['silhouette_score'] = silhouette_score(X, labels)
    except:
        metrics['silhouette_score'] = np.nan
    
    # Calinski-Harabasz score
    try:
        metrics['calinski_harabasz_score'] = calinski_harabasz_score(X, labels)
    except:
        metrics['calinski_harabasz_score'] = np.nan
    
    # Davies-Bouldin score
    try:
        metrics['davies_bouldin_score'] = davies_bouldin_score(X, labels)
    except:
        metrics['davies_bouldin_score'] = np.nan
    
    return metrics

def calculate_state_entropy(labels):
    """
    Calculate entropy of state distribution
    
    Parameters:
    -----------
    labels : np.ndarray
        State labels
        
    Returns:
    --------
    float
        Entropy value
    """
    
    unique, counts = np.unique(labels, return_counts=True)
    probabilities = counts / len(labels)
    entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
    
    return entropy

def calculate_transition_stability(states):
    """
    Calculate stability of state transitions
    
    Parameters:
    -----------
    states : np.ndarray
        State sequence
        
    Returns:
    --------
    dict
        Transition statistics
    """
    
    # Count transitions
    transitions = 0
    for i in range(len(states) - 1):
        if states[i] != states[i+1]:
            transitions += 1
    
    # Calculate average duration per state
    durations = []
    current_state = states[0]
    duration = 1
    
    for i in range(1, len(states)):
        if states[i] == current_state:
            duration += 1
        else:
            durations.append(duration)
            current_state = states[i]
            duration = 1
    durations.append(duration)
    
    return {
        'n_transitions': transitions,
        'transition_rate': transitions / (len(states) - 1),
        'avg_duration': np.mean(durations),
        'median_duration': np.median(durations),
        'max_duration': np.max(durations),
        'min_duration': np.min(durations)
    }

def evaluate_model_quality(X, labels, model_type='hmm'):
    """
    Comprehensive model quality evaluation
    
    Parameters:
    -----------
    X : np.ndarray
        Feature matrix
    labels : np.ndarray
        Predicted labels
    model_type : str
        Type of model ('hmm' or 'gmm')
        
    Returns:
    --------
    dict
        Quality metrics
    """
    
    print(f"Evaluating {model_type.upper()} model quality...")
    
    metrics = {}
    
    # Clustering metrics
    clustering_metrics = calculate_clustering_metrics(X, labels)
    metrics.update(clustering_metrics)
    print(f"  Silhouette Score: {metrics['silhouette_score']:.4f}")
    
    # State distribution
    unique_states = len(np.unique(labels))
    metrics['n_states'] = unique_states
    metrics['entropy'] = calculate_state_entropy(labels)
    print(f"  Number of states: {unique_states}")
    print(f"  Entropy: {metrics['entropy']:.4f}")
    
    # Transition statistics
    transition_stats = calculate_transition_stability(labels)
    metrics.update(transition_stats)
    print(f"  Transition rate: {metrics['transition_rate']:.4f}")
    print(f"  Avg state duration: {metrics['avg_duration']:.1f} days")
    
    return metrics

if __name__ == "__main__":
    # Load data
    df_scaled = pd.read_csv('data/processed/features_scaled.csv')
    hmm_labels = pd.read_csv('results/tables/hmm_labels.csv')
    
    # Prepare feature matrix
    feature_cols = df_scaled.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [col for col in feature_cols if 'date' not in col.lower()]
    X = df_scaled[feature_cols].values
    
    # Evaluate
    metrics = evaluate_model_quality(X, hmm_labels['state'].values, 'hmm')
    
    print("\\nAll metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")