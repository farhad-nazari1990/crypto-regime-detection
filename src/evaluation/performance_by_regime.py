"""
Evaluate trading performance metrics by regime
"""

import pandas as pd
import numpy as np
from pathlib import Path

def calculate_sharpe_ratio(returns, risk_free_rate=0.0, periods_per_year=365):
    """
    Calculate annualized Sharpe ratio
    
    Parameters:
    -----------
    returns : pd.Series
        Series of returns
    risk_free_rate : float
        Annual risk-free rate
    periods_per_year : int
        Number of trading periods per year
        
    Returns:
    --------
    float
        Annualized Sharpe ratio
    """
    excess_returns = returns - risk_free_rate / periods_per_year
    if len(excess_returns) < 2:
        return np.nan
    
    if excess_returns.std() == 0:
        return np.nan
    
    return np.sqrt(periods_per_year) * excess_returns.mean() / excess_returns.std()

def calculate_max_drawdown(prices):
    """
    Calculate maximum drawdown
    
    Parameters:
    -----------
    prices : pd.Series
        Series of prices
        
    Returns:
    --------
    float
        Maximum drawdown (negative value)
    """
    if len(prices) < 2:
        return np.nan
    
    cumulative = (1 + prices.pct_change()).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    return drawdown.min()

def calculate_regime_metrics(df, state, eval_config):
    """
    Calculate metrics for a single regime
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with returns and prices
    state : int
        State/regime number
    eval_config : dict
        Evaluation configuration
        
    Returns:
    --------
    dict
        Dictionary of metrics
    """
    
    state_data = df[df['state'] == state].copy()
    
    if len(state_data) == 0:
        return None
    
    metrics = {
        'state': state,
        'n_days': len(state_data),
        'pct_days': len(state_data) / len(df) * 100
    }
    
    # Return metrics
    if 'return' in state_data.columns:
        metrics['mean_daily_return'] = state_data['return'].mean()
        metrics['median_daily_return'] = state_data['return'].median()
        metrics['total_return'] = (1 + state_data['return']).prod() - 1
        
        # Annualized return
        n_days = len(state_data)
        if n_days > 0:
            metrics['annualized_return'] = (1 + metrics['total_return']) ** (365 / n_days) - 1
        else:
            metrics['annualized_return'] = 0
    
    # Volatility metrics
    if 'return' in state_data.columns:
        metrics['volatility'] = state_data['return'].std()
        metrics['annualized_volatility'] = metrics['volatility'] * np.sqrt(365)
    
    # Sharpe ratio
    if 'return' in state_data.columns and len(state_data) > 1:
        metrics['sharpe_ratio'] = calculate_sharpe_ratio(
            state_data['return'],
            eval_config.get('risk_free_rate', 0.0),
            eval_config.get('trading_days_per_year', 365)
        )
    else:
        metrics['sharpe_ratio'] = np.nan
    
    # Max drawdown
    if 'close' in state_data.columns and len(state_data) > 1:
        metrics['max_drawdown'] = calculate_max_drawdown(state_data['close'])
    else:
        metrics['max_drawdown'] = np.nan
    
    # Win rate
    if 'return' in state_data.columns:
        metrics['win_rate'] = (state_data['return'] > 0).mean()
    
    # Average gain/loss
    if 'return' in state_data.columns:
        gains = state_data[state_data['return'] > 0]['return']
        losses = state_data[state_data['return'] < 0]['return']
        metrics['avg_gain'] = gains.mean() if len(gains) > 0 else 0
        metrics['avg_loss'] = losses.mean() if len(losses) > 0 else 0
        
        if metrics['avg_loss'] != 0:
            metrics['gain_loss_ratio'] = abs(metrics['avg_gain'] / metrics['avg_loss'])
        else:
            metrics['gain_loss_ratio'] = np.nan
    
    return metrics

def evaluate_performance_by_regime(df, states, eval_config):
    """
    Evaluate performance for all regimes
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with price and return data
    states : np.ndarray or pd.Series
        State labels
    eval_config : dict
        Evaluation configuration
        
    Returns:
    --------
    pd.DataFrame
        Performance metrics by regime
    """
    
    print("Evaluating performance by regime...")
    
    # Add states to dataframe with length matching
    df_eval = df.copy()
    
    # Convert states to numpy array if it's a Series
    if isinstance(states, pd.Series):
        states = states.values
    
    # ✅ Fix: Match lengths of df and states
    if len(states) != len(df_eval):
        print(f"  ⚠️ Length mismatch detected:")
        print(f"     DataFrame length: {len(df_eval)}")
        print(f"     States length: {len(states)}")
        
        if len(states) < len(df_eval):
            # If states is shorter, take the last N rows of df
            # This matches the behavior where feature engineering removes early rows
            df_eval = df_eval.tail(len(states)).reset_index(drop=True)
            print(f"  ✓ Adjusted DataFrame to {len(df_eval)} rows to match states")
        else:
            # If states is longer (shouldn't happen), truncate states
            states = states[:len(df_eval)]
            print(f"  ✓ Adjusted states to {len(states)} elements to match DataFrame")
    
    # Now lengths match
    df_eval['state'] = states
    
    # Calculate metrics for each state
    all_metrics = []
    unique_states = sorted(df_eval['state'].unique())
    
    for state in unique_states:
        print(f"  Analyzing State {state}...")
        metrics = calculate_regime_metrics(df_eval, state, eval_config)
        if metrics:
            all_metrics.append(metrics)
    
    # Create results dataframe
    performance_df = pd.DataFrame(all_metrics)
    
    # Save results
    Path('results/tables').mkdir(parents=True, exist_ok=True)
    output_path = 'results/tables/performance_per_regime.csv'
    performance_df.to_csv(output_path, index=False)
    print(f"✓ Saved performance metrics to {output_path}")
    
    # Print summary
    print("\n  Performance Summary:")
    if len(performance_df) > 0:
        display_cols = ['state', 'n_days', 'mean_daily_return', 'volatility', 'sharpe_ratio']
        # Only show columns that exist
        available_cols = [col for col in display_cols if col in performance_df.columns]
        if available_cols:
            print(performance_df[available_cols].to_string(index=False))
        else:
            print(performance_df.to_string(index=False))
    else:
        print("  No performance data available")
    
    return performance_df

if __name__ == "__main__":
    # Load data
    df = pd.read_csv('data/processed/btc_clean.csv')
    df['date'] = pd.to_datetime(df['date'])
    
    labels_df = pd.read_csv('results/tables/hmm_labels.csv')
    
    # Evaluate
    eval_config = {
        'risk_free_rate': 0.0,
        'trading_days_per_year': 365
    }
    
    performance = evaluate_performance_by_regime(df, labels_df['state'].values, eval_config)
    
    print("\nFull Performance Metrics:")
    print(performance)