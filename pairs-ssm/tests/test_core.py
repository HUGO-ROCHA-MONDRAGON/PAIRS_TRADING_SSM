"""
Quick validation tests for the pairs_ssm package.

Run with: python -m pytest tests/ -v
"""

import pytest
import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestKalmanFilter:
    """Tests for Kalman filter implementations."""
    
    def test_linear_kalman_filter(self):
        """Test standard Kalman filter on synthetic data."""
        from pairs_ssm.models.params import ModelParams
        from pairs_ssm.filtering.kalman_linear import kalman_filter
        
        # Generate synthetic data
        np.random.seed(42)
        n = 200
        mu, phi, q, r = 0.0, 0.9, 0.01, 0.05
        
        # Simulate AR(1) + noise
        x_true = np.zeros(n)
        x_true[0] = mu
        for t in range(1, n):
            x_true[t] = mu + phi * (x_true[t-1] - mu) + np.sqrt(q) * np.random.randn()
        y = x_true + np.sqrt(r) * np.random.randn(n)
        
        # Create params object (theta0 = mu*(1-phi), theta1 = phi)
        params = ModelParams(
            theta0=mu * (1 - phi),
            theta1=phi,
            q_base=q,
            r=r,
        )
        
        # Run filter
        ll, x_filt, _, P_filt, _ = kalman_filter(y, params)
        
        # Checks
        assert x_filt.shape == (n,)
        assert P_filt.shape == (n,)
        assert np.isfinite(ll)
        assert ll < 0  # Log-likelihood is negative
        
        # Filtered estimate should be smoother than observations
        assert np.std(x_filt) < np.std(y)
    
    def test_kalman_smoother(self):
        """Test Kalman smoother."""
        from pairs_ssm.models.params import ModelParams
        from pairs_ssm.filtering.kalman_linear import kalman_filter, kalman_smoother
        
        np.random.seed(42)
        n = 100
        mu, phi, q, r = 0.5, 0.85, 0.02, 0.1
        
        x_true = np.zeros(n)
        x_true[0] = mu
        for t in range(1, n):
            x_true[t] = mu + phi * (x_true[t-1] - mu) + np.sqrt(q) * np.random.randn()
        y = x_true + np.sqrt(r) * np.random.randn(n)
        
        params = ModelParams(
            theta0=mu * (1 - phi),
            theta1=phi,
            q_base=q,
            r=r,
        )
        
        x_smooth, P_smooth = kalman_smoother(y, params)
        
        assert x_smooth.shape == (n,)
        assert P_smooth.shape == (n,)
        
        # Get filtered for comparison
        _, x_filt, _, P_filt, _ = kalman_filter(y, params)
        
        # On average, smoothed variance <= filtered variance
        assert np.mean(P_smooth) <= np.mean(P_filt)


class TestMLE:
    """Tests for MLE parameter estimation."""
    
    def test_fit_model_I(self):
        """Test MLE fitting for Model I."""
        from pairs_ssm.filtering.mle import fit_model
        
        np.random.seed(42)
        n = 300
        true_mu, true_phi, true_q, true_r = 0.2, 0.92, 0.005, 0.02
        
        x = np.zeros(n)
        x[0] = true_mu
        for t in range(1, n):
            x[t] = true_mu + true_phi * (x[t-1] - true_mu) + np.sqrt(true_q) * np.random.randn()
        y = pd.Series(x + np.sqrt(true_r) * np.random.randn(n))
        
        result = fit_model(y, model_type="model_I", verbose=False)
        
        params = result.params
        
        # Check params are within valid range
        assert 0.0 < params.theta1 < 1.0  # AR coefficient (phi)
        assert params.q_base > 0  # State noise variance
        assert params.r > 0  # Observation noise variance


class TestStrategies:
    """Tests for trading strategies."""
    
    def test_strategy_A(self):
        """Test Strategy A signals."""
        from pairs_ssm.trading import strategy_A_signals
        
        # Simple spread
        spread = pd.Series([0.0, 0.1, 0.2, 0.15, 0.05, 0.0, -0.1, -0.05, 0.0])
        U, L, C = 0.15, -0.15, 0.0
        
        signals = strategy_A_signals(spread, U, L, C)
        
        assert len(signals) == len(spread)
        assert set(signals.unique()).issubset({-1, 0, 1})
    
    def test_strategy_C(self):
        """Test Strategy C signals."""
        from pairs_ssm.trading import strategy_C_signals
        
        # Spread that crosses boundaries
        spread = pd.Series([0.0, 0.1, 0.2, 0.18, 0.1, 0.0, -0.1, -0.2, -0.15, 0.0])
        U, L, C = 0.15, -0.15, 0.0
        
        signals = strategy_C_signals(spread, U, L, C)
        
        assert len(signals) == len(spread)
        assert set(signals.unique()).issubset({-1, 0, 1})


class TestBacktest:
    """Tests for backtesting."""
    
    def test_backtest_signals(self):
        """Test P&L computation."""
        from pairs_ssm.trading import backtest_signals
        
        np.random.seed(42)
        n = 100
        
        # Random log prices
        log_p1 = pd.Series(np.cumsum(0.01 * np.random.randn(n)))
        log_p2 = pd.Series(np.cumsum(0.01 * np.random.randn(n)))
        
        # Simple alternating signals
        signals = pd.Series([1, 0, -1, 0] * 25)
        
        result = backtest_signals(signals, log_p1, log_p2, gamma=1.0, cost_bp=20)
        
        assert len(result.pnl) == n
        assert result.n_trades >= 0
        assert hasattr(result, 'sharpe_ratio')


class TestDataLoading:
    """Tests for data loading (if data file exists)."""
    
    def test_compute_spread(self):
        """Test spread computation."""
        from pairs_ssm.data.transforms import compute_spread
        
        np.random.seed(42)
        n = 100
        
        # Generate correlated log prices (so gamma is positive)
        base = np.cumsum(0.01 * np.random.randn(n))
        log_p1 = pd.Series(base + 0.005 * np.random.randn(n) + 4.0)
        log_p2 = pd.Series(base * 0.8 + 0.005 * np.random.randn(n) + 4.0)
        
        spread_data = compute_spread(log_p1, log_p2)
        
        assert len(spread_data.spread) == n
        # Gamma can be negative for uncorrelated series, so just check it exists
        assert spread_data.gamma != 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
