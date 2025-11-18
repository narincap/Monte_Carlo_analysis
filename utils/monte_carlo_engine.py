"""
Monte Carlo Simulation Engine
Implements Geometric Brownian Motion (GBM) for portfolio simulation.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from scipy.linalg import cholesky


class MonteCarloEngine:
    """Monte Carlo simulation engine for portfolio analysis."""

    def __init__(self, mean_returns: pd.Series, cov_matrix: pd.DataFrame,
                 initial_prices: pd.Series):
        """
        Initialize Monte Carlo engine.

        Args:
            mean_returns: Mean daily returns for each asset
            cov_matrix: Covariance matrix of returns
            initial_prices: Starting prices for each asset
        """
        self.mean_returns = mean_returns
        self.cov_matrix = cov_matrix
        self.initial_prices = initial_prices
        self.tickers = list(mean_returns.index)
        self.num_assets = len(self.tickers)

        # Precompute Cholesky decomposition for correlated returns
        try:
            self.cholesky_matrix = cholesky(cov_matrix.values, lower=True)
        except np.linalg.LinAlgError:
            # If Cholesky fails, use eigenvalue decomposition (slower but more robust)
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix.values)
            eigenvalues = np.maximum(eigenvalues, 1e-10)  # Ensure positive semi-definite
            self.cholesky_matrix = eigenvectors @ np.diag(np.sqrt(eigenvalues))

    def simulate_gbm_paths(self, num_simulations: int, time_horizon: int,
                          initial_investment: float = 10000,
                          weights: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate portfolio paths using Geometric Brownian Motion.

        The GBM formula for each asset:
        S(t+1) = S(t) * exp((μ - σ²/2)Δt + σ√Δt * Z)

        where:
        - μ = mean return (drift)
        - σ = volatility (standard deviation)
        - Δt = time step (1 day)
        - Z = random normal variable

        Args:
            num_simulations: Number of simulation paths
            time_horizon: Number of days to simulate
            initial_investment: Starting portfolio value
            weights: Portfolio weights (default: equal weight)

        Returns:
            Tuple of (asset_paths, portfolio_values)
            - asset_paths: shape (num_simulations, time_horizon+1, num_assets)
            - portfolio_values: shape (num_simulations, time_horizon+1)
        """
        if weights is None:
            weights = np.array([1.0 / self.num_assets] * self.num_assets)
        else:
            weights = np.array(weights)

        # Ensure weights sum to 1
        weights = weights / weights.sum()

        # Initialize arrays
        asset_paths = np.zeros((num_simulations, time_horizon + 1, self.num_assets))
        portfolio_values = np.zeros((num_simulations, time_horizon + 1))

        # Calculate initial shares for each asset
        initial_allocation = initial_investment * weights
        shares = initial_allocation / self.initial_prices.values

        # Set initial values
        asset_paths[:, 0, :] = self.initial_prices.values
        portfolio_values[:, 0] = initial_investment

        # Convert mean returns and volatilities to numpy arrays
        mu = self.mean_returns.values
        sigma = np.sqrt(np.diag(self.cov_matrix.values))

        # Time step (daily)
        dt = 1.0

        # Simulate paths
        for t in range(1, time_horizon + 1):
            # Generate correlated random variables
            # Shape: (num_simulations, num_assets)
            random_normals = np.random.standard_normal((num_simulations, self.num_assets))
            correlated_randoms = random_normals @ self.cholesky_matrix.T

            # GBM formula: S(t+1) = S(t) * exp((μ - σ²/2)dt + σ√dt * Z)
            drift = (mu - 0.5 * sigma**2) * dt
            diffusion = sigma * np.sqrt(dt) * correlated_randoms

            # Calculate new prices
            asset_paths[:, t, :] = asset_paths[:, t-1, :] * np.exp(drift + diffusion)

            # Calculate portfolio value based on fixed shares (no rebalancing)
            portfolio_values[:, t] = np.sum(asset_paths[:, t, :] * shares, axis=1)

        return asset_paths, portfolio_values

    def simulate_with_rebalancing(self, num_simulations: int, time_horizon: int,
                                  initial_investment: float = 10000,
                                  weights: Optional[np.ndarray] = None,
                                  rebalance_frequency: int = 21) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate portfolio paths with periodic rebalancing.

        Args:
            num_simulations: Number of simulation paths
            time_horizon: Number of days to simulate
            initial_investment: Starting portfolio value
            weights: Portfolio weights (default: equal weight)
            rebalance_frequency: Days between rebalancing (default: 21 ~monthly)

        Returns:
            Tuple of (asset_paths, portfolio_values)
        """
        if weights is None:
            weights = np.array([1.0 / self.num_assets] * self.num_assets)
        else:
            weights = np.array(weights)

        weights = weights / weights.sum()

        # Initialize arrays
        asset_paths = np.zeros((num_simulations, time_horizon + 1, self.num_assets))
        portfolio_values = np.zeros((num_simulations, time_horizon + 1))

        # Set initial values
        asset_paths[:, 0, :] = self.initial_prices.values
        portfolio_values[:, 0] = initial_investment

        # Calculate initial shares
        initial_allocation = initial_investment * weights
        shares = np.tile(initial_allocation / self.initial_prices.values, (num_simulations, 1))

        # Convert mean returns and volatilities
        mu = self.mean_returns.values
        sigma = np.sqrt(np.diag(self.cov_matrix.values))
        dt = 1.0

        # Simulate paths with rebalancing
        for t in range(1, time_horizon + 1):
            # Generate correlated random variables
            random_normals = np.random.standard_normal((num_simulations, self.num_assets))
            correlated_randoms = random_normals @ self.cholesky_matrix.T

            # GBM formula
            drift = (mu - 0.5 * sigma**2) * dt
            diffusion = sigma * np.sqrt(dt) * correlated_randoms

            # Calculate new prices
            asset_paths[:, t, :] = asset_paths[:, t-1, :] * np.exp(drift + diffusion)

            # Calculate portfolio value
            portfolio_values[:, t] = np.sum(asset_paths[:, t, :] * shares, axis=1)

            # Rebalance if needed
            if t % rebalance_frequency == 0 and t < time_horizon:
                # Rebalance to target weights
                target_allocation = portfolio_values[:, t:t+1] * weights
                shares = target_allocation / asset_paths[:, t, :]

        return asset_paths, portfolio_values

    def calculate_individual_asset_returns(self, asset_paths: np.ndarray) -> np.ndarray:
        """
        Calculate returns for individual assets.

        Args:
            asset_paths: Asset price paths from simulation

        Returns:
            Array of returns for each asset (num_simulations, num_assets)
        """
        initial_prices = asset_paths[:, 0, :]
        final_prices = asset_paths[:, -1, :]

        returns = (final_prices - initial_prices) / initial_prices
        return returns

    def calculate_portfolio_returns(self, portfolio_values: np.ndarray) -> np.ndarray:
        """
        Calculate portfolio returns.

        Args:
            portfolio_values: Portfolio value paths from simulation

        Returns:
            Array of returns (num_simulations,)
        """
        initial_value = portfolio_values[:, 0]
        final_value = portfolio_values[:, -1]

        returns = (final_value - initial_value) / initial_value
        return returns


class MonteCarloOptimizer:
    """Utility class for portfolio optimization within Monte Carlo framework."""

    @staticmethod
    def equal_weights(num_assets: int) -> np.ndarray:
        """Generate equal weights."""
        return np.array([1.0 / num_assets] * num_assets)

    @staticmethod
    def inverse_volatility_weights(volatilities: np.ndarray) -> np.ndarray:
        """
        Generate inverse volatility weights.

        Args:
            volatilities: Array of asset volatilities

        Returns:
            Normalized weights inversely proportional to volatility
        """
        inv_vol = 1.0 / volatilities
        weights = inv_vol / inv_vol.sum()
        return weights

    @staticmethod
    def random_weights(num_assets: int) -> np.ndarray:
        """Generate random weights that sum to 1."""
        weights = np.random.random(num_assets)
        return weights / weights.sum()

    @staticmethod
    def validate_weights(weights: np.ndarray, num_assets: int) -> bool:
        """
        Validate portfolio weights.

        Args:
            weights: Array of weights
            num_assets: Expected number of assets

        Returns:
            True if valid, False otherwise
        """
        if len(weights) != num_assets:
            return False

        if not np.isclose(weights.sum(), 1.0, atol=1e-3):
            return False

        if np.any(weights < 0):
            return False

        return True


# Convenience function
def run_monte_carlo_simulation(mean_returns: pd.Series, cov_matrix: pd.DataFrame,
                               initial_prices: pd.Series, num_simulations: int = 10000,
                               time_horizon: int = 252, initial_investment: float = 10000,
                               weights: Optional[List[float]] = None,
                               rebalancing: bool = False) -> dict:
    """
    Run Monte Carlo simulation and return results.

    Args:
        mean_returns: Mean daily returns
        cov_matrix: Covariance matrix
        initial_prices: Starting prices
        num_simulations: Number of paths
        time_horizon: Days to simulate
        initial_investment: Starting value
        weights: Portfolio weights
        rebalancing: Whether to rebalance periodically

    Returns:
        Dictionary with simulation results
    """
    engine = MonteCarloEngine(mean_returns, cov_matrix, initial_prices)

    if rebalancing:
        asset_paths, portfolio_values = engine.simulate_with_rebalancing(
            num_simulations, time_horizon, initial_investment, weights
        )
    else:
        asset_paths, portfolio_values = engine.simulate_gbm_paths(
            num_simulations, time_horizon, initial_investment, weights
        )

    portfolio_returns = engine.calculate_portfolio_returns(portfolio_values)
    asset_returns = engine.calculate_individual_asset_returns(asset_paths)

    return {
        'asset_paths': asset_paths,
        'portfolio_values': portfolio_values,
        'portfolio_returns': portfolio_returns,
        'asset_returns': asset_returns,
        'initial_investment': initial_investment,
        'time_horizon': time_horizon,
        'num_simulations': num_simulations
    }
