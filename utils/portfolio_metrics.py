"""
Portfolio Risk Metrics Calculator
Calculates VaR, CVaR, Sharpe, Sortino, and other risk-adjusted metrics.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional


class PortfolioMetrics:
    """Calculate comprehensive portfolio risk metrics."""

    def __init__(self, portfolio_values: np.ndarray, portfolio_returns: np.ndarray,
                 initial_investment: float, time_horizon: int, risk_free_rate: float = 0.045):
        """
        Initialize metrics calculator.

        Args:
            portfolio_values: Simulated portfolio values (num_simulations, time_steps)
            portfolio_returns: Final portfolio returns (num_simulations,)
            initial_investment: Starting portfolio value
            time_horizon: Number of days in simulation
            risk_free_rate: Annual risk-free rate (default: 4.5%)
        """
        self.portfolio_values = portfolio_values
        self.portfolio_returns = portfolio_returns
        self.initial_investment = initial_investment
        self.time_horizon = time_horizon
        self.risk_free_rate = risk_free_rate

        # Final portfolio values
        self.final_values = portfolio_values[:, -1]

    def calculate_var(self, confidence_level: float = 0.95) -> Dict[str, float]:
        """
        Calculate Value at Risk (VaR).

        VaR represents the maximum loss at a given confidence level.
        For example, 95% VaR means there's a 5% chance of losing more than this amount.

        Args:
            confidence_level: Confidence level (e.g., 0.95 for 95%)

        Returns:
            Dictionary with VaR in dollar and percentage terms
        """
        alpha = 1 - confidence_level
        var_percentile = np.percentile(self.portfolio_returns, alpha * 100)

        var_dollar = var_percentile * self.initial_investment

        return {
            'VaR_percent': var_percentile * 100,
            'VaR_dollar': var_dollar,
            'confidence_level': confidence_level
        }

    def calculate_cvar(self, confidence_level: float = 0.95) -> Dict[str, float]:
        """
        Calculate Conditional Value at Risk (CVaR), also known as Expected Shortfall.

        CVaR is the average loss in the worst (1-confidence_level)% of cases.
        It provides a better measure of tail risk than VaR.

        Args:
            confidence_level: Confidence level (e.g., 0.95 for 95%)

        Returns:
            Dictionary with CVaR in dollar and percentage terms
        """
        alpha = 1 - confidence_level
        var_threshold = np.percentile(self.portfolio_returns, alpha * 100)

        # CVaR is the mean of all returns below VaR threshold
        tail_losses = self.portfolio_returns[self.portfolio_returns <= var_threshold]
        cvar_percent = tail_losses.mean()

        cvar_dollar = cvar_percent * self.initial_investment

        return {
            'CVaR_percent': cvar_percent * 100,
            'CVaR_dollar': cvar_dollar,
            'confidence_level': confidence_level
        }

    def calculate_sharpe_ratio(self) -> float:
        """
        Calculate Sharpe Ratio.

        Sharpe Ratio = (Portfolio Return - Risk-Free Rate) / Portfolio Volatility

        Returns:
            Annualized Sharpe Ratio
        """
        # Annualize returns and volatility
        days_per_year = 252
        annualization_factor = days_per_year / self.time_horizon

        mean_return = self.portfolio_returns.mean() * annualization_factor
        volatility = self.portfolio_returns.std() * np.sqrt(annualization_factor)

        if volatility == 0:
            return 0.0

        sharpe_ratio = (mean_return - self.risk_free_rate) / volatility
        return sharpe_ratio

    def calculate_sortino_ratio(self) -> float:
        """
        Calculate Sortino Ratio.

        Similar to Sharpe but only considers downside volatility.
        Sortino Ratio = (Portfolio Return - Risk-Free Rate) / Downside Deviation

        Returns:
            Annualized Sortino Ratio
        """
        days_per_year = 252
        annualization_factor = days_per_year / self.time_horizon

        mean_return = self.portfolio_returns.mean() * annualization_factor

        # Calculate downside deviation (only negative returns)
        downside_returns = self.portfolio_returns[self.portfolio_returns < 0]

        if len(downside_returns) == 0:
            return np.inf  # No downside risk

        downside_deviation = downside_returns.std() * np.sqrt(annualization_factor)

        if downside_deviation == 0:
            return 0.0

        sortino_ratio = (mean_return - self.risk_free_rate) / downside_deviation
        return sortino_ratio

    def calculate_max_drawdown(self) -> Dict[str, float]:
        """
        Calculate Maximum Drawdown across all simulation paths.

        Maximum Drawdown is the largest peak-to-trough decline.

        Returns:
            Dictionary with max drawdown statistics
        """
        max_drawdowns = []

        for sim in range(self.portfolio_values.shape[0]):
            path = self.portfolio_values[sim, :]

            # Calculate running maximum
            running_max = np.maximum.accumulate(path)

            # Calculate drawdown at each point
            drawdown = (path - running_max) / running_max

            # Get maximum drawdown for this path
            max_drawdowns.append(drawdown.min())

        max_drawdowns = np.array(max_drawdowns)

        return {
            'mean_max_drawdown': max_drawdowns.mean() * 100,
            'worst_drawdown': max_drawdowns.min() * 100,
            'median_drawdown': np.median(max_drawdowns) * 100,
            'percentile_95_drawdown': np.percentile(max_drawdowns, 95) * 100
        }

    def calculate_calmar_ratio(self) -> float:
        """
        Calculate Calmar Ratio.

        Calmar Ratio = Annualized Return / Maximum Drawdown

        Returns:
            Calmar Ratio
        """
        days_per_year = 252
        annualization_factor = days_per_year / self.time_horizon

        mean_return = self.portfolio_returns.mean() * annualization_factor

        max_dd = self.calculate_max_drawdown()
        abs_worst_drawdown = abs(max_dd['worst_drawdown'] / 100)

        if abs_worst_drawdown == 0:
            return np.inf

        calmar_ratio = mean_return / abs_worst_drawdown
        return calmar_ratio

    def calculate_return_statistics(self) -> Dict[str, float]:
        """
        Calculate comprehensive return statistics.

        Returns:
            Dictionary with various return metrics
        """
        days_per_year = 252
        annualization_factor = days_per_year / self.time_horizon

        final_returns_pct = self.portfolio_returns * 100
        annualized_returns = self.portfolio_returns * annualization_factor * 100

        return {
            'mean_return': final_returns_pct.mean(),
            'median_return': np.median(final_returns_pct),
            'std_return': final_returns_pct.std(),
            'min_return': final_returns_pct.min(),
            'max_return': final_returns_pct.max(),
            'percentile_5': np.percentile(final_returns_pct, 5),
            'percentile_25': np.percentile(final_returns_pct, 25),
            'percentile_75': np.percentile(final_returns_pct, 75),
            'percentile_95': np.percentile(final_returns_pct, 95),
            'annualized_mean': annualized_returns.mean(),
            'annualized_std': annualized_returns.std(),
            'probability_profit': (self.portfolio_returns > 0).sum() / len(self.portfolio_returns) * 100
        }

    def calculate_value_statistics(self) -> Dict[str, float]:
        """
        Calculate portfolio value statistics.

        Returns:
            Dictionary with value-based metrics
        """
        return {
            'initial_value': self.initial_investment,
            'mean_final_value': self.final_values.mean(),
            'median_final_value': np.median(self.final_values),
            'std_final_value': self.final_values.std(),
            'min_final_value': self.final_values.min(),
            'max_final_value': self.final_values.max(),
            'percentile_5_value': np.percentile(self.final_values, 5),
            'percentile_25_value': np.percentile(self.final_values, 25),
            'percentile_75_value': np.percentile(self.final_values, 75),
            'percentile_95_value': np.percentile(self.final_values, 95)
        }

    def get_comprehensive_metrics(self, confidence_level: float = 0.95) -> Dict:
        """
        Calculate all metrics and return comprehensive summary.

        Args:
            confidence_level: Confidence level for VaR/CVaR

        Returns:
            Dictionary with all metrics organized by category
        """
        var = self.calculate_var(confidence_level)
        cvar = self.calculate_cvar(confidence_level)
        return_stats = self.calculate_return_statistics()
        value_stats = self.calculate_value_statistics()
        drawdown = self.calculate_max_drawdown()

        metrics = {
            'risk_metrics': {
                'VaR_%': var['VaR_percent'],
                'VaR_$': var['VaR_dollar'],
                'CVaR_%': cvar['CVaR_percent'],
                'CVaR_$': cvar['CVaR_dollar'],
                'confidence_level': confidence_level
            },
            'performance_metrics': {
                'Sharpe_Ratio': self.calculate_sharpe_ratio(),
                'Sortino_Ratio': self.calculate_sortino_ratio(),
                'Calmar_Ratio': self.calculate_calmar_ratio()
            },
            'return_statistics': return_stats,
            'value_statistics': value_stats,
            'drawdown_metrics': drawdown
        }

        return metrics

    def create_summary_dataframe(self, confidence_level: float = 0.95) -> pd.DataFrame:
        """
        Create a formatted DataFrame with key metrics.

        Args:
            confidence_level: Confidence level for VaR/CVaR

        Returns:
            DataFrame with formatted metrics
        """
        metrics = self.get_comprehensive_metrics(confidence_level)

        summary_data = {
            'Metric': [],
            'Value': []
        }

        # Return statistics
        summary_data['Metric'].append('Mean Return (%)')
        summary_data['Value'].append(f"{metrics['return_statistics']['mean_return']:.2f}%")

        summary_data['Metric'].append('Median Return (%)')
        summary_data['Value'].append(f"{metrics['return_statistics']['median_return']:.2f}%")

        summary_data['Metric'].append('Probability of Profit')
        summary_data['Value'].append(f"{metrics['return_statistics']['probability_profit']:.2f}%")

        # Risk metrics
        summary_data['Metric'].append(f'VaR ({int(confidence_level*100)}%)')
        summary_data['Value'].append(f"{metrics['risk_metrics']['VaR_%']:.2f}%")

        summary_data['Metric'].append(f'CVaR ({int(confidence_level*100)}%)')
        summary_data['Value'].append(f"{metrics['risk_metrics']['CVaR_%']:.2f}%")

        # Performance ratios
        summary_data['Metric'].append('Sharpe Ratio')
        summary_data['Value'].append(f"{metrics['performance_metrics']['Sharpe_Ratio']:.3f}")

        summary_data['Metric'].append('Sortino Ratio')
        summary_data['Value'].append(f"{metrics['performance_metrics']['Sortino_Ratio']:.3f}")

        summary_data['Metric'].append('Mean Max Drawdown')
        summary_data['Value'].append(f"{metrics['drawdown_metrics']['mean_max_drawdown']:.2f}%")

        summary_data['Metric'].append('Worst Drawdown')
        summary_data['Value'].append(f"{metrics['drawdown_metrics']['worst_drawdown']:.2f}%")

        # Value statistics
        summary_data['Metric'].append('Mean Final Value')
        summary_data['Value'].append(f"${metrics['value_statistics']['mean_final_value']:,.2f}")

        summary_data['Metric'].append('Median Final Value')
        summary_data['Value'].append(f"${metrics['value_statistics']['median_final_value']:,.2f}")

        df = pd.DataFrame(summary_data)
        return df


def calculate_all_metrics(portfolio_values: np.ndarray, portfolio_returns: np.ndarray,
                         initial_investment: float, time_horizon: int,
                         risk_free_rate: float = 0.045,
                         confidence_level: float = 0.95) -> Dict:
    """
    Convenience function to calculate all metrics.

    Args:
        portfolio_values: Simulated portfolio values
        portfolio_returns: Final portfolio returns
        initial_investment: Starting value
        time_horizon: Days simulated
        risk_free_rate: Annual risk-free rate
        confidence_level: Confidence level for VaR/CVaR

    Returns:
        Dictionary with all metrics
    """
    calculator = PortfolioMetrics(portfolio_values, portfolio_returns,
                                 initial_investment, time_horizon, risk_free_rate)

    return calculator.get_comprehensive_metrics(confidence_level)
