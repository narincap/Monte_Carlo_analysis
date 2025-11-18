"""
Visualization Module
Creates interactive Plotly charts for Monte Carlo simulation results.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from typing import Dict, List, Optional


class MonteCarloVisualizer:
    """Create interactive visualizations for Monte Carlo simulation results."""

    def __init__(self, portfolio_values: np.ndarray, portfolio_returns: np.ndarray,
                 asset_paths: Optional[np.ndarray] = None, tickers: Optional[List[str]] = None):
        """
        Initialize visualizer.

        Args:
            portfolio_values: Simulated portfolio values (num_simulations, time_steps)
            portfolio_returns: Final portfolio returns (num_simulations,)
            asset_paths: Individual asset paths (optional)
            tickers: List of ticker symbols (optional)
        """
        self.portfolio_values = portfolio_values
        self.portfolio_returns = portfolio_returns
        self.asset_paths = asset_paths
        self.tickers = tickers or []

    def plot_distribution(self, metrics: Dict, bins: int = 100) -> go.Figure:
        """
        Create histogram of final portfolio values with VaR/CVaR markers.

        Args:
            metrics: Dictionary with risk metrics (from PortfolioMetrics)
            bins: Number of bins for histogram

        Returns:
            Plotly Figure object
        """
        final_values = self.portfolio_values[:, -1]
        initial_value = metrics['value_statistics']['initial_value']

        # Create histogram
        fig = go.Figure()

        fig.add_trace(go.Histogram(
            x=final_values,
            nbinsx=bins,
            name='Portfolio Value Distribution',
            marker_color='lightblue',
            opacity=0.7
        ))

        # Add VaR line
        var_value = initial_value + metrics['risk_metrics']['VaR_$']
        fig.add_vline(
            x=var_value,
            line_dash="dash",
            line_color="red",
            annotation_text=f"VaR ({metrics['risk_metrics']['confidence_level']*100:.0f}%): ${var_value:,.0f}",
            annotation_position="top left"
        )

        # Add CVaR line
        cvar_value = initial_value + metrics['risk_metrics']['CVaR_$']
        fig.add_vline(
            x=cvar_value,
            line_dash="dash",
            line_color="darkred",
            annotation_text=f"CVaR ({metrics['risk_metrics']['confidence_level']*100:.0f}%): ${cvar_value:,.0f}",
            annotation_position="top left"
        )

        # Add mean line
        mean_value = metrics['value_statistics']['mean_final_value']
        fig.add_vline(
            x=mean_value,
            line_dash="solid",
            line_color="green",
            annotation_text=f"Mean: ${mean_value:,.0f}",
            annotation_position="top right"
        )

        # Add median line
        median_value = metrics['value_statistics']['median_final_value']
        fig.add_vline(
            x=median_value,
            line_dash="dot",
            line_color="blue",
            annotation_text=f"Median: ${median_value:,.0f}",
            annotation_position="bottom right"
        )

        fig.update_layout(
            title='Distribution of Final Portfolio Values',
            xaxis_title='Final Portfolio Value ($)',
            yaxis_title='Frequency',
            showlegend=True,
            height=500,
            hovermode='x unified'
        )

        return fig

    def plot_return_distribution(self, metrics: Dict, bins: int = 100) -> go.Figure:
        """
        Create histogram of portfolio returns.

        Args:
            metrics: Dictionary with risk metrics
            bins: Number of bins

        Returns:
            Plotly Figure object
        """
        returns_pct = self.portfolio_returns * 100

        fig = go.Figure()

        fig.add_trace(go.Histogram(
            x=returns_pct,
            nbinsx=bins,
            name='Return Distribution',
            marker_color='lightgreen',
            opacity=0.7
        ))

        # Add VaR line
        var_pct = metrics['risk_metrics']['VaR_%']
        fig.add_vline(
            x=var_pct,
            line_dash="dash",
            line_color="red",
            annotation_text=f"VaR: {var_pct:.2f}%",
            annotation_position="top left"
        )

        # Add mean line
        mean_return = metrics['return_statistics']['mean_return']
        fig.add_vline(
            x=mean_return,
            line_dash="solid",
            line_color="green",
            annotation_text=f"Mean: {mean_return:.2f}%",
            annotation_position="top right"
        )

        # Add zero line
        fig.add_vline(
            x=0,
            line_dash="solid",
            line_color="black",
            line_width=1
        )

        fig.update_layout(
            title='Distribution of Portfolio Returns',
            xaxis_title='Return (%)',
            yaxis_title='Frequency',
            showlegend=True,
            height=500,
            hovermode='x unified'
        )

        return fig

    def plot_simulation_paths(self, num_paths: int = 100, percentiles: bool = True) -> go.Figure:
        """
        Plot sample simulation paths over time.

        Args:
            num_paths: Number of paths to display (randomly sampled)
            percentiles: Whether to show percentile bands

        Returns:
            Plotly Figure object
        """
        fig = go.Figure()

        # Randomly sample paths to display
        num_simulations = self.portfolio_values.shape[0]
        if num_paths < num_simulations:
            sample_indices = np.random.choice(num_simulations, num_paths, replace=False)
            sample_paths = self.portfolio_values[sample_indices, :]
        else:
            sample_paths = self.portfolio_values

        time_steps = np.arange(self.portfolio_values.shape[1])

        # Plot sample paths with low opacity
        for i in range(sample_paths.shape[0]):
            fig.add_trace(go.Scatter(
                x=time_steps,
                y=sample_paths[i, :],
                mode='lines',
                line=dict(width=0.5, color='lightblue'),
                opacity=0.3,
                showlegend=False,
                hoverinfo='skip'
            ))

        if percentiles:
            # Add percentile bands
            percentile_5 = np.percentile(self.portfolio_values, 5, axis=0)
            percentile_25 = np.percentile(self.portfolio_values, 25, axis=0)
            percentile_50 = np.percentile(self.portfolio_values, 50, axis=0)
            percentile_75 = np.percentile(self.portfolio_values, 75, axis=0)
            percentile_95 = np.percentile(self.portfolio_values, 95, axis=0)

            # 5th-95th percentile band
            fig.add_trace(go.Scatter(
                x=time_steps,
                y=percentile_95,
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            ))

            fig.add_trace(go.Scatter(
                x=time_steps,
                y=percentile_5,
                mode='lines',
                line=dict(width=0),
                fill='tonexty',
                fillcolor='rgba(0, 100, 80, 0.2)',
                name='5th-95th Percentile',
                hoverinfo='skip'
            ))

            # Median line
            fig.add_trace(go.Scatter(
                x=time_steps,
                y=percentile_50,
                mode='lines',
                line=dict(width=2, color='darkblue'),
                name='Median Path'
            ))

        fig.update_layout(
            title=f'Monte Carlo Simulation Paths (Showing {min(num_paths, num_simulations)} of {num_simulations})',
            xaxis_title='Days',
            yaxis_title='Portfolio Value ($)',
            showlegend=True,
            height=600,
            hovermode='x unified'
        )

        return fig

    def plot_correlation_heatmap(self, returns: pd.DataFrame) -> go.Figure:
        """
        Create correlation heatmap of asset returns.

        Args:
            returns: DataFrame of asset returns

        Returns:
            Plotly Figure object
        """
        corr_matrix = returns.corr()

        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate='%{text}',
            textfont={"size": 10},
            colorbar=dict(title="Correlation")
        ))

        fig.update_layout(
            title='Asset Return Correlation Matrix',
            height=500,
            width=600
        )

        return fig

    def plot_risk_metrics_comparison(self, metrics: Dict) -> go.Figure:
        """
        Create bar chart comparing risk-adjusted performance metrics.

        Args:
            metrics: Dictionary with performance metrics

        Returns:
            Plotly Figure object
        """
        perf = metrics['performance_metrics']

        metric_names = ['Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio']
        metric_values = [
            perf['Sharpe_Ratio'],
            perf['Sortino_Ratio'],
            min(perf['Calmar_Ratio'], 10)  # Cap at 10 for visualization
        ]

        colors = ['lightblue' if v > 0 else 'lightcoral' for v in metric_values]

        fig = go.Figure(data=[
            go.Bar(
                x=metric_names,
                y=metric_values,
                marker_color=colors,
                text=[f'{v:.3f}' for v in metric_values],
                textposition='outside'
            )
        ])

        fig.update_layout(
            title='Risk-Adjusted Performance Metrics',
            yaxis_title='Ratio',
            showlegend=False,
            height=400
        )

        # Add reference line at 0
        fig.add_hline(y=0, line_dash="solid", line_color="black", line_width=1)

        return fig

    def plot_percentile_chart(self, metrics: Dict) -> go.Figure:
        """
        Create percentile chart showing distribution of outcomes.

        Args:
            metrics: Dictionary with return statistics

        Returns:
            Plotly Figure object
        """
        stats = metrics['return_statistics']

        percentiles = ['5th', '25th', 'Median', '75th', '95th']
        values = [
            stats['percentile_5'],
            stats['percentile_25'],
            stats['median_return'],
            stats['percentile_75'],
            stats['percentile_95']
        ]

        colors = ['darkred', 'lightcoral', 'lightgray', 'lightgreen', 'darkgreen']

        fig = go.Figure(data=[
            go.Bar(
                x=percentiles,
                y=values,
                marker_color=colors,
                text=[f'{v:.2f}%' for v in values],
                textposition='outside'
            )
        ])

        fig.update_layout(
            title='Return Percentiles',
            xaxis_title='Percentile',
            yaxis_title='Return (%)',
            showlegend=False,
            height=400
        )

        # Add zero line
        fig.add_hline(y=0, line_dash="solid", line_color="black", line_width=1)

        return fig

    def create_dashboard(self, metrics: Dict, returns: Optional[pd.DataFrame] = None) -> go.Figure:
        """
        Create comprehensive dashboard with multiple subplots.

        Args:
            metrics: Dictionary with all metrics
            returns: DataFrame with historical returns (optional)

        Returns:
            Plotly Figure with subplots
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Portfolio Value Distribution', 'Return Distribution',
                          'Risk-Adjusted Metrics', 'Return Percentiles'),
            specs=[[{"type": "histogram"}, {"type": "histogram"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )

        # Distribution of final values
        final_values = self.portfolio_values[:, -1]
        fig.add_trace(
            go.Histogram(x=final_values, nbinsx=50, name='Value Dist',
                        marker_color='lightblue', showlegend=False),
            row=1, col=1
        )

        # Distribution of returns
        returns_pct = self.portfolio_returns * 100
        fig.add_trace(
            go.Histogram(x=returns_pct, nbinsx=50, name='Return Dist',
                        marker_color='lightgreen', showlegend=False),
            row=1, col=2
        )

        # Risk-adjusted metrics
        perf = metrics['performance_metrics']
        metric_names = ['Sharpe', 'Sortino', 'Calmar']
        metric_values = [
            perf['Sharpe_Ratio'],
            perf['Sortino_Ratio'],
            min(perf['Calmar_Ratio'], 10)
        ]
        colors = ['lightblue' if v > 0 else 'lightcoral' for v in metric_values]

        fig.add_trace(
            go.Bar(x=metric_names, y=metric_values, marker_color=colors,
                  showlegend=False),
            row=2, col=1
        )

        # Percentiles
        stats = metrics['return_statistics']
        percentiles = ['5th', '25th', 'Median', '75th', '95th']
        values = [
            stats['percentile_5'],
            stats['percentile_25'],
            stats['median_return'],
            stats['percentile_75'],
            stats['percentile_95']
        ]
        perc_colors = ['darkred', 'lightcoral', 'lightgray', 'lightgreen', 'darkgreen']

        fig.add_trace(
            go.Bar(x=percentiles, y=values, marker_color=perc_colors,
                  showlegend=False),
            row=2, col=2
        )

        fig.update_xaxes(title_text="Portfolio Value ($)", row=1, col=1)
        fig.update_xaxes(title_text="Return (%)", row=1, col=2)
        fig.update_xaxes(title_text="Metric", row=2, col=1)
        fig.update_xaxes(title_text="Percentile", row=2, col=2)

        fig.update_yaxes(title_text="Frequency", row=1, col=1)
        fig.update_yaxes(title_text="Frequency", row=1, col=2)
        fig.update_yaxes(title_text="Ratio", row=2, col=1)
        fig.update_yaxes(title_text="Return (%)", row=2, col=2)

        fig.update_layout(
            title_text="Monte Carlo Simulation Dashboard",
            height=800,
            showlegend=False
        )

        return fig
