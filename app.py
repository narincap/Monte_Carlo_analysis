"""
Monte Carlo Portfolio Risk Analysis - Streamlit Web Application

A web application for portfolio risk analysis using Monte Carlo simulation.
Supports both US and Indonesian stocks.
"""

import streamlit as st
import pandas as pd
import numpy as np
from utils import (
    DataFetcher,
    MonteCarloEngine,
    MonteCarloOptimizer,
    PortfolioMetrics,
    MonteCarloVisualizer
)
import config


# Page configuration
st.set_page_config(
    page_title="Monte Carlo Portfolio Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=config.CACHE_TTL)
def fetch_market_data(tickers, period):
    """Fetch and cache market data."""
    fetcher = DataFetcher()
    try:
        data = fetcher.fetch_data(tickers, period)
        fetcher.calculate_statistics()
        return fetcher, None
    except Exception as e:
        return None, str(e)


def validate_weights(weights, num_assets):
    """Validate portfolio weights."""
    if len(weights) != num_assets:
        return False, f"Number of weights ({len(weights)}) doesn't match number of assets ({num_assets})"

    if not np.isclose(sum(weights), 100.0, atol=0.1):
        return False, f"Weights must sum to 100% (current: {sum(weights):.2f}%)"

    if any(w < 0 for w in weights):
        return False, "Weights cannot be negative"

    return True, "Valid"


def main():
    """Main application."""

    # Header
    st.markdown('<p class="main-header">📊 Monte Carlo Portfolio Risk Analyzer</p>',
                unsafe_allow_html=True)
    st.markdown("Analyze portfolio risk using Monte Carlo simulation for US and Indonesian stocks")
    st.markdown("---")

    # Sidebar - Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Ticker input
        st.subheader("1. Portfolio Selection")
        ticker_input = st.text_area(
            "Enter tickers (one per line or comma-separated)",
            value="AAPL\nMSFT\nBBCA.JK\nBMRI.JK",
            height=100,
            help="US stocks: ticker only (e.g., AAPL)\nIndonesian stocks: add .JK suffix (e.g., BBCA.JK)"
        )

        # Parse tickers
        tickers = [t.strip() for t in ticker_input.replace(',', '\n').split('\n') if t.strip()]
        num_assets = len(tickers)

        if num_assets == 0:
            st.warning("Please enter at least one ticker")
            return

        st.success(f"Selected {num_assets} assets")

        # Weight allocation method
        st.subheader("2. Portfolio Weights")
        weight_method = st.radio(
            "Allocation method:",
            ["Equal Weight", "Custom Weights", "Inverse Volatility"],
            help="Equal: 1/n for each asset\nCustom: Specify manually\nInverse Vol: Weight by inverse volatility"
        )

        weights = []
        if weight_method == "Custom Weights":
            st.write("Enter weights (must sum to 100%):")
            weight_cols = st.columns(2)
            for i, ticker in enumerate(tickers):
                with weight_cols[i % 2]:
                    w = st.number_input(
                        f"{ticker} (%)",
                        min_value=0.0,
                        max_value=100.0,
                        value=100.0 / num_assets,
                        step=0.1,
                        key=f"weight_{i}"
                    )
                    weights.append(w)

            # Validate weights
            is_valid, msg = validate_weights(weights, num_assets)
            if not is_valid:
                st.error(msg)
                return
            else:
                st.success("✓ Weights are valid")

        # Simulation parameters
        st.subheader("3. Simulation Parameters")

        initial_investment = st.number_input(
            "Initial Investment ($)",
            min_value=100,
            max_value=10000000,
            value=10000,
            step=1000
        )

        time_horizon = st.slider(
            "Time Horizon (days)",
            min_value=1,
            max_value=756,
            value=252,
            help="252 trading days ≈ 1 year"
        )

        num_simulations = st.select_slider(
            "Number of Simulations",
            options=[1000, 5000, 10000, 25000, 50000, 100000],
            value=10000,
            help="More simulations = more accurate but slower"
        )

        confidence_level = st.select_slider(
            "Confidence Level (%)",
            options=[90, 95, 99],
            value=95,
            help="Confidence level for VaR/CVaR calculation"
        ) / 100.0

        # Advanced options
        with st.expander("Advanced Options"):
            data_period = st.selectbox(
                "Historical Data Period",
                options=["1y", "2y", "3y", "5y"],
                index=1
            )

            risk_free_rate = st.number_input(
                "Risk-Free Rate (annual %)",
                min_value=0.0,
                max_value=20.0,
                value=config.RISK_FREE_RATE * 100,
                step=0.1
            ) / 100.0

            rebalancing = st.checkbox(
                "Enable Rebalancing",
                value=False,
                help="Rebalance portfolio to target weights monthly"
            )

            num_display_paths = st.slider(
                "Paths to Display in Chart",
                min_value=10,
                max_value=500,
                value=100,
                help="Number of simulation paths to show in visualization"
            )

        # Run simulation button
        st.markdown("---")
        run_button = st.button("🚀 Run Simulation", type="primary", use_container_width=True)

    # Main content area
    if run_button:
        with st.spinner("Fetching market data..."):
            # Fetch data
            fetcher, error = fetch_market_data(tuple(tickers), data_period)

            if error:
                st.error(f"Error fetching data: {error}")
                return

            if fetcher is None:
                st.error("Failed to fetch market data")
                return

        # Display data summary
        st.subheader("📈 Market Data Summary")

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Latest Prices**")
            latest_prices = fetcher.get_latest_prices()
            st.dataframe(latest_prices.to_frame(name="Price ($)").style.format("${:.2f}"))

        with col2:
            st.write("**Historical Statistics**")
            summary_stats = fetcher.get_summary_statistics()
            summary_df = pd.DataFrame(summary_stats).T
            st.dataframe(summary_df.style.format({
                'Latest Price': '${:.2f}',
                'Annual Return': '{:.2%}',
                'Annual Volatility': '{:.2%}',
                'Sharpe Ratio (RF=0)': '{:.3f}',
                'Data Points': '{:.0f}'
            }))

        # Correlation matrix
        with st.expander("📊 Correlation Matrix"):
            corr_matrix = fetcher.get_correlation_matrix()
            st.dataframe(corr_matrix.style.background_gradient(cmap='RdBu', vmin=-1, vmax=1).format("{:.3f}"))

        # Determine weights
        if weight_method == "Equal Weight":
            final_weights = [100.0 / num_assets] * num_assets
        elif weight_method == "Inverse Volatility":
            volatilities = np.array([fetcher.returns[t].std() for t in tickers])
            inv_vol_weights = MonteCarloOptimizer.inverse_volatility_weights(volatilities)
            final_weights = (inv_vol_weights * 100).tolist()
        else:
            final_weights = weights

        # Display portfolio composition
        st.subheader("💼 Portfolio Composition")
        weight_df = pd.DataFrame({
            'Ticker': tickers,
            'Weight (%)': final_weights,
            'Allocation ($)': [initial_investment * (w/100) for w in final_weights]
        })
        st.dataframe(weight_df.style.format({
            'Weight (%)': '{:.2f}%',
            'Allocation ($)': '${:,.2f}'
        }))

        # Run Monte Carlo simulation
        st.markdown("---")
        st.subheader("🎲 Running Monte Carlo Simulation...")

        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            status_text.text("Initializing simulation engine...")
            progress_bar.progress(10)

            # Initialize engine
            engine = MonteCarloEngine(
                fetcher.mean_returns,
                fetcher.cov_matrix,
                fetcher.get_latest_prices()
            )

            status_text.text(f"Running {num_simulations:,} simulations...")
            progress_bar.progress(30)

            # Run simulation
            weights_array = np.array(final_weights) / 100.0

            if rebalancing:
                asset_paths, portfolio_values = engine.simulate_with_rebalancing(
                    num_simulations=num_simulations,
                    time_horizon=time_horizon,
                    initial_investment=initial_investment,
                    weights=weights_array
                )
            else:
                asset_paths, portfolio_values = engine.simulate_gbm_paths(
                    num_simulations=num_simulations,
                    time_horizon=time_horizon,
                    initial_investment=initial_investment,
                    weights=weights_array
                )

            progress_bar.progress(70)
            status_text.text("Calculating metrics...")

            # Calculate metrics
            portfolio_returns = engine.calculate_portfolio_returns(portfolio_values)

            metrics_calculator = PortfolioMetrics(
                portfolio_values,
                portfolio_returns,
                initial_investment,
                time_horizon,
                risk_free_rate
            )

            metrics = metrics_calculator.get_comprehensive_metrics(confidence_level)

            progress_bar.progress(90)
            status_text.text("Generating visualizations...")

            # Create visualizer
            visualizer = MonteCarloVisualizer(
                portfolio_values,
                portfolio_returns,
                asset_paths,
                tickers
            )

            progress_bar.progress(100)
            status_text.text("✅ Simulation complete!")

            # Display results
            st.markdown("---")
            st.markdown('<p class="sub-header">📊 Simulation Results</p>', unsafe_allow_html=True)

            # Key metrics in columns
            st.subheader("Key Metrics")
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "Mean Return",
                    f"{metrics['return_statistics']['mean_return']:.2f}%",
                    delta=f"{metrics['return_statistics']['probability_profit']:.1f}% profit prob"
                )

            with col2:
                st.metric(
                    f"VaR ({int(confidence_level*100)}%)",
                    f"{metrics['risk_metrics']['VaR_%']:.2f}%",
                    delta=f"${metrics['risk_metrics']['VaR_$']:,.0f}"
                )

            with col3:
                st.metric(
                    f"CVaR ({int(confidence_level*100)}%)",
                    f"{metrics['risk_metrics']['CVaR_%']:.2f}%",
                    delta=f"${metrics['risk_metrics']['CVaR_$']:,.0f}"
                )

            with col4:
                st.metric(
                    "Sharpe Ratio",
                    f"{metrics['performance_metrics']['Sharpe_Ratio']:.3f}",
                    delta=f"Sortino: {metrics['performance_metrics']['Sortino_Ratio']:.3f}"
                )

            # Tabs for different views
            tab1, tab2, tab3, tab4 = st.tabs(["📈 Distribution", "📉 Paths", "📊 Metrics", "💾 Export"])

            with tab1:
                st.subheader("Portfolio Value Distribution")
                fig_dist = visualizer.plot_distribution(metrics)
                st.plotly_chart(fig_dist, use_container_width=True)

                col1, col2 = st.columns(2)
                with col1:
                    fig_return_dist = visualizer.plot_return_distribution(metrics)
                    st.plotly_chart(fig_return_dist, use_container_width=True)

                with col2:
                    fig_percentiles = visualizer.plot_percentile_chart(metrics)
                    st.plotly_chart(fig_percentiles, use_container_width=True)

            with tab2:
                st.subheader("Simulation Paths")
                fig_paths = visualizer.plot_simulation_paths(num_paths=num_display_paths)
                st.plotly_chart(fig_paths, use_container_width=True)

                st.info(f"Showing {min(num_display_paths, num_simulations)} of {num_simulations:,} simulation paths")

            with tab3:
                st.subheader("Detailed Metrics")

                col1, col2 = st.columns(2)

                with col1:
                    st.write("**Risk Metrics**")
                    risk_df = pd.DataFrame([
                        {"Metric": f"VaR ({int(confidence_level*100)}%)",
                         "Value": f"{metrics['risk_metrics']['VaR_%']:.2f}%",
                         "Dollar": f"${metrics['risk_metrics']['VaR_$']:,.2f}"},
                        {"Metric": f"CVaR ({int(confidence_level*100)}%)",
                         "Value": f"{metrics['risk_metrics']['CVaR_%']:.2f}%",
                         "Dollar": f"${metrics['risk_metrics']['CVaR_$']:,.2f}"}
                    ])
                    st.dataframe(risk_df, hide_index=True, use_container_width=True)

                    st.write("**Performance Ratios**")
                    perf_df = pd.DataFrame([
                        {"Metric": "Sharpe Ratio",
                         "Value": f"{metrics['performance_metrics']['Sharpe_Ratio']:.4f}"},
                        {"Metric": "Sortino Ratio",
                         "Value": f"{metrics['performance_metrics']['Sortino_Ratio']:.4f}"},
                        {"Metric": "Calmar Ratio",
                         "Value": f"{metrics['performance_metrics']['Calmar_Ratio']:.4f}"}
                    ])
                    st.dataframe(perf_df, hide_index=True, use_container_width=True)

                with col2:
                    st.write("**Return Statistics**")
                    return_df = pd.DataFrame([
                        {"Metric": "Mean Return", "Value": f"{metrics['return_statistics']['mean_return']:.2f}%"},
                        {"Metric": "Median Return", "Value": f"{metrics['return_statistics']['median_return']:.2f}%"},
                        {"Metric": "Std Deviation", "Value": f"{metrics['return_statistics']['std_return']:.2f}%"},
                        {"Metric": "Min Return", "Value": f"{metrics['return_statistics']['min_return']:.2f}%"},
                        {"Metric": "Max Return", "Value": f"{metrics['return_statistics']['max_return']:.2f}%"},
                        {"Metric": "Probability of Profit", "Value": f"{metrics['return_statistics']['probability_profit']:.2f}%"}
                    ])
                    st.dataframe(return_df, hide_index=True, use_container_width=True)

                    st.write("**Drawdown Metrics**")
                    dd_df = pd.DataFrame([
                        {"Metric": "Mean Max Drawdown",
                         "Value": f"{metrics['drawdown_metrics']['mean_max_drawdown']:.2f}%"},
                        {"Metric": "Worst Drawdown",
                         "Value": f"{metrics['drawdown_metrics']['worst_drawdown']:.2f}%"},
                        {"Metric": "Median Drawdown",
                         "Value": f"{metrics['drawdown_metrics']['median_drawdown']:.2f}%"}
                    ])
                    st.dataframe(dd_df, hide_index=True, use_container_width=True)

                # Risk-adjusted metrics chart
                fig_risk_metrics = visualizer.plot_risk_metrics_comparison(metrics)
                st.plotly_chart(fig_risk_metrics, use_container_width=True)

            with tab4:
                st.subheader("Export Results")

                # Create export dataframe
                export_data = {
                    'Simulation': range(1, num_simulations + 1),
                    'Final_Value': portfolio_values[:, -1],
                    'Return_%': portfolio_returns * 100
                }

                # Add individual asset returns
                for i, ticker in enumerate(tickers):
                    asset_returns = (asset_paths[:, -1, i] - asset_paths[:, 0, i]) / asset_paths[:, 0, i]
                    export_data[f'{ticker}_Return_%'] = asset_returns * 100

                export_df = pd.DataFrame(export_data)

                # Create summary statistics
                summary_export = pd.DataFrame([
                    {"Parameter": "Initial Investment", "Value": f"${initial_investment:,.2f}"},
                    {"Parameter": "Time Horizon (days)", "Value": time_horizon},
                    {"Parameter": "Number of Simulations", "Value": num_simulations},
                    {"Parameter": "Confidence Level", "Value": f"{confidence_level*100}%"},
                    {"Parameter": "Mean Final Value", "Value": f"${metrics['value_statistics']['mean_final_value']:,.2f}"},
                    {"Parameter": "Median Final Value", "Value": f"${metrics['value_statistics']['median_final_value']:,.2f}"},
                    {"Parameter": "Mean Return", "Value": f"{metrics['return_statistics']['mean_return']:.2f}%"},
                    {"Parameter": f"VaR ({int(confidence_level*100)}%)", "Value": f"{metrics['risk_metrics']['VaR_%']:.2f}%"},
                    {"Parameter": f"CVaR ({int(confidence_level*100)}%)", "Value": f"{metrics['risk_metrics']['CVaR_%']:.2f}%"},
                    {"Parameter": "Sharpe Ratio", "Value": f"{metrics['performance_metrics']['Sharpe_Ratio']:.4f}"},
                    {"Parameter": "Sortino Ratio", "Value": f"{metrics['performance_metrics']['Sortino_Ratio']:.4f}"}
                ])

                col1, col2 = st.columns(2)

                with col1:
                    st.write("**Simulation Results**")
                    st.write(f"Preview (showing first 100 of {num_simulations:,} simulations):")
                    st.dataframe(export_df.head(100), use_container_width=True)

                    csv_simulations = export_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download All Simulations (CSV)",
                        data=csv_simulations,
                        file_name=f"monte_carlo_simulations_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )

                with col2:
                    st.write("**Summary Statistics**")
                    st.dataframe(summary_export, hide_index=True, use_container_width=True)

                    csv_summary = summary_export.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Summary (CSV)",
                        data=csv_summary,
                        file_name=f"monte_carlo_summary_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )

        except Exception as e:
            st.error(f"Error during simulation: {str(e)}")
            st.exception(e)

    else:
        # Instructions when simulation hasn't been run
        st.info("👈 Configure your portfolio parameters in the sidebar and click 'Run Simulation' to begin")

        st.subheader("How to Use")
        st.markdown("""
        1. **Enter Tickers**: Add stock symbols in the sidebar
           - US stocks: Use ticker only (e.g., `AAPL`, `MSFT`)
           - Indonesian stocks: Add `.JK` suffix (e.g., `BBCA.JK`, `BMRI.JK`)

        2. **Choose Weights**: Select how to allocate your portfolio
           - Equal Weight: Splits investment equally
           - Custom: Specify exact percentages
           - Inverse Volatility: Weight by inverse volatility (less risky = higher weight)

        3. **Set Parameters**:
           - Initial Investment: Starting capital
           - Time Horizon: Days to simulate (252 ≈ 1 year)
           - Simulations: More = more accurate but slower
           - Confidence Level: For VaR/CVaR calculation

        4. **Run Simulation**: Click the button and wait for results

        5. **Analyze Results**: View distributions, paths, metrics, and export data
        """)

        st.subheader("Example Portfolios")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**US Tech**")
            st.code("AAPL\nMSFT\nGOOGL\nAMZN")

        with col2:
            st.markdown("**Indonesian Banks**")
            st.code("BBCA.JK\nBMRI.JK\nBBRI.JK\nBBNI.JK")

        with col3:
            st.markdown("**Mixed Portfolio**")
            st.code("AAPL\nMSFT\nBBCA.JK\nTLKM.JK")


if __name__ == "__main__":
    main()
