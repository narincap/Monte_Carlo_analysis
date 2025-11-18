# Monte Carlo Portfolio Risk Analyzer

A comprehensive web application for portfolio risk analysis using Monte Carlo simulation. Supports both US and Indonesian stocks with interactive visualizations and detailed risk metrics.

## Features

- **Multi-Market Support**: Analyze portfolios with US and Indonesian stocks
- **Monte Carlo Simulation**: Generate thousands of possible portfolio outcomes using Geometric Brownian Motion
- **Comprehensive Risk Metrics**:
  - Value at Risk (VaR)
  - Conditional VaR (CVaR/Expected Shortfall)
  - Sharpe Ratio, Sortino Ratio, Calmar Ratio
  - Maximum Drawdown analysis
- **Interactive Visualizations**:
  - Portfolio value distributions
  - Simulation path plots
  - Risk-adjusted performance charts
  - Correlation matrices
- **Flexible Portfolio Allocation**:
  - Equal weighting
  - Custom weights
  - Inverse volatility weighting
- **Export Capabilities**: Download simulation results and summary statistics

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Setup

1. **Navigate to the project directory**:
   ```bash
   cd monte_carlo_app
   ```

2. **Install required packages**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Application

Start the Streamlit web application:

```bash
streamlit run app.py
```

The application will open in your default web browser (usually at `http://localhost:8501`).

### Using the Application

#### 1. Enter Stock Tickers

In the sidebar, enter your stock tickers:

**US Stocks** (ticker only):
```
AAPL
MSFT
GOOGL
AMZN
```

**Indonesian Stocks** (add `.JK` suffix):
```
BBCA.JK
BMRI.JK
BBRI.JK
TLKM.JK
```

**Mixed Portfolio**:
```
AAPL
MSFT
BBCA.JK
BMRI.JK
```

#### 2. Configure Portfolio Weights

Choose your allocation method:

- **Equal Weight**: Automatically splits investment equally among all assets
- **Custom Weights**: Specify exact percentages (must sum to 100%)
- **Inverse Volatility**: Weight inversely by volatility (lower volatility = higher weight)

#### 3. Set Simulation Parameters

- **Initial Investment**: Starting capital (default: $10,000)
- **Time Horizon**: Number of days to simulate (252 days ≈ 1 year)
- **Number of Simulations**: More simulations = more accurate results but slower (10,000-100,000)
- **Confidence Level**: For VaR/CVaR calculation (90%, 95%, or 99%)

#### 4. Advanced Options

- **Historical Data Period**: How much historical data to use (1y, 2y, 3y, 5y)
- **Risk-Free Rate**: Annual risk-free rate for Sharpe/Sortino calculations (default: 4.5%)
- **Enable Rebalancing**: Rebalance portfolio monthly to maintain target weights

#### 5. Run Simulation

Click the **"🚀 Run Simulation"** button and wait for results.

### Understanding the Results

#### Key Metrics

- **Mean Return**: Average portfolio return across all simulations
- **VaR (Value at Risk)**: Maximum expected loss at a given confidence level
  - Example: 95% VaR of -8% means there's a 5% chance of losing more than 8%
- **CVaR (Conditional VaR)**: Average loss in the worst-case scenarios
- **Sharpe Ratio**: Risk-adjusted return (higher is better, >1.0 is good)
- **Probability of Profit**: Percentage of simulations that resulted in gains

#### Tabs

1. **Distribution**: Histogram of portfolio outcomes with VaR/CVaR markers
2. **Paths**: Visualization of simulation paths over time
3. **Metrics**: Detailed breakdown of all risk and performance metrics
4. **Export**: Download simulation results as CSV files

## Example Portfolios

### Conservative US Portfolio
```
Tickers: SPY, AGG, GLD
Weights: 60%, 30%, 10%
Time Horizon: 252 days
```

### Aggressive Tech Portfolio
```
Tickers: AAPL, MSFT, GOOGL, NVDA, META
Weights: Equal (20% each)
Time Horizon: 252 days
```

### Indonesian Blue Chip Portfolio
```
Tickers: BBCA.JK, BMRI.JK, TLKM.JK, ASII.JK
Weights: Equal (25% each)
Time Horizon: 252 days
```

### Diversified Global Portfolio
```
Tickers: SPY, AAPL, BBCA.JK, BMRI.JK, TLKM.JK
Weights: Custom (40%, 20%, 15%, 15%, 10%)
Time Horizon: 252 days
```

## Technical Details

### Monte Carlo Simulation

The application uses **Geometric Brownian Motion (GBM)** to simulate stock prices:

```
S(t+1) = S(t) × exp((μ - σ²/2)Δt + σ√Δt × Z)
```

Where:
- `μ` = mean return (drift)
- `σ` = volatility (standard deviation)
- `Δt` = time step (1 day)
- `Z` = random normal variable

### Correlation Handling

Asset correlations are preserved using **Cholesky decomposition** of the covariance matrix. This ensures realistic portfolio behavior where assets move together based on historical correlations.

### Risk Metrics Formulas

**Value at Risk (VaR)**:
```
VaR = Percentile(Returns, α) × Initial Investment
```
where α = 1 - Confidence Level

**Conditional VaR (CVaR)**:
```
CVaR = Mean(Returns | Returns ≤ VaR)
```

**Sharpe Ratio**:
```
Sharpe = (Portfolio Return - Risk-Free Rate) / Portfolio Volatility
```

**Sortino Ratio**:
```
Sortino = (Portfolio Return - Risk-Free Rate) / Downside Deviation
```

**Maximum Drawdown**:
```
MaxDD = Min((Peak - Trough) / Peak)
```

## Project Structure

```
monte_carlo_app/
├── app.py                      # Main Streamlit application
├── config.py                   # Configuration constants
├── requirements.txt            # Python dependencies
├── README.md                   # This file
└── utils/
    ├── __init__.py            # Package initialization
    ├── data_fetcher.py        # Stock data retrieval (yfinance)
    ├── monte_carlo_engine.py  # Monte Carlo simulation engine
    ├── portfolio_metrics.py   # Risk metric calculations
    └── visualizations.py      # Plotly chart generation
```

## Configuration

Edit `config.py` to customize default values:

```python
RISK_FREE_RATE = 0.045              # 4.5% annual
TRADING_DAYS_PER_YEAR = 252
DEFAULT_NUM_SIMULATIONS = 10000
DEFAULT_TIME_HORIZON = 252           # 1 year
DEFAULT_INITIAL_INVESTMENT = 10000
CACHE_TTL = 86400                   # Cache data for 24 hours
```

## Troubleshooting

### Ticker Not Found

**Problem**: "Error fetching data" or empty dataset

**Solutions**:
- Verify ticker symbol is correct
- For Indonesian stocks, ensure `.JK` suffix is added (e.g., `BBCA.JK` not `BBCA`)
- Check if the stock is actively traded
- Try a longer historical data period

### Simulation Too Slow

**Problem**: Application hangs or takes too long

**Solutions**:
- Reduce number of simulations (try 10,000 instead of 100,000)
- Reduce time horizon (try 126 days instead of 252)
- Reduce number of assets in portfolio
- Close other applications to free up memory

### Weights Don't Sum to 100%

**Problem**: "Weights must sum to 100%" error

**Solutions**:
- Double-check that all weight percentages add up to exactly 100%
- Use decimal precision (e.g., 33.33%, 33.33%, 33.34%)

### Installation Issues

**Problem**: Package installation fails

**Solutions**:
```bash
# Upgrade pip first
pip install --upgrade pip

# Install packages one by one
pip install streamlit
pip install yfinance
pip install pandas numpy scipy plotly

# Or use a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Limitations

1. **Historical Data Dependency**: Simulation quality depends on historical data quality
2. **Assumes Normal Distribution**: GBM assumes returns follow a normal distribution
3. **No Transaction Costs**: Does not account for trading fees or taxes
4. **No Dividends**: Does not incorporate dividend payments
5. **Market Assumptions**: Assumes markets behave similarly to historical patterns

## Best Practices

1. **Use Sufficient Historical Data**: At least 2 years recommended
2. **Run Enough Simulations**: Use 50,000+ for robust VaR estimates
3. **Validate Results**: Cross-reference with other tools
4. **Update Regularly**: Rerun analysis periodically as market conditions change
5. **Diversify**: Include uncorrelated assets for better risk management

## Common Indonesian Stock Tickers

| Sector | Ticker | Company |
|--------|--------|---------|
| **Banking** | BBCA.JK | Bank Central Asia |
| | BMRI.JK | Bank Mandiri |
| | BBRI.JK | Bank Rakyat Indonesia |
| | BBNI.JK | Bank Negara Indonesia |
| **Telecom** | TLKM.JK | Telkom Indonesia |
| **Consumer** | UNVR.JK | Unilever Indonesia |
| | ICBP.JK | Indofood CBP |
| | INDF.JK | Indofood Sukses Makmur |
| **Automotive** | ASII.JK | Astra International |
| **Mining** | ANTM.JK | Aneka Tambang |
| | ADRO.JK | Adaro Energy |

## Support and Contribution

### Reporting Issues

If you encounter bugs or have feature requests, please document:
1. Steps to reproduce
2. Expected behavior
3. Actual behavior
4. Screenshots if applicable
5. System information (OS, Python version)

### Future Enhancements

Potential features for future versions:
- Historical simulation (bootstrap method)
- Options pricing
- Portfolio optimization (efficient frontier)
- Stress testing scenarios
- Real-time data streaming
- Multiple currency support
- Tax calculation
- Dividend incorporation

## License

This project is provided as-is for educational and analytical purposes.

## Disclaimer

**IMPORTANT**: This tool is for educational and informational purposes only. It is NOT financial advice.

- Past performance does not guarantee future results
- Monte Carlo simulations show possible outcomes, not predictions
- Always consult with a qualified financial advisor before making investment decisions
- The developers are not responsible for any financial losses incurred from using this tool

## Credits

Built with:
- [Streamlit](https://streamlit.io/) - Web framework
- [yfinance](https://github.com/ranaroussi/yfinance) - Financial data
- [Plotly](https://plotly.com/) - Interactive visualizations
- [NumPy](https://numpy.org/) & [Pandas](https://pandas.pydata.org/) - Data processing
- [SciPy](https://scipy.org/) - Scientific computing

---

**Version**: 1.0.0
**Last Updated**: 2025-01-18
