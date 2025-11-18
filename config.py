"""
Configuration file for Monte Carlo Portfolio Analysis
"""

# Risk-free rate (annual)
# Default: 4.5% (approximate US 10-year Treasury rate)
RISK_FREE_RATE = 0.045

# Trading days per year
TRADING_DAYS_PER_YEAR = 252

# Cache settings (seconds)
CACHE_TTL = 86400  # 24 hours

# Simulation defaults
DEFAULT_NUM_SIMULATIONS = 10000
DEFAULT_TIME_HORIZON = 252  # 1 year
DEFAULT_INITIAL_INVESTMENT = 10000
DEFAULT_CONFIDENCE_LEVEL = 0.95

# Data settings
DEFAULT_DATA_PERIOD = '2y'  # 2 years of historical data
MIN_DATA_POINTS = 100  # Minimum data points required

# Indonesian stock exchange settings
IDX_SUFFIX = '.JK'  # Jakarta Stock Exchange suffix
IDX_TRADING_DAYS = 255  # Average trading days per year for IDX

# Known Indonesian stock tickers (for auto-detection)
INDONESIAN_TICKERS = [
    'BBCA', 'BBRI', 'BMRI', 'BBNI',  # Banks
    'TLKM', 'ASII', 'UNVR', 'HMSP',  # Blue chips
    'GGRM', 'KLBF', 'ICBP', 'INDF',  # Consumer goods
    'ANTM', 'ADRO', 'PTBA', 'INCO',  # Mining
    'SMGR', 'WIKA', 'WSKT', 'PTPP',  # Infrastructure
]

# Visualization settings
DEFAULT_NUM_DISPLAY_PATHS = 100
HISTOGRAM_BINS = 100
CHART_HEIGHT = 500
DASHBOARD_HEIGHT = 800

# Color schemes
COLORS = {
    'positive': '#2ecc71',
    'negative': '#e74c3c',
    'neutral': '#95a5a6',
    'primary': '#3498db',
    'secondary': '#9b59b6'
}
