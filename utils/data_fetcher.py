"""
Data Fetcher Module
Handles stock data retrieval using yfinance with support for US and Indonesian stocks.
"""

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

import config

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class DataFetchError(Exception):
    """Raised when price data cannot be fetched or validated."""


def _normalize_ticker(raw: str) -> str:
    """
    Normalize a single ticker string safely.

    - Trims whitespace
    - Uppercases the symbol
    - Ensures Indonesian tickers end with .JK
    - Leaves US tickers untouched
    """
    if raw is None:
        return ""

    cleaned = raw.strip().replace(" ", "").upper()
    if not cleaned:
        return ""

    # Already has an exchange suffix
    if "." in cleaned:
        return cleaned

    # Normalize Indonesian symbols to .JK
    known_idx = set(getattr(config, "INDONESIAN_TICKERS", []))
    if cleaned.endswith("JK"):
        return cleaned if cleaned.endswith(".JK") else f"{cleaned[:-2]}.JK"
    if cleaned in known_idx:
        return f"{cleaned}.JK"

    # Default: treat as non-Indonesian ticker
    return cleaned


def _period_days_to_str(days: int) -> str:
    """Convert a positive integer of days to a yfinance-compatible period string."""
    try:
        days_int = int(days)
    except (TypeError, ValueError) as exc:
        raise DataFetchError("Period must be an integer number of days") from exc

    if days_int <= 0:
        raise DataFetchError("Period in days must be greater than zero")

    return f"{days_int}d"


def _extract_adj_close(data: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
    """Return only the Adj Close prices as a DataFrame and keep ordering."""
    if isinstance(data.columns, pd.MultiIndex):
        level0 = data.columns.get_level_values(0)
        level1 = data.columns.get_level_values(1)

        # Case A: price fields at level 0 (default when group_by=None)
        if "Adj Close" in level0:
            adj_close = data.xs("Adj Close", level=0, axis=1)
        elif "Close" in level0:
            adj_close = data.xs("Close", level=0, axis=1)

        # Case B: price fields at level 1 (when group_by='ticker')
        elif "Adj Close" in level1:
            adj_close = data.xs("Adj Close", level=1, axis=1)
        elif "Close" in level1:
            adj_close = data.xs("Close", level=1, axis=1)
        else:
            raise DataFetchError("Downloaded data is missing Adj Close / Close columns")
    else:
        if "Adj Close" in data.columns:
            adj_close = data[["Adj Close"]]
        elif "Close" in data.columns:
            adj_close = data[["Close"]]
        else:
            raise DataFetchError("Downloaded data is missing Adj Close / Close columns")

    # Ensure DataFrame and reorder columns to match requested tickers
    adj_close = pd.DataFrame(adj_close)
    reordered_cols = [t for t in tickers if t in adj_close.columns]
    missing_cols = [t for t in tickers if t not in adj_close.columns]
    if missing_cols:
        logger.warning("Missing tickers in downloaded data: %s", ",".join(missing_cols))
    adj_close = adj_close.loc[:, reordered_cols]
    return adj_close


def fetch_price_data(tickers: List[str], days: int) -> pd.DataFrame:
    """
    Fetch adjusted close prices for the given tickers and horizon (in days).

    Guarantees a non-empty DataFrame, removes empty columns, and raises
    DataFetchError on any failure instead of returning scalars.
    """
    if not tickers or not isinstance(tickers, (list, tuple)):
        raise DataFetchError("Tickers must be a non-empty list of symbols")

    normalized = []
    for raw in tickers:
        norm = _normalize_ticker(raw)
        if norm:
            normalized.append(norm)
        else:
            logger.warning("Dropping empty/invalid ticker input: %r", raw)

    # Remove duplicates while preserving order
    normalized = list(dict.fromkeys(normalized))
    if not normalized:
        raise DataFetchError("No valid tickers after normalization")

    period = _period_days_to_str(days)
    logger.info("Fetching data for tickers=%s period=%s", normalized, period)

    try:
        yf_data = yf.download(
            normalized if len(normalized) > 1 else normalized[0],
            period=period,
            progress=False,
            auto_adjust=False,
            group_by="ticker",
            threads=False,
        )
    except Exception as exc:
        logger.exception("yfinance download failed")
        raise DataFetchError(f"Failed to download price data: {exc}") from exc

    if yf_data is None or yf_data.empty:
        raise DataFetchError(f"No price data returned for tickers {normalized} and period {period}")

    adj_close = _extract_adj_close(yf_data, normalized)

    # Drop empty columns/rows
    adj_close = adj_close.dropna(axis=1, how="all")
    adj_close = adj_close.dropna(axis=0, how="all")

    if adj_close.empty or not len(adj_close.columns):
        raise DataFetchError(f"Price data contains only empty columns for tickers {normalized}")

    logger.info(
        "Fetched adjusted close data with shape %s for tickers=%s",
        adj_close.shape,
        list(adj_close.columns),
    )
    return adj_close


class DataFetcher:
    """Fetch and process stock data for Monte Carlo simulation."""

    def __init__(self):
        self.data = None
        self.tickers: List[str] = []
        self.returns = None
        self.mean_returns = None
        self.cov_matrix = None

    @staticmethod
    def normalize_ticker(ticker: str) -> str:
        """Backward-compatible helper that delegates to the new normalizer."""
        return _normalize_ticker(ticker)

    def validate_tickers(self, tickers: List[str]) -> Tuple[List[str], List[str]]:
        """Validate tickers by attempting a quick fetch."""
        valid, invalid = [], []
        for ticker in tickers:
            norm = _normalize_ticker(ticker)
            if not norm:
                invalid.append(ticker)
                continue
            try:
                test = yf.download(norm, period="5d", progress=False)
                if test is not None and not test.empty:
                    valid.append(norm)
                else:
                    invalid.append(ticker)
            except Exception:
                invalid.append(ticker)
        return valid, invalid

    def fetch_data(self, tickers: List[str], period: str = "2y") -> pd.DataFrame:
        """
        Fetch historical price data for multiple tickers.

        Accepts legacy string periods (e.g., '1y', '252d') or integer days,
        but always delegates to fetch_price_data to enforce the new safeguards.
        """
        # Convert legacy period strings to days
        days: int
        if isinstance(period, int):
            days = period
        elif isinstance(period, str):
            p = period.strip().lower()
            if p.endswith("d") and p[:-1].isdigit():
                days = int(p[:-1])
            elif p.endswith("y") and p[:-1].isdigit():
                days = int(p[:-1]) * getattr(config, "TRADING_DAYS_PER_YEAR", 252)
            elif p.isdigit():
                days = int(p)
            else:
                raise DataFetchError(f"Unsupported period format: {period}")
        else:
            raise DataFetchError(f"Unsupported period type: {type(period)}")

        self.data = fetch_price_data(tickers, days)
        self.tickers = list(self.data.columns)
        return self.data

    def calculate_returns(self) -> pd.DataFrame:
        """
        Calculate daily returns from price data.

        Returns:
            DataFrame with daily returns
        """
        if self.data is None:
            raise ValueError("No data available. Call fetch_data() first.")

        self.returns = self.data.pct_change().dropna()
        return self.returns

    def calculate_statistics(self) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Calculate mean returns and covariance matrix.

        Returns:
            Tuple of (mean_returns, covariance_matrix)
        """
        if self.returns is None:
            self.calculate_returns()

        # Calculate mean returns - ensure it's always a Series
        mean_ret = self.returns.mean()
        if isinstance(mean_ret, (float, np.floating)):
            # Single ticker case - convert scalar to Series
            self.mean_returns = pd.Series([mean_ret], index=self.returns.columns)
        else:
            self.mean_returns = mean_ret

        # Calculate covariance matrix - ensure it's always a DataFrame
        cov = self.returns.cov()
        if isinstance(cov, (float, np.floating)):
            # Single ticker case - convert scalar to DataFrame
            self.cov_matrix = pd.DataFrame([[cov]],
                                          index=self.returns.columns,
                                          columns=self.returns.columns)
        else:
            self.cov_matrix = cov

        return self.mean_returns, self.cov_matrix

    def get_latest_prices(self) -> pd.Series:
        """
        Get the most recent prices for all tickers.

        Returns:
            Series with latest prices
        """
        if self.data is None:
            raise ValueError("No data available. Call fetch_data() first.")

        return self.data.iloc[-1]

    def get_summary_statistics(self) -> Dict:
        """
        Get summary statistics for the fetched data.

        Returns:
            Dictionary with summary statistics
        """
        if self.data is None or self.returns is None:
            raise ValueError("No data available. Call fetch_data() and calculate_returns() first.")

        summary = {}

        for ticker in self.tickers:
            annual_return = self.returns[ticker].mean() * 252
            annual_vol = self.returns[ticker].std() * np.sqrt(252)

            summary[ticker] = {
                'Latest Price': self.data[ticker].iloc[-1],
                'Annual Return': annual_return,
                'Annual Volatility': annual_vol,
                'Sharpe Ratio (RF=0)': annual_return / annual_vol if annual_vol > 0 else 0,
                'Data Points': len(self.data)
            }

        return summary

    def get_correlation_matrix(self) -> pd.DataFrame:
        """
        Get correlation matrix of returns.

        Returns:
            DataFrame with correlation matrix
        """
        if self.returns is None:
            self.calculate_returns()

        return self.returns.corr()


# Convenience functions for quick access
def fetch_stock_data(tickers: List[str], period: str = '2y') -> Tuple[pd.DataFrame, DataFetcher]:
    """
    Quick function to fetch stock data.

    Args:
        tickers: List of ticker symbols
        period: Data period

    Returns:
        Tuple of (price_data, fetcher_object)
    """
    fetcher = DataFetcher()
    data = fetcher.fetch_data(tickers, period)
    fetcher.calculate_statistics()
    return data, fetcher
