"""Technical indicator helpers (returns, MA, RSI, volatility)."""

from typing import Union
import pandas as pd
import numpy as np


def calculate_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Add simple and log returns to a copy of the DataFrame.

    Returns a DataFrame with new columns: 'Return' (simple) and 'LogReturn'.
    """
    out = df.copy()
    out['Return'] = out['Adj Close'].pct_change()
    out['LogReturn'] = np.log(out['Adj Close']).diff()
    return out


def moving_average(series: Union[pd.Series, pd.DataFrame], window: int) -> pd.Series:
    """Compute simple moving average for a series.

    If a DataFrame is passed, the first column is used.
    """
    if isinstance(series, pd.DataFrame):
        series = series.iloc[:, 0]
    return series.rolling(window=window, min_periods=1).mean()


def calculate_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Calculate the Relative Strength Index (RSI).

    This uses the typical Wilder's smoothing method.
    Returns NaN for the first rows until enough data is present.
    """
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)

    # Wilder's EMA
    roll_up = up.ewm(com=window - 1, adjust=False).mean()
    roll_down = down.ewm(com=window - 1, adjust=False).mean()

    rs = roll_up / (roll_down + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_volatility(series: pd.Series, window: int = 20) -> pd.Series:
    """Rolling standard deviation of log returns annualized approximately.

    Returns the rolling volatility (std) of log returns.
    """
    logret = np.log(series).diff()
    vol = logret.rolling(window=window).std()
    # approximate annualization factor for daily data
    vol = vol * np.sqrt(252)
    return vol
