"""Utilities: download, clean, and prepare datasets."""

from typing import Tuple, List
import pandas as pd
import yfinance as yf


def download_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download daily data using yfinance and normalize column names.
    Returns a DataFrame with a DatetimeIndex and flattened columns.
    """
    df = yf.download(ticker, start=start, end=end, progress=False)
    # flatten multiindex columns from yfinance (if present)
    if isinstance(df.columns, pd.MultiIndex):
        try:
            # often level 0 is variable like 'Open' and level 1 is ticker; prefer variable names
            df.columns = df.columns.get_level_values(0)
        except Exception:
            df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    # Ensure index is datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    # Some providers label adjusted close as 'Adj Close'; if it's missing, fallback to 'Close'
    if 'Adj Close' not in df.columns and 'Close' in df.columns:
        df['Adj Close'] = df['Close']
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning: drop duplicate index, coerce numeric columns, fill small gaps."""
    df = df.copy()
    # drop duplicate index entries
    df = df[~df.index.duplicated(keep='first')]
    # ensure numeric columns are numeric; handle unexpected 2-D cases defensively
    for col in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']:
        if col in df.columns:
            series = df[col]
            if isinstance(series, pd.DataFrame):
                # if it is a single-column DataFrame, squeeze it
                if series.shape[1] == 1:
                    series = series.iloc[:, 0]
                else:
                    # leave as-is and try to coerce element-wise by converting to string
                    series = series.astype(str)
            df[col] = pd.to_numeric(series, errors='coerce')
    # fill small gaps and drop remaining NA
    df = df.sort_index()
    df = df.ffill().bfill()
    df = df.dropna()
    return df


def prepare_dataset(df: pd.DataFrame, feature_cols: List[str], target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Split DataFrame into X and y for modeling.

    Returns X (features) and y (target), aligned by index. Assumes df has target_col.
    """
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    # drop rows with any NA
    valid_idx = X.dropna().index.intersection(y.dropna().index)
    X = X.loc[valid_idx]
    y = y.loc[valid_idx]
    return X, y
