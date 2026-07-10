"""Simple backtesting utilities for the notebook.

These functions assume daily returns and a very simple strategy: go fully to cash or fully long.
This updated version supports an optional per-trade transaction cost (fractional cost of the trade
applied when the position changes) and a simple position sizing (fraction of portfolio to allocate).
"""
from typing import Tuple
import pandas as pd
import numpy as np


def generate_signals(predictions: pd.Series) -> pd.Series:
    """Convert model predictions (0/1) into positions: 1 means long, 0 means cash.

    Aligns with the index of predictions and returns a series of positions.
    """
    return predictions.astype(int)


def calculate_returns(price: pd.Series, positions: pd.Series, position_size: float = 1.0,
                      transaction_cost: float = 0.0) -> pd.DataFrame:
    """Given price series and positions (0 or 1), calculate daily returns for market and strategy.

    Parameters
    ----------
    price : pd.Series
        Price (Adj Close) indexed by date.
    positions : pd.Series
        Series of positions (0 or 1) aligned to price index.
    position_size : float
        Fraction of portfolio allocated when long (0..1). Default 1.0 = full allocation.
    transaction_cost : float
        Fractional transaction cost applied on turnover when position changes (e.g., 0.001 = 0.1%).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: Price, MarketReturn, Position, StrategyReturn, Turnover, Cost
    """
    df = pd.DataFrame(index=price.index)
    df['Price'] = price
    df['MarketReturn'] = price.pct_change().fillna(0)
    # align positions and fill missing as 0
    df['Position'] = positions.reindex(df.index).fillna(0).astype(float)
    # effective allocation size
    df['Alloc'] = df['Position'] * float(position_size)
    # turnover is absolute change in allocation from previous day
    df['Turnover'] = df['Alloc'].diff().abs().fillna(df['Alloc'].abs())
    # cost is turnover times transaction cost
    df['Cost'] = df['Turnover'] * transaction_cost
    # Strategy return uses previous day's allocation (we assume signals are generated end-of-day and executed next open)
    df['StrategyReturn'] = df['Alloc'].shift(1).fillna(0) * df['MarketReturn'] - df['Cost']
    return df


def calculate_drawdown(cum_returns: pd.Series) -> pd.Series:
    """Compute drawdown series from cumulative returns series.

    Drawdown = cumulative_return / rolling_max - 1
    """
    running_max = cum_returns.cummax()
    drawdown = cum_returns / running_max - 1
    return drawdown


def performance_summary(df: pd.DataFrame) -> dict:
    """Compute simple performance metrics for market and strategy.

    Expects df with columns MarketReturn and StrategyReturn (daily arithmetic returns).
    Returns a small dict with cumulative returns, annualized returns, volatility, win rate, and max drawdown.
    """
    results = {}
    for col in ['MarketReturn', 'StrategyReturn']:
        if col not in df.columns:
            continue
        daily = df[col].fillna(0)
        # cumulative arithmetic returns
        cum = (1 + daily).cumprod() - 1
        total_return = cum.iloc[-1]
        ann_return = (1 + total_return) ** (252 / len(daily)) - 1 if len(daily) > 0 else 0
        ann_vol = daily.std() * np.sqrt(252)
        win_rate = (daily > 0).mean()
        drawdown = calculate_drawdown((1 + daily).cumprod())
        max_dd = drawdown.min()
        results[col] = {
            'TotalReturn': float(total_return),
            'AnnualizedReturn': float(ann_return),
            'AnnualizedVolatility': float(ann_vol),
            'WinRate': float(win_rate),
            'MaxDrawdown': float(max_dd)
        }
    return results


def equity_curve(df: pd.DataFrame, return_col: str = 'StrategyReturn', initial_capital: float = 1.0) -> pd.Series:
    """Calculate equity curve (portfolio value) from daily returns.

    initial_capital is the starting value (default 1.0 for normalized curve). Returns a series of portfolio values.
    """
    returns = df[return_col].fillna(0)
    equity = (1 + returns).cumprod() * initial_capital
    return equity
