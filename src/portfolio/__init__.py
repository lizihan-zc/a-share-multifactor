"""组合构造与回测工具。"""

from .backtest import (
    BacktestResult,
    MarketData,
    build_portfolio_targets,
    load_market_data,
    run_portfolio_backtest,
    summarize_backtest,
)
from .single_factor import (
    annualized_return,
    build_factor_return_panel,
    calculate_cumulative_returns,
    calculate_quintile_returns,
    calculate_return_metrics,
)

__all__ = [
    "BacktestResult",
    "MarketData",
    "build_portfolio_targets",
    "load_market_data",
    "run_portfolio_backtest",
    "summarize_backtest",
    "annualized_return",
    "build_factor_return_panel",
    "calculate_cumulative_returns",
    "calculate_quintile_returns",
    "calculate_return_metrics",
]
