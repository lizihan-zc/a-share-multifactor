"""组合构造与回测工具。"""

from .single_factor import (
    annualized_return,
    build_factor_return_panel,
    calculate_cumulative_returns,
    calculate_quintile_returns,
    calculate_return_metrics,
)

__all__ = [
    "annualized_return",
    "build_factor_return_panel",
    "calculate_cumulative_returns",
    "calculate_quintile_returns",
    "calculate_return_metrics",
]
