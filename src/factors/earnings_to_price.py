"""盈利市值比（EP）因子。

定义：``EP = 归母净利润 TTM / 总市值``。
作用：衡量公司盈利相对于市场估值的高低。其他条件相同时，EP 越高通常表示
股票价格相对于其盈利越便宜，因此常作为价值类选股信号；净利润为负时，其
价值解释会减弱。
"""

from __future__ import annotations

import pandas as pd

from ._utils import require_columns, safe_ratio


FACTOR_NAME = "ep"


def calculate_earnings_to_price(
    panel: pd.DataFrame,
    *,
    profit_column: str = "net_profit_ttm",
    market_cap_column: str = "market_cap",
) -> pd.Series:
    """逐行计算 ``归母净利润 TTM / 总市值``。"""

    require_columns(
        panel, {profit_column, market_cap_column}, dataset_name="panel"
    )
    return safe_ratio(
        panel[profit_column], panel[market_cap_column], name=FACTOR_NAME
    )


calculate_ep = calculate_earnings_to_price
