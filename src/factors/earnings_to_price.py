"""盈利市值比（EP）因子。"""

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
