"""账面市值比（BP）因子。"""

from __future__ import annotations

import pandas as pd

from ._utils import require_columns, safe_ratio


FACTOR_NAME = "bp"


def calculate_book_to_price(
    panel: pd.DataFrame,
    *,
    equity_column: str = "total_equity",
    market_cap_column: str = "market_cap",
) -> pd.Series:
    """逐行计算 ``账面权益 / 总市值``。"""

    require_columns(
        panel, {equity_column, market_cap_column}, dataset_name="panel"
    )
    return safe_ratio(
        panel[equity_column], panel[market_cap_column], name=FACTOR_NAME
    )


calculate_bp = calculate_book_to_price
