"""过去 60 个交易日的低波动因子。"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from ._utils import (
    add_daily_returns,
    add_strict_rolling_feature,
    align_daily_feature,
    prepare_daily_prices,
)


FACTOR_NAME = "low_volatility_60d"


def calculate_low_volatility(
    price_daily: pd.DataFrame,
    factor_index: pd.DataFrame,
    *,
    window: int = 60,
    min_periods: Optional[int] = None,
    ddof: int = 1,
    close_column: str = "close",
    adjustment_column: str = "adj_factor",
) -> pd.Series:
    """计算复权日收益率滚动波动率的相反数。

    默认 ``min_periods=window`` 时，窗口内每个交易日都必须有连续的有效价格。
    使用负号后，因子值越大代表波动越低，从而符合 README 约定的统一因子方向。
    """

    min_periods = window if min_periods is None else min_periods
    daily = prepare_daily_prices(
        price_daily,
        close_column=close_column,
        adjustment_column=adjustment_column,
    )
    daily = add_daily_returns(daily)
    return calculate_low_volatility_from_prepared(
        daily,
        factor_index,
        window=window,
        min_periods=min_periods,
        ddof=ddof,
    )


def calculate_low_volatility_from_prepared(
    daily: pd.DataFrame,
    factor_index: pd.DataFrame,
    *,
    window: int = 60,
    min_periods: Optional[int] = None,
    ddof: int = 1,
) -> pd.Series:
    """使用已计算日收益率的行情表计算低波动因子。"""

    min_periods = window if min_periods is None else min_periods
    daily = add_strict_rolling_feature(
        daily,
        daily["daily_return"],
        window=window,
        min_periods=min_periods,
        feature_name="_volatility",
        operation="std",
        ddof=ddof,
    )
    daily[FACTOR_NAME] = -daily["_volatility"]
    return align_daily_feature(
        daily,
        factor_index,
        feature_column=FACTOR_NAME,
        output_name=FACTOR_NAME,
    )


calculate_lowvol = calculate_low_volatility
