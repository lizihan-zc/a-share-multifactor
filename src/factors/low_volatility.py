"""过去 60 个交易日的低波动因子。

定义：先计算过去 60 个交易日复权日收益率的标准差 ``VOL_60``，再令
``LOWVOL = -VOL_60``。
作用：衡量股票近期价格波动风险。取负号后，因子值越大表示波动越低，使其方向
与“数值越大、预期表现越好”的统一约定一致，可用于研究低波动异象和风险暴露。
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from ._utils import (
    add_daily_returns,
    add_strict_rolling_feature,
    align_daily_feature,
    prepare_daily_factor_data,
)


FACTOR_NAME = "low_volatility_60d"


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


def calculate_low_volatility(
    cleaned_price_daily: pd.DataFrame,
    factor_index: pd.DataFrame,
    *,
    window: int = 60,
    min_periods: Optional[int] = None,
    ddof: int = 1,
) -> pd.Series:
    """计算复权日收益率滚动波动率的相反数。

    默认 ``min_periods=window`` 时，窗口内每个交易日都必须有连续的有效价格。
    使用负号后，因子值越大代表波动越低，从而符合 README 约定的统一因子方向。
    """

    daily = prepare_daily_factor_data(cleaned_price_daily)
    daily = add_daily_returns(daily)
    return calculate_low_volatility_from_prepared(
        daily,
        factor_index,
        window=window,
        min_periods=min_periods,
        ddof=ddof,
    )


calculate_lowvol = calculate_low_volatility
