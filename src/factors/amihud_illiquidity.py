"""Amihud 非流动性因子。"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from ._utils import (
    add_daily_returns,
    add_strict_rolling_feature,
    align_daily_feature,
    numeric,
    prepare_daily_prices,
)


FACTOR_NAME = "amihud_illiquidity_20d"


def calculate_amihud_illiquidity(
    price_daily: pd.DataFrame,
    factor_index: pd.DataFrame,
    *,
    window: int = 20,
    min_periods: Optional[int] = None,
    amount_column: str = "amount",
    close_column: str = "close",
    adjustment_column: str = "adj_factor",
) -> pd.Series:
    """计算 ``abs(日收益率) / 成交额`` 的滚动均值。

    ``amount`` 应使用货币单位，本项目处理后的成交额单位为元。这里保留非流动性
    指标的原始方向，即数值越大代表流动性越差。
    """

    min_periods = window if min_periods is None else min_periods
    daily = prepare_daily_prices(
        price_daily,
        close_column=close_column,
        adjustment_column=adjustment_column,
        extra_columns=(amount_column,),
    )
    daily = add_daily_returns(daily)
    return calculate_amihud_illiquidity_from_prepared(
        daily,
        factor_index,
        window=window,
        min_periods=min_periods,
        amount_column=amount_column,
    )


def calculate_amihud_illiquidity_from_prepared(
    daily: pd.DataFrame,
    factor_index: pd.DataFrame,
    *,
    window: int = 20,
    min_periods: Optional[int] = None,
    amount_column: str = "amount",
) -> pd.Series:
    """使用已计算日收益率的行情表计算 Amihud 非流动性因子。"""

    min_periods = window if min_periods is None else min_periods
    amount = numeric(daily[amount_column])
    daily["_daily_illiquidity"] = daily["daily_return"].abs().div(
        amount.where(amount.gt(0))
    )
    daily = add_strict_rolling_feature(
        daily,
        daily["_daily_illiquidity"],
        window=window,
        min_periods=min_periods,
        feature_name=FACTOR_NAME,
        operation="mean",
    )
    return align_daily_feature(
        daily,
        factor_index,
        feature_column=FACTOR_NAME,
        output_name=FACTOR_NAME,
    )


calculate_illiq = calculate_amihud_illiquidity
