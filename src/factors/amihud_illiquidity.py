"""Amihud 非流动性因子。

定义：该因子分两步计算。首先计算股票每天的价格冲击
``daily_illiq[d] = abs(return[d]) / amount[d]``；然后在每个计算日，对最近 20 个
连续交易日的 ``daily_illiq`` 取滚动平均：
``ILLIQ_20[t] = mean(daily_illiq[t-19:t])``。默认要求窗口内 20 个交易日全部连续
且数值有效，否则当日因子值为缺失。成交额 ``amount`` 的单位为元。

作用：衡量单位成交额引起的价格变动幅度。因子值越高表示较小成交金额也可能
造成较大价格变化，即流动性越差；可用于检验因子收益是否源于流动性风险补偿。
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


FACTOR_NAME = "amihud_illiquidity_20d"


def calculate_amihud_illiquidity_from_prepared(
    daily: pd.DataFrame,
    factor_index: pd.DataFrame,
    *,
    window: int = 20,
    min_periods: Optional[int] = None,
    amount_column: str = "amount",
) -> pd.Series:
    """
    使用已计算日收益率的行情表计算 Amihud 非流动性因子。
    它在已经准备好日频数据或批量计算多个因子时使用，避免重复计算。
    """

    min_periods = window if min_periods is None else min_periods

    prepared = daily.copy()
    amount = prepared[amount_column]

    prepared["_daily_illiquidity"] = prepared["daily_return"].abs().div(
        amount.where(amount>0)
    )

    prepared = add_strict_rolling_feature(
        prepared,
        prepared["_daily_illiquidity"],
        window=window,
        min_periods=min_periods,
        feature_name=FACTOR_NAME,
        operation="mean",
    )

    return align_daily_feature(
        prepared,
        factor_index,
        feature_column=FACTOR_NAME,
        output_name=FACTOR_NAME,
    )


def calculate_amihud_illiquidity(
    cleaned_price_daily: pd.DataFrame,
    factor_index: pd.DataFrame,
    *,
    window: int = 20,
    min_periods: Optional[int] = None,
    amount_column: str = "amount",
) -> pd.Series:
    """
    计算 ``abs(日收益率) / 成交额`` 的滚动均值。
    它接收 Notebook 02 保存的清洗后日频面板，完成完整流程。
    主要用于测试，在 Notebook 03 的正式工作流中没有使用它。

    ``amount`` 应使用货币单位，本项目处理后的成交额单位为元。这里保留非流动性
    指标的原始方向，即数值越大代表流动性越差。
    """

    daily = prepare_daily_factor_data(
        cleaned_price_daily,
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
