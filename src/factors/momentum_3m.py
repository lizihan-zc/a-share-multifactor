"""过去 3 个月的价格动量因子。

定义：``MOM_3M = P[t] / P[t-63] - 1``，价格使用复权收盘价，63 表示约三个
月的市场交易日。

作用：衡量股票近期的价格趋势和相对强弱。因子值越高表示过去三个月表现越强，
可与 12-1 动量比较不同回看周期的有效性和稳定性。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._utils import build_factor_index, prepare_daily_factor_data


FACTOR_NAME = "momentum_3m"


def calculate_momentum_3m_from_prepared(
    daily: pd.DataFrame,
    factor_index: pd.DataFrame,
    *,
    lookback: int = 63,
) -> pd.Series:
    """使用已规范且带交易日序号的日频行情计算 3 个月动量。"""

    if lookback < 1:
        raise ValueError("lookback 必须为正数")
    keys = build_factor_index(factor_index)
    session_dates = (
        daily.loc[:, ["date", "_session"]]
        .drop_duplicates("_session")
        .set_index("_session")["date"]
    )
    date_to_session = pd.Series(
        session_dates.index.to_numpy(), index=session_dates.array
    )
    keys["_session"] = keys["date"].map(date_to_session)
    keys["_lookback_date"] = keys["_session"].sub(lookback).map(session_dates)

    prices = daily.loc[:, ["date", "stock_code", "adjusted_close"]]
    current = prices.rename(columns={"adjusted_close": "_current_price"})
    old = prices.rename(
        columns={"date": "_lookback_date", "adjusted_close": "_old_price"}
    )
    values = keys.merge(
        current,
        on=["stock_code", "date"],
        how="left",
    ).merge(
        old,
        on=["stock_code", "_lookback_date"],
        how="left",
    )
    factor = values["_current_price"].div(values["_old_price"]).sub(1)
    factor = factor.replace([np.inf, -np.inf], np.nan)
    ordered = values.assign(_factor=factor).sort_values("_factor_row")
    return pd.Series(
        ordered["_factor"].to_numpy(),
        index=factor_index.index,
        name=FACTOR_NAME,
        dtype="float64",
    )



def calculate_momentum_3m(
    cleaned_price_daily: pd.DataFrame,
    factor_index: pd.DataFrame,
    *,
    lookback: int = 63,
) -> pd.Series:
    """使用精确交易日的复权价格计算 ``P[t] / P[t-63] - 1``。"""

    daily = prepare_daily_factor_data(cleaned_price_daily)
    return calculate_momentum_3m_from_prepared(
        daily, factor_index, lookback=lookback
    )


calculate_mom_3m = calculate_momentum_3m
