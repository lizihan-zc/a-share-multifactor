"""过去 12 个月剔除最近 1 个月的价格动量因子。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._utils import prepare_daily_prices, validate_factor_index


FACTOR_NAME = "momentum_12_1"


def calculate_momentum_12_1(
    price_daily: pd.DataFrame,
    factor_index: pd.DataFrame,
    *,
    recent_skip: int = 21,
    long_lookback: int = 252,
    close_column: str = "close",
    adjustment_column: str = "adj_factor",
) -> pd.Series:
    """使用复权价格计算 ``P[t-21] / P[t-252] - 1``。

    回看期按全市场交易日确定。如果股票在任一精确回看日期缺少行情，则因子值设为
    缺失，不跨越停牌期寻找该股票自身的上一条可用行情。
    """

    if recent_skip < 0 or long_lookback <= recent_skip:
        raise ValueError("必须满足 0 <= recent_skip < long_lookback")
    daily = prepare_daily_prices(
        price_daily,
        close_column=close_column,
        adjustment_column=adjustment_column,
    )
    return calculate_momentum_12_1_from_prepared(
        daily,
        factor_index,
        recent_skip=recent_skip,
        long_lookback=long_lookback,
    )


def calculate_momentum_12_1_from_prepared(
    daily: pd.DataFrame,
    factor_index: pd.DataFrame,
    *,
    recent_skip: int = 21,
    long_lookback: int = 252,
) -> pd.Series:
    """使用已规范且带交易日序号的日频行情计算 12-1 动量。"""

    if recent_skip < 0 or long_lookback <= recent_skip:
        raise ValueError("必须满足 0 <= recent_skip < long_lookback")
    keys = validate_factor_index(factor_index)
    session_dates = (
        daily.loc[:, ["date", "_session"]]
        .drop_duplicates("_session")
        .set_index("_session")["date"]
    )
    date_to_session = pd.Series(
        session_dates.index.to_numpy(), index=session_dates.array
    )
    keys["_session"] = keys["date"].map(date_to_session)
    keys["_recent_date"] = keys["_session"].sub(recent_skip).map(session_dates)
    keys["_long_date"] = keys["_session"].sub(long_lookback).map(session_dates)

    prices = daily.loc[:, ["date", "stock_code", "adjusted_close"]]
    recent = prices.rename(
        columns={"date": "_recent_date", "adjusted_close": "_recent_price"}
    )
    long = prices.rename(
        columns={"date": "_long_date", "adjusted_close": "_long_price"}
    )
    values = keys.merge(
        recent,
        on=["stock_code", "_recent_date"],
        how="left",
        validate="one_to_one",
    ).merge(
        long,
        on=["stock_code", "_long_date"],
        how="left",
        validate="one_to_one",
    )
    factor = values["_recent_price"].div(values["_long_price"]).sub(1)
    factor = factor.replace([np.inf, -np.inf], np.nan)
    ordered = values.assign(_factor=factor).sort_values("_factor_row")
    return pd.Series(
        ordered["_factor"].to_numpy(),
        index=factor_index.index,
        name=FACTOR_NAME,
        dtype="float64",
    )


calculate_mom_12_1 = calculate_momentum_12_1
