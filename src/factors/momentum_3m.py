"""过去 3 个月的价格动量因子。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._utils import prepare_daily_prices, validate_factor_index


FACTOR_NAME = "momentum_3m"


def calculate_momentum_3m(
    price_daily: pd.DataFrame,
    factor_index: pd.DataFrame,
    *,
    lookback: int = 63,
    close_column: str = "close",
    adjustment_column: str = "adj_factor",
) -> pd.Series:
    """使用精确交易日的复权价格计算 ``P[t] / P[t-63] - 1``。"""

    if lookback < 1:
        raise ValueError("lookback 必须为正数")
    daily = prepare_daily_prices(
        price_daily,
        close_column=close_column,
        adjustment_column=adjustment_column,
    )
    return calculate_momentum_3m_from_prepared(
        daily, factor_index, lookback=lookback
    )


def calculate_momentum_3m_from_prepared(
    daily: pd.DataFrame,
    factor_index: pd.DataFrame,
    *,
    lookback: int = 63,
) -> pd.Series:
    """使用已规范且带交易日序号的日频行情计算 3 个月动量。"""

    if lookback < 1:
        raise ValueError("lookback 必须为正数")
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
        validate="one_to_one",
    ).merge(
        old,
        on=["stock_code", "_lookback_date"],
        how="left",
        validate="one_to_one",
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


calculate_mom_3m = calculate_momentum_3m
