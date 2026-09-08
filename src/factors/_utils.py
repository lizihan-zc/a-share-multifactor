"""因子计算共用的数据校验与日频行情辅助函数。"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


PANEL_KEYS = ("date", "stock_code")


def require_columns(
    frame: pd.DataFrame,
    columns: Iterable[str],
    *,
    dataset_name: str,
) -> None:
    """检查输入是否为 DataFrame，并确认包含当前计算所需的全部字段。"""

    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{dataset_name} 必须是 pandas DataFrame")
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{dataset_name} 缺少必需字段：{missing}")


def numeric(values: pd.Series) -> pd.Series:
    """将输入转为有限浮点数，无法解析或无穷的值设为缺失。"""

    return pd.to_numeric(values, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )


def safe_ratio(
    numerator: pd.Series,
    denominator: pd.Series,
    *,
    positive_denominator: bool = True,
    name: str,
) -> pd.Series:
    """计算有限数值之比，并将不符合约束的分母对应结果设为缺失。"""

    numerator_values = numeric(numerator)
    denominator_values = numeric(denominator)
    valid = (
        denominator_values.gt(0)
        if positive_denominator
        else denominator_values.ne(0)
    )
    result = numerator_values.div(denominator_values.where(valid))
    return result.replace([np.inf, -np.inf], np.nan).rename(name)


def validate_factor_index(
    factor_index: pd.DataFrame,
    *,
    dataset_name: str = "factor_index",
) -> pd.DataFrame:
    """规范用于把日频因子对齐至月末截面的日期与股票代码。"""

    require_columns(factor_index, PANEL_KEYS, dataset_name=dataset_name)
    keys = factor_index.loc[:, list(PANEL_KEYS)].copy()
    keys["date"] = pd.to_datetime(keys["date"], errors="coerce")
    keys["stock_code"] = keys["stock_code"].astype("string").str.strip().str.upper()
    if keys["date"].isna().any() or keys["stock_code"].isna().any():
        raise ValueError(f"{dataset_name} 包含无效的 date 或 stock_code")
    if keys.duplicated(list(PANEL_KEYS)).any():
        raise ValueError(f"{dataset_name} 包含重复的 (date, stock_code) 主键")
    keys["_factor_row"] = np.arange(len(keys), dtype=np.int64)
    return keys


def prepare_daily_prices(
    price_daily: pd.DataFrame,
    *,
    close_column: str = "close",
    adjustment_column: str = "adj_factor",
    extra_columns: Iterable[str] = (),
) -> pd.DataFrame:
    """规范日频行情，并添加复权收盘价和全市场交易日序号。

    全市场交易日序号由 ``price_daily`` 中出现的全部日期生成。这样回看窗口表示
    真实的市场交易日，而不是某只股票自身的前若干条记录，避免窗口静默跨过停牌期。
    """

    required = {*PANEL_KEYS, close_column, adjustment_column, *extra_columns}
    require_columns(price_daily, required, dataset_name="price_daily")
    daily = price_daily.loc[:, list(required)].copy()
    daily["date"] = pd.to_datetime(daily["date"], errors="coerce")
    daily["stock_code"] = (
        daily["stock_code"].astype("string").str.strip().str.upper()
    )
    if daily["date"].isna().any() or daily["stock_code"].isna().any():
        raise ValueError("price_daily 包含无效的 date 或 stock_code")
    if daily.duplicated(list(PANEL_KEYS)).any():
        raise ValueError("price_daily 包含重复的 (date, stock_code) 主键")

    close = numeric(daily[close_column])
    adjustment = numeric(daily[adjustment_column])
    daily["adjusted_close"] = (close * adjustment).where(
        close.gt(0) & adjustment.gt(0)
    )
    sessions = pd.Index(daily["date"].drop_duplicates().sort_values())
    session_map = pd.Series(np.arange(len(sessions), dtype=np.int64), index=sessions)
    daily["_session"] = daily["date"].map(session_map).astype(np.int64)
    return daily.sort_values(["stock_code", "date"]).reset_index(drop=True)


def add_daily_returns(daily: pd.DataFrame) -> pd.DataFrame:
    """仅对相邻的全市场交易日计算复权收盘价收益率。"""

    enriched = daily.copy()
    grouped = enriched.groupby("stock_code", sort=False, observed=True)
    previous_price = grouped["adjusted_close"].shift(1)
    previous_session = grouped["_session"].shift(1)
    consecutive = enriched["_session"].sub(previous_session).eq(1)
    enriched["daily_return"] = (
        enriched["adjusted_close"].div(previous_price).sub(1).where(consecutive)
    )
    return enriched


def add_strict_rolling_feature(
    daily: pd.DataFrame,
    values: pd.Series,
    *,
    window: int,
    min_periods: int,
    feature_name: str,
    operation: str,
    ddof: int = 1,
) -> pd.DataFrame:
    """计算滚动统计量，并在完整窗口要求下拒绝跨越交易日缺口。"""

    if window < 1:
        raise ValueError("window 必须为正数")
    if not 1 <= min_periods <= window:
        raise ValueError("min_periods 必须位于 1 和 window 之间")

    enriched = daily.copy()
    enriched["_rolling_input"] = numeric(values)
    grouped = enriched.groupby("stock_code", sort=False, observed=True)
    rolling = grouped["_rolling_input"].rolling(
        window=window, min_periods=min_periods
    )
    if operation == "mean":
        statistic = rolling.mean()
    elif operation == "std":
        statistic = rolling.std(ddof=ddof)
    else:
        raise ValueError(f"不支持的滚动计算：{operation}")
    statistic = statistic.reset_index(level=0, drop=True).sort_index()

    first_session = grouped["_session"].shift(window - 1)
    full_session_span = enriched["_session"].sub(first_session).eq(window - 1)
    if min_periods == window:
        statistic = statistic.where(full_session_span)
    enriched[feature_name] = statistic
    return enriched.drop(columns="_rolling_input")


def align_daily_feature(
    daily: pd.DataFrame,
    factor_index: pd.DataFrame,
    *,
    feature_column: str,
    output_name: str,
) -> pd.Series:
    """按精确日期将日频特征对齐到因子截面的原始行序。"""

    keys = validate_factor_index(factor_index)
    source = daily.loc[:, [*PANEL_KEYS, feature_column]].copy()
    aligned = keys.merge(
        source,
        on=list(PANEL_KEYS),
        how="left",
        validate="one_to_one",
        sort=False,
    ).sort_values("_factor_row")
    return pd.Series(
        aligned[feature_column].to_numpy(),
        index=factor_index.index,
        name=output_name,
        dtype="float64",
    )
