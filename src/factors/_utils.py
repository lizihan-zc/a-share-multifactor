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
    if positive_denominator:
        valid = denominator_values>0
    else:
        valid = denominator_values!=0
    result = numerator_values.div(denominator_values.where(valid))
    return result.replace([np.inf, -np.inf], np.nan).rename(name)


def build_factor_index(factor_index: pd.DataFrame) -> pd.DataFrame:
    """提取月末主键并记录原始行序，供日频因子合并后恢复顺序。"""

    keys = factor_index.loc[:, list(PANEL_KEYS)].copy()
    keys["_factor_row"] = np.arange(len(keys), dtype=np.int64)
    return keys


def prepare_daily_factor_data(
    cleaned_price_daily: pd.DataFrame,
    *,
    extra_columns: Iterable[str] = (),
) -> pd.DataFrame:
    """提取因子所需的已清洗日频字段，并添加全市场交易日序号。

    输入来自 Notebook 02 保存的清洗后日频面板，因此这里不再重复转换主键、检查
    重复记录、清洗数值或计算复权收盘价。
    """

    columns = [*PANEL_KEYS, "adjusted_close", *extra_columns]
    daily = cleaned_price_daily.loc[:, columns].copy()
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
    window: int,                # 窗口宽度
    min_periods: int,           # 窗口中有效值的最少个数
    feature_name: str,
    operation: str,
    ddof: int = 1,
) -> pd.DataFrame:
    """
    计算滚动统计量，并在完整窗口要求下拒绝跨越交易日缺口。
    本质上是 Pandas 滚动均值和滚动标准差的严格封装。
    """

    if window < 1:
        raise ValueError("window 必须为正数")
    if not 1 <= min_periods <= window:
        raise ValueError("min_periods 必须位于 1 和 window 之间")

    enriched = daily.copy()
    enriched["_rolling_input"] = values
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
    """
    按日期和股票代码提取月末因子值，并恢复月度面板原有的行顺序。
    这一步主要是防止因子值计算正确，但赋值时错配到另一只股票或另一个月份。
    """

    keys = build_factor_index(factor_index)
    source = daily.loc[:, [*PANEL_KEYS, feature_column]].copy()
    aligned = keys.merge(
        source,
        on=list(PANEL_KEYS),
        how="left",
        sort=False,
    ).sort_values("_factor_row")
    return pd.Series(
        aligned[feature_column].to_numpy(),
        index=factor_index.index,
        name=output_name,
        dtype="float64",
    )
