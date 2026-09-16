"""Notebook 04 使用的单因子横截面诊断函数。"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from ._utils import numeric, require_columns


def _validate_factor_columns(factor_columns: Iterable[str]) -> tuple[str, ...]:
    columns = tuple(factor_columns)
    if not columns:
        raise ValueError("factor_columns 不能为空")
    if len(columns) != len(set(columns)):
        raise ValueError("factor_columns 不能包含重复列")
    return columns


def calculate_monthly_rank_ic(
    factor_panel: pd.DataFrame,
    factor_columns: Iterable[str],
    *,
    date_column: str = "date",
    return_column: str = "future_return_1m",
    min_observations: int = 30,
) -> pd.DataFrame:
    """逐月计算因子值与下一期收益之间的 Spearman Rank IC。

    每个因子、每个月独立进行成对缺失值删除。有效样本不足、因子为常数或
    下一期收益为常数时返回缺失值，避免把无法识别的相关系数误记为零。
    """

    factors = _validate_factor_columns(factor_columns)
    if min_observations < 2:
        raise ValueError("min_observations 至少为 2")
    require_columns(
        factor_panel,
        {date_column, return_column, *factors},
        dataset_name="factor_panel",
    )

    panel = factor_panel.loc[:, [date_column, return_column, *factors]].copy()
    panel[date_column] = pd.to_datetime(panel[date_column], errors="coerce")
    panel[return_column] = numeric(panel[return_column])
    for factor in factors:
        panel[factor] = numeric(panel[factor])
    panel = panel.loc[panel[date_column].notna()]

    rows: list[dict[str, object]] = []
    for date, cross_section in panel.groupby(date_column, sort=True, observed=True):
        row: dict[str, object] = {date_column: date}
        for factor in factors:
            paired = cross_section[[factor, return_column]].dropna()
            if (
                len(paired) < min_observations
                or paired[factor].nunique() < 2
                or paired[return_column].nunique() < 2
            ):
                row[factor] = np.nan
            else:
                row[factor] = paired[factor].corr(
                    paired[return_column], method="spearman"
                )
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=factors, index=pd.DatetimeIndex([], name=date_column))
    return pd.DataFrame(rows).set_index(date_column).loc[:, list(factors)]


def summarize_rank_ic(monthly_rank_ic: pd.DataFrame) -> pd.DataFrame:
    """汇总 Mean Rank IC、IC Std、ICIR、正 IC 比例和常规均值 t 统计量。"""

    if not isinstance(monthly_rank_ic, pd.DataFrame):
        raise TypeError("monthly_rank_ic 必须是 pandas DataFrame")
    if monthly_rank_ic.columns.duplicated().any():
        raise ValueError("monthly_rank_ic 不能包含重复列")

    numeric_ic = monthly_rank_ic.apply(numeric)
    count = numeric_ic.count()
    mean = numeric_ic.mean()
    std = numeric_ic.std(ddof=1)
    valid_scale = std.where(std.gt(0))
    summary = pd.DataFrame(
        {
            "Mean Rank IC": mean,
            "IC Std": std,
            "ICIR": mean.div(valid_scale),
            "Positive IC %": numeric_ic.gt(0).sum().div(count.where(count.gt(0))),
            "IC t-stat": mean.div(valid_scale.div(np.sqrt(count))),
            "Months": count.astype("int64"),
        }
    )
    summary.index.name = "Factor"
    return summary


def calculate_average_factor_correlation(
    factor_panel: pd.DataFrame,
    factor_columns: Iterable[str],
    *,
    date_column: str = "date",
    method: str = "spearman",
    min_observations: int = 30,
) -> pd.DataFrame:
    """计算同一个月、不同因子之间的横截面相关性，再对月份做等权时间平均。"""

    factors = _validate_factor_columns(factor_columns)
    if method not in {"spearman", "pearson"}:
        raise ValueError("method 必须是 'spearman' 或 'pearson'")
    if min_observations < 2:
        raise ValueError("min_observations 至少为 2")
    require_columns(
        factor_panel,
        {date_column, *factors},
        dataset_name="factor_panel",
    )

    panel = factor_panel.loc[:, [date_column, *factors]].copy()
    panel[date_column] = pd.to_datetime(panel[date_column], errors="coerce")
    for factor in factors:
        panel[factor] = numeric(panel[factor])
    matrices = [
        cross_section.loc[:, list(factors)].corr(
            method=method, min_periods=min_observations
        )
        for _, cross_section in panel.loc[panel[date_column].notna()].groupby(
            date_column, sort=True, observed=True
        )
    ]
    if not matrices:
        return pd.DataFrame(np.nan, index=factors, columns=factors)
    average = sum(matrix.fillna(0) for matrix in matrices)
    available = sum(matrix.notna().astype("int64") for matrix in matrices)
    return average.div(available.where(available.gt(0))).loc[
        list(factors), list(factors)
    ]


def calculate_factor_rank_autocorrelation(
    factor_panel: pd.DataFrame,
    factor_columns: Iterable[str],
    *,
    date_column: str = "date",
    stock_column: str = "stock_code",
    min_observations: int = 30,
) -> pd.DataFrame:
    """计算同一个因子在相邻两个月之间的排名相关性。

    返回长表；``date`` 表示当前调仓日，``previous_date`` 表示与之比较的上一个
    调仓日。相关系数按因子成对删除缺失值，``n_stocks`` 是实际有效股票数。
    """

    factors = _validate_factor_columns(factor_columns)
    if min_observations < 2:
        raise ValueError("min_observations 至少为 2")
    require_columns(
        factor_panel,
        {date_column, stock_column, *factors},
        dataset_name="factor_panel",
    )

    panel = factor_panel.loc[:, [date_column, stock_column, *factors]].copy()
    panel[date_column] = pd.to_datetime(panel[date_column], errors="coerce")
    if panel[date_column].isna().any() or panel[stock_column].isna().any():
        raise ValueError("factor_panel 包含无效的日期或股票代码")
    if panel.duplicated([date_column, stock_column]).any():
        raise ValueError("factor_panel 包含重复的 (date, stock_code) 主键")
    for factor in factors:
        panel[factor] = numeric(panel[factor])

    dates = pd.Index(panel[date_column].drop_duplicates().sort_values())
    rows: list[dict[str, object]] = []
    for previous_date, date in zip(dates[:-1], dates[1:]):
        previous = panel.loc[
            panel[date_column].eq(previous_date), [stock_column, *factors]
        ]
        current = panel.loc[
            panel[date_column].eq(date), [stock_column, *factors]
        ]
        paired = previous.merge(
            current,
            on=stock_column,
            how="inner",
            suffixes=("_previous", "_current"),
            validate="one_to_one",
        )
        for factor in factors:
            columns = [f"{factor}_previous", f"{factor}_current"]
            valid = paired[columns].dropna()
            correlation = np.nan
            if (
                len(valid) >= min_observations
                and valid[columns[0]].nunique() >= 2
                and valid[columns[1]].nunique() >= 2
            ):
                correlation = valid[columns[0]].corr(
                    valid[columns[1]], method="spearman"
                )
            rows.append(
                {
                    "date": date,
                    "previous_date": previous_date,
                    "factor": factor,
                    "rank_autocorrelation": correlation,
                    "n_stocks": len(valid),
                }
            )

    return pd.DataFrame(
        rows,
        columns=[
            "date",
            "previous_date",
            "factor",
            "rank_autocorrelation",
            "n_stocks",
        ],
    )


__all__ = [
    "calculate_average_factor_correlation",
    "calculate_factor_rank_autocorrelation",
    "calculate_monthly_rank_ic",
    "summarize_rank_ic",
]
