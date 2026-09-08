"""净资产收益率（ROE）质量因子。"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from ._utils import numeric, require_columns, safe_ratio


FACTOR_NAME = "roe"


def calculate_average_equity(
    panel: pd.DataFrame,
    *,
    equity_column: str = "total_equity",
    report_period_column: str = "end_date",
    date_column: str = "date",
    stock_column: str = "stock_code",
) -> pd.Series:
    """使用当前权益与上年同期权益估算 TTM 平均权益。

    上年同期权益取因子计算日之前最近一次可见的面板记录。如果输入面板包含同一财报
    的多个历史版本，这一匹配方式仍能保持严格的时点可得性。
    """

    required = {
        equity_column,
        report_period_column,
        date_column,
        stock_column,
    }
    require_columns(panel, required, dataset_name="panel")

    working = panel.loc[:, list(required)].copy()
    working["_row"] = np.arange(len(working), dtype=np.int64)
    working[date_column] = pd.to_datetime(working[date_column], errors="coerce")
    working[report_period_column] = pd.to_datetime(
        working[report_period_column], errors="coerce"
    )
    working[stock_column] = (
        working[stock_column].astype("string").str.strip().str.upper()
    )
    working[equity_column] = numeric(working[equity_column])

    left = working.loc[
        working[date_column].notna()
        & working[report_period_column].notna()
        & working[stock_column].notna(),
        ["_row", date_column, stock_column, report_period_column, equity_column],
    ].copy()
    left["_prior_report_period"] = left[report_period_column] - pd.DateOffset(years=1)

    history = working.loc[
        working[date_column].notna()
        & working[report_period_column].notna()
        & working[stock_column].notna()
        & working[equity_column].notna(),
        [date_column, stock_column, report_period_column, equity_column],
    ].rename(
        columns={
            date_column: "_equity_available_date",
            report_period_column: "_prior_report_period",
            equity_column: "_prior_equity",
        }
    )

    # merge_asof 要求时间连接字段在全表范围内有序。
    left = left.sort_values([date_column, stock_column, "_prior_report_period"])
    history = history.sort_values(
        ["_equity_available_date", stock_column, "_prior_report_period"]
    )
    if left.empty or history.empty:
        prior = pd.DataFrame({"_row": left["_row"], "_prior_equity": np.nan})
    else:
        prior = pd.merge_asof(
            left,
            history,
            left_on=date_column,
            right_on="_equity_available_date",
            by=[stock_column, "_prior_report_period"],
            direction="backward",
            allow_exact_matches=True,
        )[["_row", "_prior_equity"]]

    result = pd.Series(np.nan, index=np.arange(len(panel)), dtype="float64")
    if not prior.empty:
        result.iloc[prior["_row"].to_numpy(dtype=np.int64)] = numeric(
            prior["_prior_equity"]
        ).to_numpy()
    current = numeric(panel[equity_column]).reset_index(drop=True)
    average = current.add(result).div(2)
    average = average.where(current.gt(0) & result.gt(0))
    return pd.Series(
        average.to_numpy(), index=panel.index, name="average_equity", dtype="float64"
    )


def calculate_roe(
    panel: pd.DataFrame,
    *,
    profit_column: str = "net_profit_ttm",
    average_equity_column: Optional[str] = None,
    equity_column: str = "total_equity",
    report_period_column: str = "end_date",
) -> pd.Series:
    """计算 ``归母净利润 TTM / 平均权益``。

    如果数据源已提供可信且满足时点要求的平均权益，可通过 ``average_equity_column``
    指定该字段；否则调用 :func:`calculate_average_equity`，根据当前权益和上年同期
    权益计算平均值。
    """

    require_columns(panel, {profit_column}, dataset_name="panel")
    if average_equity_column is not None:
        require_columns(panel, {average_equity_column}, dataset_name="panel")
        average_equity = panel[average_equity_column]
    else:
        average_equity = calculate_average_equity(
            panel,
            equity_column=equity_column,
            report_period_column=report_period_column,
        )
    return safe_ratio(panel[profit_column], average_equity, name=FACTOR_NAME)
