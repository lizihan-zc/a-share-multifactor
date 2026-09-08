"""Notebook 05 使用的单因子五分位数分层回测函数。"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


def _factor_columns(columns: Iterable[str]) -> tuple[str, ...]:
    factors = tuple(columns)
    if not factors:
        raise ValueError("factor_columns 不能为空")
    if len(factors) != len(set(factors)):
        raise ValueError("factor_columns 不能包含重复列")
    return factors


def _finite_numeric(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )


def calculate_quintile_returns(
    factor_panel: pd.DataFrame,
    factor_columns: Iterable[str],
    *,
    date_column: str = "date",
    stock_column: str = "stock_code",
    return_column: str = "future_return_1m",
    n_quantiles: int = 5,
    min_observations: int = 30,
) -> pd.DataFrame:
    """逐月按因子从低到高等频分组，并计算下一期等权组合收益。

    因子值相同时使用股票代码作为稳定的次级排序键，使结果可重复。每个因子、
    每个月独立删除因子或收益缺失的股票；有效样本不足时不生成该月结果。

    返回长表中的 ``date`` 是信号形成日，``portfolio_return`` 是从该信号日到
    下一调仓日的简单收益。Q1 因子最低，Q5（当 ``n_quantiles=5`` 时）最高。
    """

    factors = _factor_columns(factor_columns)
    if n_quantiles < 2:
        raise ValueError("n_quantiles 至少为 2")
    if min_observations < n_quantiles:
        raise ValueError("min_observations 不能小于 n_quantiles")
    if not isinstance(factor_panel, pd.DataFrame):
        raise TypeError("factor_panel 必须是 pandas DataFrame")
    required = {date_column, stock_column, return_column, *factors}
    missing = sorted(required.difference(factor_panel.columns))
    if missing:
        raise ValueError(f"factor_panel 缺少必需字段：{missing}")

    panel = factor_panel.loc[
        :, [date_column, stock_column, return_column, *factors]
    ].copy()
    panel[date_column] = pd.to_datetime(panel[date_column], errors="coerce")
    panel[stock_column] = panel[stock_column].astype("string").str.strip().str.upper()
    if panel[date_column].isna().any() or panel[stock_column].isna().any():
        raise ValueError("factor_panel 包含无效的日期或股票代码")
    if panel.duplicated([date_column, stock_column]).any():
        raise ValueError("factor_panel 包含重复的 (date, stock_code) 主键")
    panel[return_column] = _finite_numeric(panel[return_column])
    for factor in factors:
        panel[factor] = _finite_numeric(panel[factor])

    output: list[pd.DataFrame] = []
    for factor in factors:
        valid = panel.loc[
            panel[[factor, return_column]].notna().all(axis=1),
            [date_column, stock_column, factor, return_column],
        ]
        for date, cross_section in valid.groupby(
            date_column, sort=True, observed=True
        ):
            if len(cross_section) < min_observations:
                continue
            ranked = cross_section.sort_values(
                [factor, stock_column], kind="mergesort"
            ).copy()
            # floor(i * K / N) 可得到数量尽可能相等的 K 个连续排名组。
            ranked["quantile_number"] = (
                np.arange(len(ranked), dtype=np.int64) * n_quantiles
                // len(ranked)
                + 1
            )
            grouped = (
                ranked.groupby("quantile_number", sort=True, observed=True)[
                    return_column
                ]
                .agg(portfolio_return="mean", n_stocks="size")
                .reset_index()
            )
            if len(grouped) != n_quantiles:
                continue
            grouped.insert(0, "factor", factor)
            grouped.insert(0, "date", date)
            grouped["quantile"] = "Q" + grouped["quantile_number"].astype(str)
            output.append(grouped)

    columns = [
        "date",
        "factor",
        "quantile",
        "quantile_number",
        "portfolio_return",
        "n_stocks",
    ]
    if not output:
        return pd.DataFrame(columns=columns)
    return (
        pd.concat(output, ignore_index=True)
        .loc[:, columns]
        .sort_values(["factor", "date", "quantile_number"])
        .reset_index(drop=True)
    )


def build_factor_return_panel(
    quintile_returns: pd.DataFrame,
    factor: str,
    *,
    n_quantiles: int = 5,
) -> pd.DataFrame:
    """将一个因子的分组收益转为宽表，并添加最高组减最低组收益。"""

    if not isinstance(quintile_returns, pd.DataFrame):
        raise TypeError("quintile_returns 必须是 pandas DataFrame")
    required = {"date", "factor", "quantile", "portfolio_return"}
    missing = sorted(required.difference(quintile_returns.columns))
    if missing:
        raise ValueError(f"quintile_returns 缺少必需字段：{missing}")
    quantiles = [f"Q{number}" for number in range(1, n_quantiles + 1)]
    selected = quintile_returns.loc[quintile_returns["factor"].eq(factor)]
    if selected.empty:
        raise ValueError(f"没有因子 {factor!r} 的分组收益")
    if selected.duplicated(["date", "quantile"]).any():
        raise ValueError(f"因子 {factor!r} 存在重复的日期和分组")
    panel = selected.pivot(
        index="date", columns="quantile", values="portfolio_return"
    ).reindex(columns=quantiles)
    panel.index = pd.to_datetime(panel.index)
    panel = panel.sort_index().dropna(subset=[quantiles[0], quantiles[-1]])
    panel["Long-Short"] = panel[quantiles[-1]] - panel[quantiles[0]]
    return panel


def annualized_return(
    monthly_returns: pd.Series,
    *,
    periods_per_year: int = 12,
) -> float:
    """用复合增长率计算简单收益序列的年化收益。"""

    if periods_per_year < 1:
        raise ValueError("periods_per_year 必须为正数")
    returns = _finite_numeric(pd.Series(monthly_returns)).dropna()
    if returns.empty:
        return np.nan
    gross_returns = 1.0 + returns
    if gross_returns.le(0).any():
        return np.nan
    return float(gross_returns.prod() ** (periods_per_year / len(returns)) - 1.0)


def calculate_cumulative_returns(monthly_returns: pd.Series) -> pd.Series:
    """把简单收益复利累积为从零开始的累计收益序列。"""

    returns = _finite_numeric(pd.Series(monthly_returns)).dropna()
    return returns.add(1.0).cumprod().sub(1.0).rename("cumulative_return")


def calculate_return_metrics(
    monthly_returns: pd.Series,
    *,
    periods_per_year: int = 12,
) -> pd.Series:
    """计算年化收益、年化波动、零无风险利率 Sharpe、最大回撤和胜率。"""

    if periods_per_year < 1:
        raise ValueError("periods_per_year 必须为正数")
    returns = _finite_numeric(pd.Series(monthly_returns)).dropna()
    if returns.empty:
        return pd.Series(
            {
                "Annualized Return": np.nan,
                "Annualized Volatility": np.nan,
                "Sharpe Ratio": np.nan,
                "Maximum Drawdown": np.nan,
                "Win Rate": np.nan,
                "Months": 0,
            }
        )

    volatility = returns.std(ddof=1) * np.sqrt(periods_per_year)
    sharpe = (
        returns.mean() / returns.std(ddof=1) * np.sqrt(periods_per_year)
        if len(returns) > 1 and returns.std(ddof=1) > 0
        else np.nan
    )
    wealth = returns.add(1.0).cumprod()
    wealth_with_initial = pd.concat(
        [pd.Series([1.0]), wealth.reset_index(drop=True)], ignore_index=True
    )
    drawdown = wealth_with_initial.div(wealth_with_initial.cummax()).sub(1.0)
    return pd.Series(
        {
            "Annualized Return": annualized_return(
                returns, periods_per_year=periods_per_year
            ),
            "Annualized Volatility": float(volatility),
            "Sharpe Ratio": float(sharpe),
            "Maximum Drawdown": float(drawdown.min()),
            "Win Rate": float(returns.gt(0).mean()),
            "Months": int(len(returns)),
        }
    )
