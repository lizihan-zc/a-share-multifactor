"""综合评分的分层持仓、可观察收益及目标持仓换手率。

分组与收益评估分成两步：先用当期评分确定持仓，再连接未来收益。
这样不会因为事后缺少收益标签而把其他股票补进最高组。
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


def build_quantile_holdings(
    score_panel: pd.DataFrame,
    score_columns: Iterable[str],
    *,
    n_quantiles: int = 5,
    min_observations: int = 30,
) -> pd.DataFrame:
    """按当期评分分组，返回各组等权目标持仓；不使用任何收益标签。

    Q1 最低、Q5 最高。分组算法与 Notebook 05 相同：股票代码是评分并列
    时的次级排序键；floor(i*K/N) 让各组数量尽可能相等。
    """

    if n_quantiles < 2 or min_observations < n_quantiles:
        raise ValueError("需要 n_quantiles >= 2 且 min_observations >= n_quantiles")

    parts = []
    for score in score_columns:
        sample = score_panel.loc[
            score_panel["is_eligible"].fillna(False) & score_panel[score].notna(),
            ["date", "stock_code", score],
        ]
        for date, cross_section in sample.groupby("date", sort=True, observed=True):
            if len(cross_section) < min_observations:
                continue
            ranked = cross_section.sort_values([score, "stock_code"]).copy()
            ranked["quantile_number"] = (
                np.arange(len(ranked)) * n_quantiles // len(ranked) + 1
            )
            ranked["weight"] = 1.0 / ranked.groupby("quantile_number")[
                "stock_code"
            ].transform("size")
            ranked["factor"] = score
            ranked["quantile"] = "Q" + ranked["quantile_number"].astype(str)
            parts.append(ranked.drop(columns=score))

    columns = ["date", "stock_code", "factor", "quantile_number", "quantile", "weight"]
    if not parts:
        return pd.DataFrame(columns=columns)
    return pd.concat(parts, ignore_index=True).loc[:, columns]


def calculate_holdings_returns(
    holdings: pd.DataFrame,
    label_panel: pd.DataFrame,
) -> pd.DataFrame:
    """连接既定持仓与下一期标签，计算各组可观察收益和标签覆盖率。

    沿用本项目标签资格 is_in_label_sample。收益缺失时保留原持仓，但仅用
    有效标签计算收益，并将其权重重新归一化；因此 portfolio_return 是
    “可观察持仓的条件平均收益”，不是包含停牌/退市处理的完整策略收益。
    label_coverage 记录有效标签占原目标权重的比例，缺失收益绝不视作零。
    """

    labels = label_panel[
        ["date", "stock_code", "future_return_1m", "is_in_label_sample"]
    ]
    joined = holdings.merge(labels, on=["date", "stock_code"], how="left",
                            validate="many_to_one")
    observable = joined["is_in_label_sample"].fillna(False) & joined[
        "future_return_1m"
    ].notna()
    joined["observed_return"] = joined["future_return_1m"].where(observable)
    joined["observed_weight"] = joined["weight"].where(observable, 0.0)
    joined["observed_pnl"] = joined["weight"] * joined["observed_return"]

    result = joined.groupby(
        ["date", "factor", "quantile_number", "quantile"], observed=True, sort=True
    ).agg(
        n_stocks=("stock_code", "size"),
        n_observed=("observed_return", "count"),
        label_coverage=("observed_weight", "sum"),
        observed_pnl=("observed_pnl", "sum"),
    ).reset_index()
    # 无有效标签时分母为 NaN，确保最后一个右截尾月份不会产生虚假的零收益。
    result["portfolio_return"] = result["observed_pnl"].div(
        result["label_coverage"].where(result["label_coverage"].gt(0))
    )
    return result.drop(columns="observed_pnl")


def calculate_target_turnover(
    holdings: pd.DataFrame,
    signal_dates: pd.DatetimeIndex,
    *,
    quantile_number: int = 5,
) -> pd.DataFrame:
    """计算相邻正式月份最高组的目标权重换手：0.5*sum(abs(w_t-w_t-1))。

    这里比较两次目标权重，尚未计算期间收益造成的权重漂移。首个建仓月
    不计入均值；当中断一个月份时，恢复后的月份也不跨越中断计算换手。
    不读取未来收益，因此包含事后标签缺失的股票及最后一个信号月。
    """

    top = holdings.loc[holdings["quantile_number"].eq(quantile_number)]
    rows = []
    for factor, sample in top.groupby("factor", sort=True, observed=True):
        by_date = {
            date: cross_section.set_index("stock_code")["weight"]
            for date, cross_section in sample.groupby("date", observed=True)
        }
        previous = None
        for date in signal_dates:
            current = by_date.get(date)
            turnover = np.nan
            if current is not None and previous is not None:
                current_aligned, previous_aligned = current.align(
                    previous, join="outer", fill_value=0.0
                )
                turnover = 0.5 * (current_aligned - previous_aligned).abs().sum()
            rows.append({
                "date": date,
                "factor": factor,
                "target_turnover": turnover,
                "n_stocks": 0 if current is None else len(current),
            })
            previous = current
    return pd.DataFrame(rows, columns=["date", "factor", "target_turnover", "n_stocks"])
