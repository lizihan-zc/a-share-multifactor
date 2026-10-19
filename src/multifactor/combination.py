"""Notebook 07：等权合成与只使用已实现收益的滚动 IC 加权。

输入约定沿用 Notebook 03/06：日期是正式月末调仓日，因子已经去极值、
统一方向并标准化；每个股票—日期只有一条记录。这里不重复清洗上游数据。
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Optional

import numpy as np
import pandas as pd


def combine_factor_scores(
    factor_panel: pd.DataFrame,
    factor_columns: Iterable[str],
    weights: Optional[pd.DataFrame] = None,
    *,
    score_name: str = "equal_weight_score",
) -> pd.Series:
    """合成股票评分；weights=None 时，每个因子的权重相同。

    weights 的索引是信号日，列名是因子名。带符号权重直接用于合成，
    不在股票层重新归一化。任一输入因子缺失、当月权重缺失或股票不合格，
    评分即为缺失，避免不同股票实际上使用不同的合成公式。

    此函数完全不读取未来收益和标签资格；最后一期没有收益也应生成信号。
    """

    factors = list(factor_columns)
    values = factor_panel[factors]
    if weights is None:
        # min_count 禁止 pandas 默认跳过缺失因子后，对剩余因子求平均。
        score = values.sum(axis=1, min_count=len(factors)) / len(factors)
    else:
        # 先按日期查找权重，再恢复原来的行索引，保持乱序面板也能正确对齐。
        row_weights = weights.reindex(pd.DatetimeIndex(factor_panel["date"]))
        row_weights.index = factor_panel.index
        score = (values * row_weights[factors]).sum(
            axis=1, min_count=len(factors)
        )
    return score.where(factor_panel["is_eligible"].fillna(False)).rename(score_name)


def calculate_rolling_ic_weights(
    monthly_ic: pd.DataFrame,
    ic_available_dates: pd.Series,
    signal_dates: pd.DatetimeIndex,
    *,
    window: int = 36,
    min_history: Optional[int] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """按已知信息计算滚动平均 IC 权重，并返回逐月历史窗口审计表。

    monthly_ic 的索引是因子信号形成日，而非收益实现日；例如 1 月信号的
    IC 要到 2 月月末收益实现后才可使用。ic_available_dates 用相同索引，
    指明每个 IC 最早可用的日期，NaT 表示其收益尚未实现。

    对每个信号日，选取已经可用的最近 window 个正式月份，再逐列计算
    平均 IC。窗口中的缺失 IC 保留，不向更早月份无限追溯补足。默认每个
    因子都要有 window 个有效 IC；min_history 可显式放宽有效月份要求。

    权重为 mean_ic / sum(abs(mean_ic))。允许负权重，绝对值之和为 1；
    历史不足或所有平均 IC 为零时整行留空，不偷偷回退到等权。
    """

    min_history = window if min_history is None else min_history
    if not 1 <= min_history <= window:
        raise ValueError("必须满足 1 <= min_history <= window")

    ic = monthly_ic.sort_index()
    available_dates = pd.to_datetime(ic_available_dates.reindex(ic.index))
    # 这是时间契约检查：错误的可用日期会让整套回测产生未来信息泄漏。
    if (available_dates <= ic.index).any():
        raise ValueError("IC 可用日必须晚于其信号日")

    weights = pd.DataFrame(np.nan, index=signal_dates, columns=ic.columns)
    weights.index.name = "date"
    audit_rows = []
    for date in signal_dates:
        known = available_dates.notna() & available_dates.le(date) & (ic.index < date)
        history = ic.loc[known].tail(window)
        valid_counts = history.count()
        mean_ic = history.mean()
        scale = mean_ic.abs().sum()
        ready = bool(valid_counts.ge(min_history).all() and scale > 0)
        if ready:
            weights.loc[date] = mean_ic / scale

        # 保留实际使用的起止月份与最近收益实现日，便于直接审核时间对齐。
        audit_rows.append({
            "date": date,
            "history_start": history.index.min(),
            "history_end": history.index.max(),
            "latest_available_date": available_dates.loc[history.index].max(),
            "n_periods": len(history),
            "min_valid_ic_months": int(valid_counts.min()),
            "is_ready": ready,
        })

    audit = pd.DataFrame(audit_rows).set_index("date")
    return weights, audit
