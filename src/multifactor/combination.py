"""Notebook 07：等权合成与只使用已实现收益的滚动 IC / ICIR 加权。

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

    return _calculate_rolling_weights(
        monthly_ic, ic_available_dates, signal_dates,
        window=window, min_history=min_history, method="ic",
    )


def calculate_rolling_icir_weights(
    monthly_ic: pd.DataFrame,
    ic_available_dates: pd.Series,
    signal_dates: pd.DatetimeIndex,
    *,
    window: int = 36,
    min_history: Optional[int] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """用历史 ICIR 分配因子权重；时间窗口与滚动 IC 加权完全一致。

    每个因子的 ICIR = 历史 IC 均值 / IC 样本标准差（ddof=1），不年化；
    再除以所有因子 ICIR 绝对值之和。允许负权重，含义与 IC 加权相同。
    更稳定的 IC 序列会在相同平均 IC 下获得更大的绝对权重。

    标准差至少需要两个有效月份。只要一个因子的标准差为零或无法估计，
    当月整组权重就留空，不删除该因子，也不回退为等权或 IC 加权。
    对很小但仍为正的标准差不另设截断参数，避免引入事后调参；其可能
    导致集中配置，应结合权重图观察。审计表额外保留最小 IC 标准差和
    所有因子标准差是否可用，区分预热不足与 ICIR 分母无法识别。
    """

    return _calculate_rolling_weights(
        monthly_ic, ic_available_dates, signal_dates,
        window=window, min_history=min_history, method="icir",
    )


def _calculate_rolling_weights(
    monthly_ic: pd.DataFrame,
    ic_available_dates: pd.Series,
    signal_dates: pd.DatetimeIndex,
    *,
    window: int,
    min_history: Optional[int],
    method: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """共用收益可得日筛选与窗口审计，只在因子权重的估计量上区分方法。"""

    min_history = window if min_history is None else min_history
    minimum = 2 if method == "icir" else 1
    if not minimum <= min_history <= window:
        raise ValueError(f"必须满足 {minimum} <= min_history <= window")

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
        raw_weights = history.mean()
        if method == "icir":
            # 标准差只来自当前历史窗口；不能用全样本波动率缩放过去的权重。
            ic_std = history.std(ddof=1)
            # 完全相同的浮点数（如三个 0.2）也可能算出极小的非零标准差。
            # 直接识别常数序列，防止把舍入误差误认成极高的稳定性。
            constant = valid_counts.ge(2) & history.nunique().eq(1)
            ic_std = ic_std.mask(constant, 0.0)
            raw_weights = raw_weights.div(ic_std.where(ic_std.gt(0)))
        scale = raw_weights.abs().sum()
        ready = bool(
            valid_counts.ge(min_history).all()
            and np.isfinite(raw_weights).all()
            and scale > 0
        )
        if ready:
            weights.loc[date] = raw_weights / scale

        # 保留实际使用的起止月份与最近收益实现日，便于直接审核时间对齐。
        row = {
            "date": date,
            "history_start": history.index.min(),
            "history_end": history.index.max(),
            "latest_available_date": available_dates.loc[history.index].max(),
            "n_periods": len(history),
            "min_valid_ic_months": int(valid_counts.min()),
            "is_ready": ready,
        }
        if method == "icir":
            row["min_ic_std"] = ic_std.min()
            row["has_valid_ic_std"] = bool(ic_std.notna().all() and ic_std.gt(0).all())
        audit_rows.append(row)

    audit = pd.DataFrame(audit_rows).set_index("date")
    return weights, audit
