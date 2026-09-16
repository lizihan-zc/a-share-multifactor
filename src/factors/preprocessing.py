"""Notebook 03 要求的横截面因子预处理函数。"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Optional

import numpy as np
import pandas as pd

from ._utils import numeric, require_columns


def winsorize_cross_section(
    values: pd.Series,
    dates: pd.Series,
    *,
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
) -> pd.Series:
    """
    在每个日期的横截面内按上下分位数截断因子值，降低少数异常或极端因子值对研究结果的过度影响。
    """

    if not 0 <= lower_quantile < upper_quantile <= 1:
        raise ValueError("分位数必须满足 0 <= lower < upper <= 1")
    values = numeric(values)
    grouped = values.groupby(pd.to_datetime(dates, errors="coerce"), dropna=False)
    lower = grouped.transform("quantile", q=lower_quantile)
    upper = grouped.transform("quantile", q=upper_quantile)
    return values.clip(lower=lower, upper=upper)


def zscore_cross_section(values: pd.Series, dates: pd.Series) -> pd.Series:
    """在每个日期的横截面内使用总体标准差对因子进行标准化。zscore: 标准化分数。"""

    values = numeric(values)
    grouped = values.groupby(pd.to_datetime(dates, errors="coerce"), dropna=False)
    mean = grouped.transform("mean")
    std = grouped.transform("std", ddof=0)
    return values.sub(mean).div(std.where(std.gt(0))).replace(
        [np.inf, -np.inf], np.nan
    )


def preprocess_factor_panel(
    factor_panel: pd.DataFrame,
    factor_columns: Iterable[str],
    *,
    date_column: str = "date",
    directions: Optional[Mapping[str, int]] = None,
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
    sample_mask: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """在每个日期独立完成去极值、方向统一和 Z-score 标准化。

    因子方向只能为 ``1``（越大越好）或 ``-1``（越小越好）。先去极值，再按需反转
    方向。``sample_mask`` 用于限定估计和生成标准化因子的横截面；掩码外的上下文字段
    继续保留，因子值则设为缺失。未提供掩码时兼容原行为，使用全部记录。
    """

    factor_columns = tuple(factor_columns)
    require_columns(
        factor_panel,
        {date_column, *factor_columns},
        dataset_name="factor_panel",
    )
    directions = {} if directions is None else dict(directions)
    unknown = sorted(set(directions).difference(factor_columns))
    if unknown:
        raise ValueError(f"directions 包含未知因子列：{unknown}")

    if sample_mask is None:
        selected = pd.Series(True, index=factor_panel.index, dtype="bool")
    else:
        if not isinstance(sample_mask, pd.Series):
            raise TypeError("sample_mask 必须是 pandas Series")
        if not sample_mask.index.equals(factor_panel.index):
            raise ValueError("sample_mask 的索引必须与 factor_panel 完全一致")
        try:
            selected = sample_mask.astype("boolean").fillna(False).astype(bool)
        except (TypeError, ValueError) as exc:
            raise TypeError("sample_mask 必须只包含布尔值或缺失值") from exc

    processed = factor_panel.copy()
    reference = factor_panel.loc[selected]
    for column in factor_columns:
        direction = directions.get(column, 1)
        if direction not in (-1, 1):
            raise ValueError(f"{column} 的 direction 必须为 1 或 -1")
        clipped = winsorize_cross_section(
            reference[column],
            reference[date_column],
            lower_quantile=lower_quantile,
            upper_quantile=upper_quantile,
        )
        standardized = zscore_cross_section(
            clipped.mul(direction), reference[date_column]
        )
        processed[column] = np.nan
        processed.loc[selected, column] = standardized.to_numpy()
    return processed
