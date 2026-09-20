"""因子横截面市值与行业中性化。"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Optional

import numpy as np
import pandas as pd

from .construction import (
    CORE_FACTOR_COLUMNS,
    OPTIONAL_FACTOR_COLUMNS,
    build_zscore_factor_panel,
)
from .preprocessing import zscore_cross_section


def neutralize_cross_section(
    factor_values: pd.Series,
    *,
    size_values: Optional[pd.Series] = None,
    industry_values: Optional[pd.Series] = None,
) -> pd.Series:
    # 在单个日期横截面内对市值和/或行业做 OLS 回归，并返回回归残差。
    """对一个已完成缺失值筛选的因子横截面执行中性化。"""

    design_parts = [np.ones((len(factor_values), 1), dtype=float)]
    if size_values is not None:
        design_parts.append(
            pd.to_numeric(size_values, errors="coerce")
            .to_numpy(dtype=float)
            .reshape(-1, 1)
        )
    if industry_values is not None:
        industry_dummies = pd.get_dummies(
            industry_values,
            prefix="industry",
            drop_first=True,
            dtype=float,
        )
        if not industry_dummies.empty:
            design_parts.append(industry_dummies.to_numpy(dtype=float))

    design = np.column_stack(design_parts)
    values = pd.to_numeric(factor_values, errors="coerce").to_numpy(dtype=float)
    if len(values) <= np.linalg.matrix_rank(design):
        return pd.Series(np.nan, index=factor_values.index, dtype=float)
    coefficients = np.linalg.lstsq(design, values, rcond=None)[0]
    residuals = values - design @ coefficients
    residual_scale = np.std(residuals, ddof=0)
    numerical_tolerance = np.finfo(float).eps * max(1.0, np.std(values, ddof=0)) * 100
    if residual_scale <= numerical_tolerance:
        residuals[:] = np.nan
    return pd.Series(residuals, index=factor_values.index, dtype=float)


def _default_factor_columns(raw_factor_panel: pd.DataFrame) -> tuple[str, ...]:
    # 选择默认可中性化因子，并排除作为控制变量使用的 size 因子。
    candidates = (*CORE_FACTOR_COLUMNS, *OPTIONAL_FACTOR_COLUMNS)
    return tuple(
        column
        for column in candidates
        if column != "size" and column in raw_factor_panel.columns
    )


def build_neutralized_factor_panel(
    raw_factor_panel: pd.DataFrame,
    *,
    factor_columns: Optional[Iterable[str]] = None,
    neutralize_size: bool = True,
    neutralize_industry: bool = True,
    date_column: str = "date",
    size_column: str = "size",
    industry_column: str = "industry",
    min_observations: int = 30,
) -> pd.DataFrame:
    # 复用基准因子预处理，逐月取得中性化残差，并对残差再次做横截面 Z-score。
    """构造市值、行业或市值加行业中性化的标准化因子面板。

    输入默认为已经清洗且包含 ``is_eligible`` 的原始因子面板。函数先调用
    :func:`build_zscore_factor_panel` 完成合格股票池内的去极值、方向统一和
    基准标准化，再在每个日期横截面内对指定控制变量做含截距 OLS。由于月度
    Z-score 是正的仿射变换，这与在去极值和方向统一后先回归、再对残差做
    Z-score 得到相同的最终标准化残差。
    """

    if not neutralize_size and not neutralize_industry:
        raise ValueError("至少需要启用一种中性化控制变量")

    factors = (
        _default_factor_columns(raw_factor_panel)
        if factor_columns is None
        else tuple(factor_columns)
    )
    baseline = build_zscore_factor_panel(
        raw_factor_panel,
        factor_columns=factors,
    )
    neutralized = baseline.copy()
    neutralized.loc[:, list(factors)] = np.nan

    for _, cross_section in baseline.groupby(date_column, sort=True, observed=True):
        raw_cross_section = raw_factor_panel.loc[cross_section.index]
        for factor in factors:
            valid = cross_section[factor].notna()
            if neutralize_size:
                valid &= pd.to_numeric(
                    raw_cross_section[size_column], errors="coerce"
                ).notna()
            if neutralize_industry:
                valid &= raw_cross_section[industry_column].notna()
            if valid.sum() < min_observations:
                continue

            valid_index = cross_section.index[valid]
            residuals = neutralize_cross_section(
                cross_section.loc[valid_index, factor],
                size_values=(
                    raw_cross_section.loc[valid_index, size_column]
                    if neutralize_size
                    else None
                ),
                industry_values=(
                    raw_cross_section.loc[valid_index, industry_column]
                    if neutralize_industry
                    else None
                ),
            )
            neutralized.loc[valid_index, factor] = residuals

    for factor in factors:
        neutralized[factor] = zscore_cross_section(
            neutralized[factor], neutralized[date_column]
        )
    return neutralized


__all__ = [
    "build_neutralized_factor_panel",
    "neutralize_cross_section",
]
