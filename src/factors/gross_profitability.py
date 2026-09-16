"""毛利资产比（GP）质量因子。

定义：``GP = 毛利润 TTM / 总资产``。
作用：衡量公司每单位资产创造毛利润的能力。GP 越高通常表示主营业务盈利能力
和资产使用效率越强，可作为公司质量与基本面稳定性的信号；该口径通常不适用于
银行和非银金融公司。
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from ._utils import require_columns, safe_ratio


FACTOR_NAME = "gross_profitability"


def calculate_gross_profitability(
    panel: pd.DataFrame,
    *,
    gross_profit_column: str = "gross_profit_ttm",
    assets_column: str = "total_assets",
    applicability_column: Optional[str] = None,
) -> pd.Series:
    """逐行计算 ``毛利润 TTM / 总资产``。

    提供 ``applicability_column`` 时，只为明确标记为适用的记录返回因子值；
    False 或缺失标记对应的结果为缺失。
    """

    required = {gross_profit_column, assets_column}
    if applicability_column is not None:
        required.add(applicability_column)
    require_columns(
        panel, required, dataset_name="panel"
    )
    result = safe_ratio(
        panel[gross_profit_column], panel[assets_column], name=FACTOR_NAME
    )
    if applicability_column is not None:
        try:
            applicable = (
                panel[applicability_column]
                .astype("boolean")
                .fillna(False)
                .astype(bool)
            )
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"{applicability_column} 必须只包含布尔值或缺失值"
            ) from exc
        result = result.where(applicable)
    return result


calculate_gp = calculate_gross_profitability
