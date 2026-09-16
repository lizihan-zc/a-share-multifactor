"""毛利资产比（GP）质量因子。

定义：``GP = 毛利润 TTM / 总资产``。
作用：衡量公司每单位资产创造毛利润的能力。GP 越高通常表示主营业务盈利能力
和资产使用效率越强，可作为公司质量与基本面稳定性的信号；该口径通常不适用于
银行和非银金融公司。
"""

from __future__ import annotations

import pandas as pd

from ._utils import require_columns, safe_ratio


FACTOR_NAME = "gross_profitability"


def calculate_gross_profitability(
    panel: pd.DataFrame,
    *,
    gross_profit_column: str = "gross_profit_ttm",
    assets_column: str = "total_assets",
) -> pd.Series:
    """逐行计算 ``毛利润 TTM / 总资产``。"""

    require_columns(
        panel, {gross_profit_column, assets_column}, dataset_name="panel"
    )
    return safe_ratio(
        panel[gross_profit_column], panel[assets_column], name=FACTOR_NAME
    )


calculate_gp = calculate_gross_profitability
