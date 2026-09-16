"""对数市值（SIZE）控制因子。

定义：``SIZE = ln(总市值)``。
作用：描述公司的市场规模，并压缩市值分布的右偏和数量级差异。本项目主要将
SIZE 用作中性化变量、风险暴露和稳健性检验变量，而不是默认视为主 Alpha 因子。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._utils import numeric, require_columns


FACTOR_NAME = "size"


def calculate_size(
    panel: pd.DataFrame,
    *,
    market_cap_column: str = "market_cap",
) -> pd.Series:
    """对严格为正的总市值取自然对数。"""

    require_columns(panel, {market_cap_column}, dataset_name="panel")
    market_cap = numeric(panel[market_cap_column])
    result = np.log(market_cap.where(market_cap.gt(0)))
    return result.replace([np.inf, -np.inf], np.nan).rename(FACTOR_NAME)
