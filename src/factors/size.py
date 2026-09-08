"""对数市值（SIZE）控制因子。"""

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
