"""构造并保存 Notebook 03 定义的因子面板。"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Optional, Union

import pandas as pd

from ._utils import (
    PANEL_KEYS,
    add_daily_returns,
    prepare_daily_factor_data,
    require_columns,
)
from .amihud_illiquidity import calculate_amihud_illiquidity_from_prepared
from .book_to_price import calculate_book_to_price
from .earnings_to_price import calculate_earnings_to_price
from .gross_profitability import calculate_gross_profitability
from .low_volatility import calculate_low_volatility_from_prepared
from .momentum_12_1 import calculate_momentum_12_1_from_prepared
from .momentum_3m import calculate_momentum_3m_from_prepared
from .preprocessing import preprocess_factor_panel
from .roe import calculate_roe
from .size import calculate_size


CORE_FACTOR_COLUMNS = (
    "ep",
    "bp",
    "roe",
    "gross_profitability",
    "momentum_12_1",
    "momentum_3m",
    "low_volatility_60d",
)
OPTIONAL_FACTOR_COLUMNS = ("amihud_illiquidity_20d", "size")
FACTOR_PANEL_CONTEXT_COLUMNS = (
    "date",
    "stock_code",
    "future_return_1m",
    "label_available",
    "label_status",
    "is_in_label_sample",
    "is_eligible",
    "is_buyable",
    "is_suspended",
    "is_gross_profit_applicable",
    "has_gp_factor_data",
    "industry",
    "market_cap",
)
DEFAULT_DIRECTIONS = {
    # LOWVOL 已在原始计算中取负号；非流动性是默认因子中唯一需要反向的原始指标。
    "amihud_illiquidity_20d": -1,
}


def build_raw_factor_panel(
    monthly_panel: pd.DataFrame,
    cleaned_price_daily: pd.DataFrame,
    *,
    include_optional: bool = True,
    context_columns: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """构造只包含研究上下文字段和因子值的精简月度因子面板。

    ``context_columns`` 默认保留主键、未来收益标签、股票池状态以及行业和市值
    控制变量。行情、财务原始值和中间清洗字段仍保存在源面板中，不在因子面板
    重复存储。``cleaned_price_daily`` 应为 Notebook 02 保存的清洗后日频面板。
    """

    if context_columns is None:
        context_columns = FACTOR_PANEL_CONTEXT_COLUMNS
    context_columns = tuple(dict.fromkeys(context_columns))
    require_columns(monthly_panel, context_columns, dataset_name="monthly_panel")
    missing_keys = sorted(set(PANEL_KEYS).difference(context_columns))
    if missing_keys:
        raise ValueError(f"context_columns 必须包含面板主键：{missing_keys}")

    factors = monthly_panel.loc[:, list(context_columns)].copy()
    factors["ep"] = calculate_earnings_to_price(monthly_panel)
    factors["bp"] = calculate_book_to_price(monthly_panel)
    factors["roe"] = calculate_roe(monthly_panel)
    factors["gross_profitability"] = calculate_gross_profitability(
        monthly_panel,
        applicability_column="is_gross_profit_applicable",
    )
    extra_columns = ("amount",) if include_optional else ()
    daily = prepare_daily_factor_data(
        cleaned_price_daily, extra_columns=extra_columns
    )
    daily = add_daily_returns(daily)
    factors["momentum_12_1"] = calculate_momentum_12_1_from_prepared(
        daily, monthly_panel
    )
    factors["momentum_3m"] = calculate_momentum_3m_from_prepared(
        daily, monthly_panel
    )
    factors["low_volatility_60d"] = calculate_low_volatility_from_prepared(
        daily, monthly_panel
    )
    if include_optional:
        factors["amihud_illiquidity_20d"] = (
            calculate_amihud_illiquidity_from_prepared(daily, monthly_panel)
        )
        factors["size"] = calculate_size(monthly_panel)
    return factors


def build_zscore_factor_panel(
    raw_factor_panel: pd.DataFrame,
    *,
    factor_columns: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """在 ``is_eligible`` 股票池内执行去极值、方向统一和 Z-score。

    输出保留完整月度骨架和上下文字段，但非研究样本的标准化因子设为缺失，
    防止 ST、新股、停牌或不可买记录影响研究横截面的变换参数。
    """

    if factor_columns is None:
        factor_columns = tuple(
            column
            for column in (*CORE_FACTOR_COLUMNS, *OPTIONAL_FACTOR_COLUMNS)
            if column in raw_factor_panel.columns
        )
    factor_columns = tuple(factor_columns)
    require_columns(
        raw_factor_panel,
        {"is_eligible", *factor_columns},
        dataset_name="raw_factor_panel",
    )
    directions = {
        column: DEFAULT_DIRECTIONS[column]
        for column in factor_columns
        if column in DEFAULT_DIRECTIONS
    }
    return preprocess_factor_panel(
        raw_factor_panel,
        factor_columns,
        directions=directions,
        sample_mask=raw_factor_panel["is_eligible"],
    )


def save_factor_panels(
    raw_factor_panel: pd.DataFrame,
    zscore_factor_panel: pd.DataFrame,
    *,
    output_directory: Union[Path, str],
) -> tuple[Path, Path]:
    """以原子方式保存 README 指定的两份 Parquet 因子面板。"""

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    raw_path = output_directory / "factor_panel_raw.parquet"
    zscore_path = output_directory / "factor_panel_zscore.parquet"
    for frame, path in (
        (raw_factor_panel, raw_path),
        (zscore_factor_panel, zscore_path),
    ):
        temporary = path.with_name(f".{path.name}.tmp")
        frame.to_parquet(temporary, index=False)
        temporary.replace(path)
    return raw_path, zscore_path
