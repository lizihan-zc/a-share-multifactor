"""根据清洗后的月度股票池构造未来一个月收益标签和月度研究面板。

推荐调用顺序：

1. 先在 ``src.data.clean`` 中调用 ``clean_universe_monthly``，得到带有
   ``adjusted_close``、``has_valid_label_price`` 和 ``is_clean_for_label`` 的
   清洗后月度股票池。
2. 调用 ``validate_cleaned_universe`` 检查输入字段、主键和清洗标记。
3. 调用 ``build_next_rebalance_date_map`` 建立相邻正式调仓日映射。
4. 调用 ``build_label_endpoint_table`` 准备下一调仓日价格，再连接清洗阶段得到的
   月末停牌记录和停牌分区覆盖表。
5. 调用 ``calculate_future_return_1m`` 计算复权 close-to-close 简单收益，并通过
   ``classify_label_status`` 将终点问题互斥地标记为停牌、退市、行情缺失或原因
   不明的终点缺失。
6. 调用 ``build_monthly_panel`` 整理字段顺序，得到最终月度研究面板。
7. 可选调用 ``build_label_quality_report`` 生成标签覆盖率报告，再调用
   ``save_monthly_panel`` 原子写入 Parquet。

主标签定义为：

``future_return_1m = adjusted_close_t1 / adjusted_close_t - 1``

其中 ``t+1`` 必须是全市场月末调仓日序列中紧接 ``t`` 的下一日期，不能使用某只
股票自己的下一条可用记录，也不能回填缺失的终点价格。研究样本只依据 ``t`` 时点
的 ``is_clean_for_label`` 筛选，不要求股票在 ``t+1`` 仍满足 ``is_eligible``。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd

from src.data.clean import assert_unique_panel_keys, validate_required_columns


CLEANED_UNIVERSE_REQUIRED_COLUMNS = frozenset(
    {
        "date",
        "stock_code",
        "close",
        "adj_factor",
        "adjusted_close",
        "has_valid_label_price",
        "is_clean_for_label",
        "is_eligible",
        "is_suspended",
        "is_one_price_limit_down",
        "list_date",
        "delist_date",
    }
)

LABEL_COLUMNS = (
    "label_start_date",
    "label_end_date",
    "adjusted_close_t",
    "adjusted_close_t1",
    "future_return_1m",
    "label_available",
    "label_status",
    "exit_blocked",
    "is_in_label_sample",
)

ENDPOINT_COLUMNS = (
    "has_exit_record",
    "close_t1",
    "adj_factor_t1",
    "has_valid_label_price_t1",
    "is_suspended_t1_from_universe",
    "is_one_price_limit_down_t1",
)

VALID_LABEL_STATUS = "valid"


def _fill_boolean_missing_with_false(values: pd.Series) -> pd.Series:
    """把连接产生的可空布尔字段显式转型，并将缺失状态解释为 False。"""

    return values.astype("boolean").fillna(False).astype(bool)


def validate_cleaned_universe(cleaned_universe_monthly: pd.DataFrame) -> None:
    """确认输入确实是经过 clean.py 处理且主键唯一的月度股票池。"""

    validate_required_columns(
        cleaned_universe_monthly,
        CLEANED_UNIVERSE_REQUIRED_COLUMNS,
        dataset_name="cleaned_universe_monthly",
    )
    assert_unique_panel_keys(
        cleaned_universe_monthly, dataset_name="cleaned_universe_monthly"
    )
    if cleaned_universe_monthly.empty:
        raise ValueError("cleaned_universe_monthly 不能为空")

    invalid_date = pd.to_datetime(
        cleaned_universe_monthly["date"], errors="coerce"
    ).isna()
    if invalid_date.any():
        raise ValueError(
            "cleaned_universe_monthly 仍包含无法解析的 date："
            f"{int(invalid_date.sum())} 条"
        )

    # 防止重复调用标签函数后产生带后缀的同名字段。
    existing_label_columns = sorted(
        set(LABEL_COLUMNS).intersection(cleaned_universe_monthly.columns)
    )
    if existing_label_columns:
        raise ValueError(
            "输入表已经包含标签字段，请使用未构造标签的清洗后股票池："
            f"{existing_label_columns}"
        )


def validate_month_end_suspension_data(
    month_end_suspensions: pd.DataFrame,
    suspension_date_coverage: pd.DataFrame,
) -> None:
    """检查月末停牌记录和分区覆盖表的字段、日期及唯一性。"""

    validate_required_columns(
        month_end_suspensions,
        {"date", "stock_code", "is_suspended"},
        dataset_name="month_end_suspensions",
    )
    validate_required_columns(
        suspension_date_coverage,
        {"date", "suspension_data_available"},
        dataset_name="suspension_date_coverage",
    )
    assert_unique_panel_keys(
        month_end_suspensions, dataset_name="month_end_suspensions"
    )
    duplicated_coverage = suspension_date_coverage["date"].duplicated(keep=False)
    if duplicated_coverage.any():
        raise ValueError("suspension_date_coverage 存在重复日期")


def build_next_rebalance_date_map(
    cleaned_universe_monthly: pd.DataFrame,
) -> Dict[pd.Timestamp, pd.Timestamp]:
    """根据月度面板的 date 唯一值建立当前调仓日到下一调仓日的映射。"""

    validate_required_columns(
        cleaned_universe_monthly,
        {"date"},
        dataset_name="cleaned_universe_monthly",
    )
    dates = pd.DatetimeIndex(
        pd.to_datetime(cleaned_universe_monthly["date"], errors="coerce")
        .dropna()
        .unique()
    ).sort_values()
    if dates.empty:
        raise ValueError("cleaned_universe_monthly 中没有有效调仓日")

    # 每个自然月只能对应一个正式调仓日，否则“未来一个月”没有唯一含义。
    periods = pd.Series(dates.to_period("M"))
    if periods.duplicated().any():
        repeated = periods.loc[periods.duplicated(keep=False)].astype(str).unique()
        raise ValueError(f"同一自然月存在多个调仓日：{repeated[:5].tolist()}")

    return {
        pd.Timestamp(dates[index]): pd.Timestamp(dates[index + 1])
        for index in range(len(dates) - 1)
    }


def build_label_endpoint_table(
    cleaned_universe_monthly: pd.DataFrame,
) -> pd.DataFrame:
    """提取并重命名下一调仓日的价格、停牌和一字跌停审计字段。"""

    validate_required_columns(
        cleaned_universe_monthly,
        {
            "date",
            "stock_code",
            "close",
            "adj_factor",
            "adjusted_close",
            "has_valid_label_price",
            "is_suspended",
            "is_one_price_limit_down",
        },
        dataset_name="cleaned_universe_monthly",
    )
    endpoints = cleaned_universe_monthly[
        [
            "date",
            "stock_code",
            "close",
            "adj_factor",
            "adjusted_close",
            "has_valid_label_price",
            "is_suspended",
            "is_one_price_limit_down",
        ]
    ].copy()
    endpoints["has_exit_record"] = True
    return endpoints.rename(
        columns={
            "date": "label_end_date",
            "close": "close_t1",
            "adj_factor": "adj_factor_t1",
            "adjusted_close": "adjusted_close_t1",
            "has_valid_label_price": "has_valid_label_price_t1",
            "is_suspended": "is_suspended_t1_from_universe",
            "is_one_price_limit_down": "is_one_price_limit_down_t1",
        }
    )


def attach_exit_suspension_status(
    panel: pd.DataFrame,
    month_end_suspensions: pd.DataFrame,
    suspension_date_coverage: pd.DataFrame,
) -> pd.DataFrame:
    """把原始月末停牌记录和分区覆盖状态连接到标签终点。"""

    validate_month_end_suspension_data(
        month_end_suspensions, suspension_date_coverage
    )
    enriched = panel.copy()
    suspension_endpoints = month_end_suspensions[
        ["date", "stock_code", "is_suspended"]
    ].rename(
        columns={
            "date": "label_end_date",
            "is_suspended": "has_suspension_record_t1",
        }
    )
    coverage_endpoints = suspension_date_coverage[
        ["date", "suspension_data_available"]
    ].rename(
        columns={
            "date": "label_end_date",
            "suspension_data_available": "suspension_data_available_t1",
        }
    )
    enriched = enriched.merge(
        suspension_endpoints,
        on=["label_end_date", "stock_code"],
        how="left",
        validate="many_to_one",
    )
    enriched = enriched.merge(
        coverage_endpoints,
        on="label_end_date",
        how="left",
        validate="many_to_one",
    )
    enriched["has_suspension_record_t1"] = _fill_boolean_missing_with_false(
        enriched["has_suspension_record_t1"]
    )
    enriched["suspension_data_available_t1"] = _fill_boolean_missing_with_false(
        enriched["suspension_data_available_t1"]
    )
    enriched["is_suspended_t1"] = (
        _fill_boolean_missing_with_false(
            enriched["is_suspended_t1_from_universe"]
        )
        | enriched["has_suspension_record_t1"]
    )
    return enriched


def classify_label_status(panel: pd.DataFrame) -> pd.Series:
    """按固定优先级为每条记录分配互斥的中文可审计标签状态码。"""

    validate_required_columns(
        panel,
        {
            "label_start_date",
            "label_end_date",
            "list_date",
            "delist_date",
            "has_exit_record",
            "has_valid_label_price",
            "has_valid_label_price_t1",
            "is_suspended_t1",
            "suspension_data_available_t1",
            "_raw_future_return_1m",
        },
        dataset_name="标签中间面板",
    )
    status = pd.Series(
        VALID_LABEL_STATUS,
        index=panel.index,
        dtype="string",
        name="label_status",
    )

    has_exit_record = _fill_boolean_missing_with_false(panel["has_exit_record"])
    right_censored = panel["label_end_date"].isna()
    missing_exit_record = ~right_censored & ~has_exit_record
    delisted_in_horizon = (
        missing_exit_record
        & panel["delist_date"].notna()
        & panel["delist_date"].gt(panel["label_start_date"])
        & panel["delist_date"].le(panel["label_end_date"])
    )
    suspended_at_exit = (
        ~right_censored
        & ~delisted_in_horizon
        & _fill_boolean_missing_with_false(panel["is_suspended_t1"])
    )
    expected_listed_at_exit = (
        panel["list_date"].notna()
        & panel["list_date"].le(panel["label_end_date"])
        & (
            panel["delist_date"].isna()
            | panel["delist_date"].gt(panel["label_end_date"])
        )
    )
    market_data_missing = (
        missing_exit_record
        & ~delisted_in_horizon
        & ~suspended_at_exit
        & _fill_boolean_missing_with_false(
            panel["suspension_data_available_t1"]
        )
        & expected_listed_at_exit
    )
    other_missing_exit_record = (
        missing_exit_record
        & ~delisted_in_horizon
        & ~suspended_at_exit
        & ~market_data_missing
    )
    missing_start_price = ~_fill_boolean_missing_with_false(
        panel["has_valid_label_price"]
    )
    missing_exit_price = (
        has_exit_record
        & ~_fill_boolean_missing_with_false(panel["has_valid_label_price_t1"])
    )
    raw_return = pd.to_numeric(panel["_raw_future_return_1m"], errors="coerce")
    invalid_return = (
        raw_return.notna()
        & (~np.isfinite(raw_return) | raw_return.lt(-1.0))
    )

    # 从低到高覆盖，使最终状态符合模块文档约定的首要原因优先级。
    status.loc[invalid_return] = "invalid_return"
    status.loc[missing_start_price] = "missing_start_price"
    status.loc[missing_exit_price] = "missing_exit_price"
    status.loc[other_missing_exit_record] = "other_missing_exit_record"
    status.loc[market_data_missing] = "market_data_missing"
    status.loc[suspended_at_exit] = "suspended_at_exit"
    status.loc[delisted_in_horizon] = "delisted_no_exit_price"
    status.loc[right_censored] = "right_censored"
    return status


def calculate_future_return_1m(
    cleaned_universe_monthly: pd.DataFrame,
    month_end_suspensions: pd.DataFrame,
    suspension_date_coverage: pd.DataFrame,
) -> pd.DataFrame:
    """连接月末价格和停牌状态，计算 future_return_1m 及审计字段。"""

    validate_cleaned_universe(cleaned_universe_monthly)
    validate_month_end_suspension_data(
        month_end_suspensions, suspension_date_coverage
    )
    panel = cleaned_universe_monthly.copy()
    panel["date"] = pd.to_datetime(panel["date"])
    panel["list_date"] = pd.to_datetime(panel["list_date"], errors="coerce")
    panel["delist_date"] = pd.to_datetime(panel["delist_date"], errors="coerce")

    next_rebalance_date = build_next_rebalance_date_map(panel)
    panel["label_start_date"] = panel["date"]
    panel["label_end_date"] = panel["date"].map(next_rebalance_date)
    panel["adjusted_close_t"] = panel["adjusted_close"]

    endpoints = build_label_endpoint_table(panel)
    panel = panel.merge(
        endpoints,
        on=["label_end_date", "stock_code"],
        how="left",
        validate="one_to_one",
    )
    panel["has_exit_record"] = _fill_boolean_missing_with_false(
        panel["has_exit_record"]
    )
    panel = attach_exit_suspension_status(
        panel,
        month_end_suspensions,
        suspension_date_coverage,
    )

    # 先保留原始计算值供状态判断，只有状态为 valid 时才进入主标签。
    panel["_raw_future_return_1m"] = (
        panel["adjusted_close_t1"] / panel["adjusted_close_t"] - 1.0
    )
    panel["label_status"] = classify_label_status(panel)
    panel["label_available"] = panel["label_status"].eq(VALID_LABEL_STATUS)
    panel["future_return_1m"] = panel["_raw_future_return_1m"].where(
        panel["label_available"]
    )

    # 一字跌停不篡改市场收益标签，只记录实际退出可能受阻。
    panel["exit_blocked"] = _fill_boolean_missing_with_false(
        panel["is_one_price_limit_down_t1"].eq(True)
    )
    panel["is_in_label_sample"] = (
        _fill_boolean_missing_with_false(panel["is_clean_for_label"])
        & panel["label_available"]
    )
    return panel.drop(columns=["_raw_future_return_1m"])


def build_monthly_panel(
    cleaned_universe_monthly: pd.DataFrame,
    month_end_suspensions: pd.DataFrame,
    suspension_date_coverage: pd.DataFrame,
) -> pd.DataFrame:
    """合并月末停牌信息，保留清洗字段并构造最终 monthly_panel。"""

    panel = calculate_future_return_1m(
        cleaned_universe_monthly,
        month_end_suspensions,
        suspension_date_coverage,
    )
    assert_unique_panel_keys(panel, dataset_name="monthly_panel")

    leading_columns = [
        "date",
        "stock_code",
        *LABEL_COLUMNS,
    ]
    trailing_columns = [
        column
        for column in panel.columns
        if column not in leading_columns
    ]
    return (
        panel.loc[:, [*leading_columns, *trailing_columns]]
        .sort_values(["date", "stock_code"])
        .reset_index(drop=True)
    )


def build_label_quality_report(monthly_panel: pd.DataFrame) -> pd.DataFrame:
    """按信号月和标签状态汇总数量、占比及主研究样本数量。"""

    validate_required_columns(
        monthly_panel,
        {"date", "label_status", "label_available", "is_in_label_sample"},
        dataset_name="monthly_panel",
    )
    report = (
        monthly_panel.groupby(["date", "label_status"], dropna=False)
        .agg(
            count=("stock_code", "size"),
            label_available_count=("label_available", "sum"),
            label_sample_count=("is_in_label_sample", "sum"),
        )
        .reset_index()
    )
    totals = report.groupby("date")["count"].transform("sum")
    report["rate"] = report["count"].div(totals.where(totals.ne(0)))
    return report.sort_values(["date", "label_status"]).reset_index(drop=True)


def save_monthly_panel(monthly_panel: pd.DataFrame, output_path: Path) -> Path:
    """以原子替换方式保存 monthly_panel，避免中断产生残缺 Parquet 文件。"""

    validate_required_columns(
        monthly_panel,
        {"date", "stock_code", "future_return_1m", "label_status"},
        dataset_name="monthly_panel",
    )
    assert_unique_panel_keys(monthly_panel, dataset_name="monthly_panel")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    monthly_panel.to_parquet(temporary_path, index=False)
    temporary_path.replace(path)
    return path


def label_status_counts(monthly_panel: pd.DataFrame) -> pd.DataFrame:
    """汇总全样本各标签状态的记录数和比例，便于快速检查构造结果。"""

    validate_required_columns(
        monthly_panel,
        {"label_status"},
        dataset_name="monthly_panel",
    )
    counts = (
        monthly_panel["label_status"]
        .value_counts(dropna=False)
        .rename_axis("label_status")
        .rename("count")
        .reset_index()
    )
    counts["rate"] = counts["count"].div(len(monthly_panel))
    return counts


__all__: Iterable[str] = [
    "attach_exit_suspension_status",
    "build_label_endpoint_table",
    "build_label_quality_report",
    "build_monthly_panel",
    "build_next_rebalance_date_map",
    "calculate_future_return_1m",
    "classify_label_status",
    "label_status_counts",
    "save_monthly_panel",
    "validate_cleaned_universe",
    "validate_month_end_suspension_data",
]
