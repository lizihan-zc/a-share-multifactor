"""清洗 Notebook 02 构造标签前使用的 processed 面板。

思路：先把字段按结构要求、业务用途和清洗规则分组，在开头定义一些大写的模块级的配置常量；
然后按顺序清洗对应的数据。

1. 结构清洗：
   ``validate_required_columns`` → ``normalize_panel_keys`` →
   ``assert_unique_panel_keys``。检查字段、统一 ``date`` 和 ``stock_code``，并保证
   ``(date, stock_code)`` 主键唯一。
2. 数值清洗：
   ``clean_numeric_values`` → ``add_price_quality_flags``。把无法解析或无穷的数值
   设为缺失，把不可能为负/非正的值设为缺失，并生成价格质量标记。
3. 时间对齐：
   ``normalize_datetime_columns`` → ``add_point_in_time_flags`` →
   ``validate_point_in_time_financials`` → ``validate_monthly_rebalance_dates``。
   检查财务数据只在公告后使用，并确认月频记录对应每月最后一个交易日。
4. 特殊状态处理：
   ``add_universe_quality_flags`` → ``load_month_end_suspensions``。保留 ST、停牌、
   新股和涨跌停记录，不静默删除；同时读取月末停牌分区及其日期覆盖，供标签阶段
   区分终点停牌、退市、行情缺失和原因不明的终点缺失。
5. 异常收益诊断：
   ``add_return_diagnostics``。使用复权收盘价计算相邻有效行情记录收益，仅标记
   极端值和跨日间隔，不把真实市场极端收益自动删除。
6. 质量报告：
   ``build_missing_rate_report`` → ``build_special_state_report``。汇总缺失率及特殊
   状态数量，为 Notebook 中的解释和图表提供可审计数据。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd



PANEL_KEY = ("date", "stock_code")

PRICE_DAILY_REQUIRED_COLUMNS = frozenset(
    {
        "stock_code",
        "date",
        "open",
        "high",
        "low",
        "close",
        "prev_close",
        "volume",
        "amount",
        "adj_factor",
        "market_cap",
        "is_one_price_limit_up",
        "is_one_price_limit_down",
    }
)

UNIVERSE_MONTHLY_REQUIRED_COLUMNS = frozenset(
    {
        *PRICE_DAILY_REQUIRED_COLUMNS,
        "list_date",
        "delist_date",
        "is_listed",
        "listing_trading_days",
        "is_st",
        "is_suspended",
        "financial_available_date",
        "is_buyable",
        "passes_listing_age",
        "passes_liquidity",
        "has_core_data",
        "is_eligible",
    }
)

PRICE_COLUMNS = (
    "open",
    "high",
    "low",
    "close",
    "prev_close",
    "up_limit_price",
    "down_limit_price",
)

NON_NEGATIVE_COLUMNS = (
    "volume",
    "amount",
    "listing_trading_days",
)

POSITIVE_COLUMNS = (
    *PRICE_COLUMNS,
    "adj_factor",
    "market_cap",
    "total_assets",
)

FINANCIAL_COLUMNS = (
    "revenue_ttm",
    "net_profit_ttm",
    "gross_profit_ttm",
    "total_assets",
    "total_equity",
)

MONTHLY_DATE_COLUMNS = (
    "list_date",
    "delist_date",
    "end_date",
    "announcement_date",
    "financial_available_date",
    "current_income_announcement_date",
    "prior_same_announcement_date",
    "prior_annual_announcement_date",
    "balance_announcement_date",
)

ELIGIBILITY_COLUMNS = (
    "is_listed",
    "is_st",
    "is_suspended",
    "is_buyable",
    "passes_listing_age",
    "passes_liquidity",
    "has_core_data",
    "is_eligible",
)

SPECIAL_STATE_COLUMNS = (
    "is_st",
    "is_suspended",
    "is_one_price_limit_up",
    "is_one_price_limit_down",
    "is_entry_blocked",
    "has_invalid_label_price",
    "has_future_financial_data",
    "eligibility_rule_inconsistent",
)


@dataclass(frozen=True)
class CleaningResult:
    """保存清洗后的面板、月末停牌数据及质量报告。"""

    price_daily: pd.DataFrame
    universe_monthly: pd.DataFrame
    month_end_suspensions: pd.DataFrame
    suspension_date_coverage: pd.DataFrame
    missing_rate_report: pd.DataFrame
    special_state_report: pd.DataFrame


'''
============ 结构清洗 ============
'''


def validate_required_columns(
    frame: pd.DataFrame,
    required_columns: Iterable[str],
    *,
    dataset_name: str,
) -> None:
    """检查输入表是否包含当前清洗步骤所需的全部字段。"""

    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{dataset_name} 必须是 pandas DataFrame")
    
    missing = sorted(set(required_columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{dataset_name} 缺少必需字段：{missing}")


def normalize_panel_keys(
    frame: pd.DataFrame,
    *,
    dataset_name: str,
) -> pd.DataFrame:
    """统一股票代码和日期主键格式，并拒绝无法识别的空主键。"""

    validate_required_columns(frame, PANEL_KEY, dataset_name=dataset_name)

    cleaned = frame.copy()

    # 转换成 str, 删除空格，英文字母大写，并把缺失值保留为 <NA>
    cleaned["stock_code"] = cleaned["stock_code"].astype("string")

    cleaned["stock_code"] = cleaned["stock_code"].str.strip()

    cleaned["stock_code"] = cleaned["stock_code"].str.upper()

    invalid_code = cleaned["stock_code"].isna() | cleaned["stock_code"].eq("")

    cleaned["date"] = pd.to_datetime(cleaned["date"], errors="coerce")

    invalid_date = cleaned["date"].isna()

    if invalid_code.any() or invalid_date.any():
        raise ValueError(
            f"{dataset_name} 存在无效主键："
            f"股票代码 {int(invalid_code.sum())} 条，日期 {int(invalid_date.sum())} 条"
        )
    
    return cleaned


def assert_unique_panel_keys(
    frame: pd.DataFrame,
    *,
    dataset_name: str,
) -> None:
    """
    确认每个股票和日期组合只对应一条记录，避免连接后重复计算。
    主键重复通常不是普通的脏数据，应该先查明原因，不能简单地去重。
    """

    validate_required_columns(frame, PANEL_KEY, dataset_name=dataset_name)

    duplicated = frame.duplicated(list(PANEL_KEY), keep=False)

    if duplicated.any():
        examples = frame.loc[duplicated, list(PANEL_KEY)].head(5).to_dict("records")
        raise ValueError(
            f"{dataset_name} 存在 {int(duplicated.sum())} 条重复主键记录，"
            f"示例：{examples}"
        )


'''
============ 数值清洗 ============
'''


def clean_numeric_values(
    frame: pd.DataFrame,
    *,
    numeric_columns: Iterable[str],
    positive_columns: Iterable[str] = (),
    non_negative_columns: Iterable[str] = (),
) -> pd.DataFrame:
    """规范数值字段，并把无穷值和符号错误的数值设为缺失。"""

    cleaned = frame.copy()

    existing_numeric_columns = [column for column in numeric_columns if column in cleaned]

    for column in existing_numeric_columns:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
        cleaned[column] = cleaned[column].replace([np.inf, -np.inf], np.nan)

    for column in positive_columns:
        if column in cleaned:
            non_positive_mask = (
                cleaned[column].notna() & (cleaned[column]<=0)
            )
            cleaned.loc[non_positive_mask, column] = np.nan

    for column in non_negative_columns:
        if column in cleaned:
            negative_mask = (
                cleaned[column].notna() & (cleaned[column]<0)
            )
            cleaned.loc[negative_mask, column] = np.nan

    return cleaned


def add_price_quality_flags(frame: pd.DataFrame) -> pd.DataFrame:
    """检查 OHLC、成交数据、复权因子和市值的缺失值和符号，生成合法性检查字段。"""

    validate_required_columns(
        frame,
        {
            "open",
            "high",
            "low",
            "close",
            "prev_close",
            "volume",
            "amount",
            "adj_factor",
            "market_cap",
        },
        dataset_name="价格面板",
    )

    cleaned = frame.copy()

    # 判断每一行是否数据齐全
    # axis=1 每一行计算一次，得到“每行一个结果”
    # axis=0 每一列计算一次，得到“每列一个结果”
    ohlc_complete = cleaned[["open", "high", "low", "close"]].notna().all(axis=1)
    
    valid_high = cleaned["high"] >= cleaned[["open", "low", "close"]].max(axis=1)
    valid_low = cleaned["low"] <= cleaned[["open", "high", "close"]].min(axis=1)
    ohlc_ordered = valid_high & valid_low

    cleaned["has_valid_ohlc"] = ohlc_complete & ohlc_ordered

    valid_close = (
        cleaned["close"].notna() & (cleaned["close"]>0)
    )
    valid_adj_factor = (
        cleaned["adj_factor"].notna() & (cleaned["adj_factor"]>0)
    )
    cleaned["has_valid_label_price"] = (
        valid_close & valid_adj_factor
    )

    cleaned["adjusted_close"] = cleaned["close"] * cleaned["adj_factor"]

    volume_amount_complete = cleaned[["volume", "amount"]].notna().all(axis=1)
    cleaned["has_valid_trading_value"] = (
        volume_amount_complete
        & (cleaned["volume"]>=0)
        & (cleaned["amount"]>=0)
    )

    cleaned["has_valid_market_cap"] = (
        cleaned["market_cap"].notna() & (cleaned["market_cap"]>0)
    )

    return cleaned


'''
============ 时间对齐 ============
'''


def normalize_datetime_columns(
    frame: pd.DataFrame,
    columns: Iterable[str],     # MONTHLY_DATE_COLUMNS, 主要包含月度表中的辅助日期字段
) -> pd.DataFrame:
    """
    把主键以外的其它日期字段统一转为时间戳，非法文本转为缺失。
    """

    cleaned = frame.copy()

    for column in columns:
        if column in cleaned.columns:
            cleaned[column] = pd.to_datetime(cleaned[column], errors="coerce")

    return cleaned


def add_point_in_time_flags(universe_monthly: pd.DataFrame) -> pd.DataFrame:
    """标记财务数据是否在对应调仓日已经公开，防止未来信息泄漏。"""

    validate_required_columns(
        universe_monthly,
        {"date", "financial_available_date"},
        dataset_name="universe_monthly",
    )

    cleaned = universe_monthly.copy()

    # 检查是否存在未来财报
    cleaned["has_future_financial_data"] = (
        cleaned["financial_available_date"] > cleaned["date"]
    )
    
    cleaned["has_point_in_time_financials"] = (
        cleaned["financial_available_date"].notna()
        & ~cleaned["has_future_financial_data"]     # 没有未来财报
    )

    return cleaned


def validate_point_in_time_financials(universe_monthly: pd.DataFrame) -> None:
    """拒绝含未来财务信息或核心财务完整但可用日期缺失的月度面板。"""

    validate_required_columns(
        universe_monthly,
        {
            "financial_available_date",
            "has_future_financial_data",
            "has_core_data",
        },
        dataset_name="universe_monthly",
    )

    future_count = int(universe_monthly["has_future_financial_data"].sum())

    missing_date = (
        universe_monthly["has_core_data"].fillna(False)
        & universe_monthly["financial_available_date"].isna()
    )

    if future_count or missing_date.any():
        raise ValueError(
            "universe_monthly 的财务时点不合法："
            f"未来财务 {future_count} 条，"
            f"核心财务完整但可用日期缺失 {int(missing_date.sum())} 条"
        )


def validate_monthly_rebalance_dates(
    universe_monthly: pd.DataFrame,
    price_daily: pd.DataFrame,
) -> None:
    """
    确认每月只有一个调仓日，且该日等于日频面板中的当月最后交易日。
    月度股票池中每个月有很多股票，每个股票有一个 date，
    确保每个月只有唯一的调仓日，需要按月分组（于是需要将日期转换成月）。
    """

    validate_required_columns(
        universe_monthly, {"date"}, dataset_name="universe_monthly"
    )

    validate_required_columns(
        price_daily, {"date"}, dataset_name="price_daily"
    )

    # .unique() 返回非重复元素的一个没有索引的数组
    # 它与 .duplicated() 的区别是后者返回的是bool值
    monthly_dates = universe_monthly["date"].dropna().unique()

    monthly_dates = pd.DatetimeIndex(monthly_dates).sort_values()

    # 构造一个月度股票池中所有日期的 DataFrame
    monthly_table = pd.DataFrame({"date": monthly_dates})

    # 将日期转换为月份
    monthly_table["period"] = monthly_table["date"].dt.to_period("M")

    # 检查每个月是否有多个调仓日
    # .nunique() 返回不同元素的个数
    repeated_dates = monthly_table.groupby("period")["date"].nunique()

    repeated_months = repeated_dates>1

    # .any() 一旦有 True 则返回 True
    if repeated_months.any():
        periods = repeated_months.index[repeated_months].astype(str).tolist()
        raise ValueError(f"universe_monthly 同一自然月存在多个调仓日：{periods[:5]}")

    daily_dates = price_daily["date"].dropna().unique()

    daily_dates = pd.DataFrame({"date": daily_dates})

    daily_dates["period"] = daily_dates["date"].dt.to_period("M")

    expected_month_end = daily_dates.groupby("period")["date"].max()

    expected_month_end = expected_month_end.to_dict()

    mismatches = [
        date
        for date in monthly_dates
        if expected_month_end.get(date.to_period("M")) != date
    ]

    if mismatches:
        examples = [pd.Timestamp(date).strftime("%Y-%m-%d") for date in mismatches[:5]]
        raise ValueError(f"月度调仓日不是日频面板中的当月最后交易日：{examples}")


'''
============ 特殊状态处理 ============
'''


def add_universe_quality_flags(universe_monthly: pd.DataFrame) -> pd.DataFrame:
    """汇总股票池特殊状态，并生成标签入口和筛选规则一致性标记。"""

    validate_required_columns(
        universe_monthly,
        {
            *ELIGIBILITY_COLUMNS,
            "has_valid_label_price",
            "has_future_financial_data",
            "is_one_price_limit_up",
            "is_one_price_limit_down",
        },
        dataset_name="universe_monthly",
    )

    cleaned = universe_monthly.copy()

    expected_eligible = (
        cleaned["is_listed"].fillna(False)
        & ~cleaned["is_st"].fillna(False)
        & ~cleaned["is_suspended"].fillna(False)
        & cleaned["is_buyable"].fillna(False)
        & cleaned["passes_listing_age"].fillna(False)
        & cleaned["passes_liquidity"].fillna(False)
        & cleaned["has_core_data"].fillna(False)
    )

    cleaned["eligibility_rule_inconsistent"] = (
        cleaned["is_eligible"].fillna(False) != expected_eligible
    )

    cleaned["has_invalid_label_price"] = ~cleaned["has_valid_label_price"]

    cleaned["is_entry_blocked"] = (
        cleaned["is_suspended"].fillna(False)
        | cleaned["is_one_price_limit_up"].fillna(False)
        | ~cleaned["is_buyable"].fillna(False)
    )

    cleaned["has_special_state"] = (
        cleaned["is_st"].fillna(False)
        | cleaned["is_suspended"].fillna(False)
        | cleaned["is_one_price_limit_up"].fillna(False)
        | cleaned["is_one_price_limit_down"].fillna(False)
        | cleaned["has_invalid_label_price"]
        | cleaned["has_future_financial_data"]
        | cleaned["eligibility_rule_inconsistent"]
    )

    cleaned["is_clean_for_label"] = (
        cleaned["is_eligible"].fillna(False)
        & cleaned["has_valid_label_price"]
        & ~cleaned["has_future_financial_data"]
        & ~cleaned["eligibility_rule_inconsistent"]
    )

    return cleaned


def load_month_end_suspensions(
    suspension_directory: Path,
    rebalance_dates: Iterable[pd.Timestamp],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """读取各调仓日的停牌分区，并同时返回逐日数据覆盖状态。"""

    directory = Path(suspension_directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"停牌数据目录不存在：{directory}")

    dates = (
        pd.to_datetime(pd.Series(list(rebalance_dates)), errors="coerce")
        .dropna()
        .unique()
    )
    dates = pd.DatetimeIndex(dates).sort_values()

    if dates.empty:
        raise ValueError("rebalance_dates 中没有有效调仓日")

    suspension_frames = []

    coverage_records = []
    
    for date in dates:
        path = directory / f"trade_date={pd.Timestamp(date):%Y%m%d}.parquet"
        partition_available = path.is_file()
        coverage_records.append(
            {
                "date": pd.Timestamp(date),
                "suspension_data_available": partition_available,
            }
        )
        if not partition_available:
            continue

        partition = pd.read_parquet(path)
        validate_required_columns(
            partition,
            {"ts_code", "trade_date"},
            dataset_name=f"停牌分区 {path.name}",
        )
        if partition.empty:
            continue
        records = partition[["ts_code", "trade_date"]].copy()
        records = records.rename(
            columns={"ts_code": "stock_code", "trade_date": "date"}
        )
        records["stock_code"] = (
            records["stock_code"].astype("string").str.strip().str.upper()
        )
        records["date"] = pd.to_datetime(
            records["date"].astype("string"), format="%Y%m%d", errors="coerce"
        )
        records = records.loc[
            records["stock_code"].notna()
            & records["stock_code"].ne("")
            & records["date"].eq(pd.Timestamp(date))
        ].copy()
        records["is_suspended"] = True
        suspension_frames.append(records)

    if suspension_frames:
        suspensions = pd.concat(suspension_frames, ignore_index=True)
        suspensions = (
            suspensions.drop_duplicates(["date", "stock_code"], keep="last")
            .sort_values(["date", "stock_code"])
            .reset_index(drop=True)
        )
    else:
        suspensions = pd.DataFrame(
            {
                "stock_code": pd.Series(dtype="string"),
                "date": pd.Series(dtype="datetime64[ns]"),
                "is_suspended": pd.Series(dtype="bool"),
            }
        )

    coverage = pd.DataFrame.from_records(coverage_records).sort_values("date")
    coverage = coverage.reset_index(drop=True)
    return suspensions, coverage


'''
============ 异常收益诊断 ============
'''


def add_return_diagnostics(
    price_daily: pd.DataFrame,
    *,
    extreme_return_threshold: float = 0.20,
) -> pd.DataFrame:
    """计算逐股相邻行情收益和日期间隔，仅标记极端或不可能的收益。"""

    if extreme_return_threshold <= 0:
        raise ValueError("extreme_return_threshold 必须为正数")
    validate_required_columns(
        price_daily,
        {"stock_code", "date", "adjusted_close"},
        dataset_name="price_daily",
    )
    cleaned = price_daily.sort_values(list(PANEL_KEY)).reset_index(drop=True).copy()
    grouped = cleaned.groupby("stock_code", sort=False)
    cleaned["previous_observation_date"] = grouped["date"].shift(1)
    previous_adjusted_close = grouped["adjusted_close"].shift(1)
    cleaned["observation_gap_days"] = (
        cleaned["date"] - cleaned["previous_observation_date"]
    ).dt.days
    cleaned["adjacent_adjusted_return"] = (
        cleaned["adjusted_close"] / previous_adjusted_close - 1.0
    )
    finite_return = np.isfinite(cleaned["adjacent_adjusted_return"])
    cleaned["has_impossible_adjacent_return"] = (
        finite_return & cleaned["adjacent_adjusted_return"].lt(-1.0)
    )
    cleaned["is_extreme_adjacent_return"] = (
        finite_return
        & cleaned["adjacent_adjusted_return"].abs().ge(extreme_return_threshold)
    )
    return cleaned


'''
============ 质量报告 ============
'''


def build_missing_rate_report(
    datasets: Mapping[str, pd.DataFrame],
    *,
    selected_fields: Optional[Mapping[str, Sequence[str]]] = None,
) -> pd.DataFrame:
    """按数据集和字段汇总缺失数量与缺失率，供清洗前后审计。"""

    records = []
    for dataset_name, frame in datasets.items():
        fields = (
            list(selected_fields[dataset_name])
            if selected_fields is not None and dataset_name in selected_fields
            else list(frame.columns)
        )
        unknown = sorted(set(fields).difference(frame.columns))
        if unknown:
            raise ValueError(f"{dataset_name} 的缺失率字段不存在：{unknown}")
        row_count = len(frame)
        for field in fields:
            missing_count = int(frame[field].isna().sum())
            records.append(
                {
                    "dataset": dataset_name,
                    "field": field,
                    "row_count": row_count,
                    "missing_count": missing_count,
                    "missing_rate": missing_count / row_count if row_count else np.nan,
                }
            )
    return pd.DataFrame.from_records(records)


def build_special_state_report(universe_monthly: pd.DataFrame) -> pd.DataFrame:
    """汇总月度股票池中各类特殊状态的数量和占比。"""

    fields = [field for field in SPECIAL_STATE_COLUMNS if field in universe_monthly]
    validate_required_columns(
        universe_monthly, fields, dataset_name="universe_monthly"
    )
    row_count = len(universe_monthly)
    records = []
    for field in fields:
        count = int(universe_monthly[field].fillna(False).astype(bool).sum())
        records.append(
            {
                "state": field,
                "row_count": row_count,
                "count": count,
                "rate": count / row_count if row_count else np.nan,
            }
        )
    return pd.DataFrame.from_records(records)


'''
============ 工作流构建 ============
'''


def clean_price_daily(
    price_daily: pd.DataFrame,
    *,
    extreme_return_threshold: float = 0.20,
) -> pd.DataFrame:
    """按结构、数值和异常收益步骤清洗日频价格面板并保留诊断字段。"""

    validate_required_columns(
        price_daily, PRICE_DAILY_REQUIRED_COLUMNS, dataset_name="price_daily"
    )
    cleaned = normalize_panel_keys(price_daily, dataset_name="price_daily")
    assert_unique_panel_keys(cleaned, dataset_name="price_daily")
    numeric_columns = (
        *PRICE_COLUMNS,
        "volume",
        "amount",
        "adj_factor",
        "market_cap",
        "limit_status",
    )
    cleaned = clean_numeric_values(
        cleaned,
        numeric_columns=numeric_columns,
        positive_columns=POSITIVE_COLUMNS,
        non_negative_columns=NON_NEGATIVE_COLUMNS,
    )
    cleaned = add_price_quality_flags(cleaned)
    return add_return_diagnostics(
        cleaned, extreme_return_threshold=extreme_return_threshold
    )


def clean_universe_monthly(universe_monthly: pd.DataFrame) -> pd.DataFrame:
    """按结构、数值、财务时点和特殊状态步骤清洗月度股票池面板。"""

    validate_required_columns(
        universe_monthly,
        UNIVERSE_MONTHLY_REQUIRED_COLUMNS,
        dataset_name="universe_monthly",
    )
    cleaned = normalize_panel_keys(universe_monthly, dataset_name="universe_monthly")
    assert_unique_panel_keys(cleaned, dataset_name="universe_monthly")
    cleaned = normalize_datetime_columns(cleaned, MONTHLY_DATE_COLUMNS)
    numeric_columns = (
        *PRICE_COLUMNS,
        *FINANCIAL_COLUMNS,
        "volume",
        "amount",
        "adj_factor",
        "market_cap",
        "limit_status",
        "listing_trading_days",
        "report_age_days",
        "announcement_age_days",
    )
    cleaned = clean_numeric_values(
        cleaned,
        numeric_columns=numeric_columns,
        positive_columns=POSITIVE_COLUMNS,
        non_negative_columns=NON_NEGATIVE_COLUMNS,
    )
    cleaned = add_price_quality_flags(cleaned)
    cleaned = add_point_in_time_flags(cleaned)
    validate_point_in_time_financials(cleaned)
    return add_universe_quality_flags(cleaned)


def run_cleaning_workflow(
    price_daily: pd.DataFrame,
    universe_monthly: pd.DataFrame,
    *,
    suspension_directory: Path,
    extreme_return_threshold: float = 0.20,
) -> CleaningResult:
    """清洗两个面板并读取月末停牌数据，返回结果但不写入磁盘。"""

    cleaned_daily = clean_price_daily(
        price_daily, extreme_return_threshold=extreme_return_threshold
    )
    cleaned_monthly = clean_universe_monthly(universe_monthly)
    validate_monthly_rebalance_dates(cleaned_monthly, cleaned_daily)
    month_end_suspensions, suspension_date_coverage = load_month_end_suspensions(
        suspension_directory,
        cleaned_monthly["date"].drop_duplicates(),
    )

    missing_report = build_missing_rate_report(
        {
            "price_daily": cleaned_daily,
            "universe_monthly": cleaned_monthly,
        }
    )
    special_report = build_special_state_report(cleaned_monthly)
    return CleaningResult(
        price_daily=cleaned_daily,
        universe_monthly=cleaned_monthly,
        month_end_suspensions=month_end_suspensions,
        suspension_date_coverage=suspension_date_coverage,
        missing_rate_report=missing_report,
        special_state_report=special_report,
    )


__all__ = [
    "CleaningResult",
    "add_point_in_time_flags",
    "add_price_quality_flags",
    "add_return_diagnostics",
    "add_universe_quality_flags",
    "assert_unique_panel_keys",
    "build_missing_rate_report",
    "build_special_state_report",
    "clean_numeric_values",
    "clean_price_daily",
    "clean_universe_monthly",
    "load_month_end_suspensions",
    "normalize_datetime_columns",
    "normalize_panel_keys",
    "run_cleaning_workflow",
    "validate_monthly_rebalance_dates",
    "validate_point_in_time_financials",
    "validate_required_columns",
]
