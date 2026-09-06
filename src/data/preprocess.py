"""预处理 Tushare 原始数据并构建 Notebook 01 的股票池。

本模块只读取 ``data/raw`` 中的原始数据，不调用 Tushare 接口。预处理结果统一写入
``data/processed``。

预处理数据的工作流：

1. 读取原始分区：调用 ``read_date_partitions`` 读取日线、复权因子、每日指标、
   涨跌停、ST 和停牌数据。
2. 构建日频面板：调用 ``build_price_daily`` 合并行情相关接口，并统一成交量、
   成交额和市值单位。
3. 构建财务面板：调用 ``prepare_financial_panel`` 清理报表、计算 TTM 指标并合并
   资产负债表；随后由 ``attach_latest_reports`` 按公告日匹配当时已知财务数据。
4. 匹配历史行业：调用 ``attach_historical_industry``，使用 ``in_date`` 和
   ``out_date`` 确定每个调仓日的申万一级行业。
5. 构建月度股票池：调用 ``build_monthly_universe``，处理上市状态、上市交易日数、
   ST、停牌、涨停可买性、流动性和核心字段完整性。
6. 保存结果：调用 ``save_notebook01_outputs``，将 ``price_daily.parquet`` 和
   ``universe_monthly.parquet`` 写入 ``data/processed``。

如需从原始数据开始执行完整预处理流程，可直接调用 ``run_preprocess_pipeline``；
如只需检查或重跑某一步，则调用对应的单个函数。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIRECTORY = PROJECT_ROOT / "data" / "raw"
DEFAULT_PROCESSED_DIRECTORY = PROJECT_ROOT / "data" / "processed"


@dataclass(frozen=True)
class UniverseConfig:
    """定义 Notebook 01 预处理和股票池筛选参数。"""

    study_start: str
    study_end: str
    min_listing_trading_days: int = 120
    liquidity_quantile: Optional[float] = None

    def __post_init__(self) -> None:
        if pd.Timestamp(self.study_start) > pd.Timestamp(self.study_end):
            raise ValueError("study_start 必须早于或等于 study_end")
        if self.min_listing_trading_days < 1:
            raise ValueError("min_listing_trading_days 必须为正数")
        if self.liquidity_quantile is not None and not 0 < self.liquidity_quantile < 1:
            raise ValueError("liquidity_quantile 必须位于 0 和 1 之间")


def _write_parquet_atomic(frame: pd.DataFrame, path: Path) -> None:
    """原子写入 Parquet，防止中断处理生成的残缺文件被复用。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def _as_timestamp(values: pd.Series) -> pd.Series:
    """将 Tushare 的 YYYYMMDD 日期解析为时间戳，同时保留缺失值。"""

    return pd.to_datetime(values.astype("string"), format="%Y%m%d", errors="coerce")


def read_date_partitions(directory: Path) -> pd.DataFrame:
    """读取一个原始 Tushare 接口的全部 Parquet 日期分区。"""

    paths = sorted(directory.glob("trade_date=*.parquet"))
    if not paths:
        return pd.DataFrame()
    return pd.concat((pd.read_parquet(path) for path in paths), ignore_index=True)


def build_price_daily(raw_directory: Path) -> pd.DataFrame:
    """将日线行情、复权因子、市值和涨跌停价格合并为单一面板。

    在此统一 Tushare 的原始单位：成交量转为股、成交额转为元、市值转为元；
    原始文件保持不变。
    """

    required = ("daily", "adj_factor", "daily_basic", "stk_limit")
    frames = {name: read_date_partitions(raw_directory / name) for name in required}
    missing = [name for name, frame in frames.items() if frame.empty]
    if missing:
        raise FileNotFoundError(f"未找到以下接口的原始分区：{', '.join(missing)}")

    prices = frames["daily"].copy()
    key = ["ts_code", "trade_date"]
    for name in ("adj_factor", "daily_basic", "stk_limit"):
        frame = frames[name].drop_duplicates(key)
        columns = [
            column
            for column in frame.columns
            if column not in prices.columns or column in key
        ]
        prices = prices.merge(frame[columns], on=key, how="left")

    prices = prices.rename(
        columns={
            "ts_code": "stock_code",
            "trade_date": "date",
            "pre_close": "prev_close",
            "vol": "volume",
            "total_mv": "market_cap",
            "up_limit": "up_limit_price",
            "down_limit": "down_limit_price",
        }
    )
    prices["date"] = _as_timestamp(prices["date"])
    prices["volume"] = prices["volume"] * 100.0
    prices["amount"] = prices["amount"] * 1_000.0
    prices["market_cap"] = prices["market_cap"] * 10_000.0

    status = prices.get(
        "limit_status", pd.Series(index=prices.index, dtype="float64")
    )
    prices["is_limit_up_close"] = status.isin([2, 3])
    prices["is_limit_down_close"] = status.isin([5, 6])
    tolerance = 1e-8
    prices["is_one_price_limit_up"] = (
        prices["low"].notna()
        & prices["up_limit_price"].notna()
        & (prices["low"] >= prices["up_limit_price"] - tolerance)
    )
    prices["is_one_price_limit_down"] = (
        prices["high"].notna()
        & prices["down_limit_price"].notna()
        & (prices["high"] <= prices["down_limit_price"] + tolerance)
    )
    return prices.sort_values(["date", "stock_code"]).reset_index(drop=True)


def _deduplicate_reports(frame: pd.DataFrame) -> pd.DataFrame:
    """对每只股票、每个报告期保留最新的合并报表。"""

    reports = frame.copy()
    reports["end_date"] = _as_timestamp(reports["end_date"])
    reports["ann_date"] = _as_timestamp(reports["ann_date"])
    reports["f_ann_date"] = _as_timestamp(reports["f_ann_date"])
    reports["announcement_date"] = reports["f_ann_date"].fillna(
        reports["ann_date"]
    )

    if "report_type" in reports:
        report_type = pd.to_numeric(reports["report_type"], errors="coerce")
        reports = reports.loc[report_type.eq(1)].copy()
    reports = reports.dropna(
        subset=["ts_code", "end_date", "announcement_date"]
    )
    reports = reports.sort_values(
        ["ts_code", "end_date", "announcement_date"]
    )
    return reports.drop_duplicates(["ts_code", "end_date"], keep="last")


def prepare_financial_panel(
    income: pd.DataFrame,
    balancesheet: pd.DataFrame,
) -> pd.DataFrame:
    """计算 TTM 利润表字段，并与相应资产负债表合并。

    利润表使用年初至报告期末累计值。中期报告的计算公式为
    ``TTM = 当期累计 + 上年年报 - 上年同期累计``。毛利润定义为营业收入减营业
    成本；任一组成项缺失时保留为空值。
    """

    income_clean = _deduplicate_reports(income)
    balance_clean = _deduplicate_reports(balancesheet)
    income_clean["year"] = income_clean["end_date"].dt.year
    income_clean["month"] = income_clean["end_date"].dt.month

    lookup = income_clean.set_index(["ts_code", "year", "month"])
    annual = income_clean.loc[income_clean["month"].eq(12)].set_index(
        ["ts_code", "year"]
    )
    values = ["revenue", "oper_cost", "n_income_attr_p"]

    result = income_clean.copy()
    for value in values:
        prior_same = [
            lookup[value].get((row.ts_code, row.year - 1, row.month), np.nan)
            for row in result[["ts_code", "year", "month"]].itertuples(index=False)
        ]
        prior_annual = [
            annual[value].get((row.ts_code, row.year - 1), np.nan)
            for row in result[["ts_code", "year"]].itertuples(index=False)
        ]
        ttm = (
            result[value]
            + pd.Series(prior_annual, index=result.index)
            - pd.Series(prior_same, index=result.index)
        )
        result[f"{value}_ttm"] = result[value].where(
            result["month"].eq(12), ttm
        )

    result = result.rename(columns={"n_income_attr_p_ttm": "net_profit_ttm"})
    result["gross_profit_ttm"] = (
        result["revenue_ttm"] - result["oper_cost_ttm"]
    )
    result = result[
        [
            "ts_code",
            "end_date",
            "announcement_date",
            "revenue_ttm",
            "net_profit_ttm",
            "gross_profit_ttm",
        ]
    ]

    balance_columns = [
        "ts_code",
        "end_date",
        "announcement_date",
        "total_assets",
        "total_hldr_eqy_exc_min_int",
    ]
    balance_clean = balance_clean[balance_columns].rename(
        columns={
            "announcement_date": "balance_announcement_date",
            "total_hldr_eqy_exc_min_int": "total_equity",
        }
    )
    panel = result.merge(
        balance_clean, on=["ts_code", "end_date"], how="left"
    )
    panel["announcement_date"] = panel[
        ["announcement_date", "balance_announcement_date"]
    ].max(axis=1)
    panel = panel.drop(columns="balance_announcement_date")
    return panel.sort_values(["ts_code", "announcement_date", "end_date"])


def attach_latest_reports(
    left: pd.DataFrame,
    reports: pd.DataFrame,
    *,
    date_column: str = "date",
    stock_column: str = "stock_code",
) -> pd.DataFrame:
    """为左表每个日期附加当时已公告的最新财报。

    按股票执行 as-of 合并，保证时点可得性，防止较晚公布的年报泄漏到更早的月度调仓日。
    """

    output: list[pd.DataFrame] = []
    reports = reports.rename(columns={"ts_code": stock_column}).copy()
    reports = reports.dropna(subset=[stock_column, "announcement_date"])

    for stock_code, left_group in left.groupby(stock_column, sort=False):
        right_group = reports.loc[
            reports[stock_column].eq(stock_code)
        ].sort_values("announcement_date")
        left_group = left_group.sort_values(date_column)
        if right_group.empty:
            output.append(left_group)
            continue
        output.append(
            pd.merge_asof(
                left_group,
                right_group.drop(columns=stock_column),
                left_on=date_column,
                right_on="announcement_date",
                direction="backward",
            )
        )

    return pd.concat(output, ignore_index=True) if output else left.copy()


def _monthly_rebalance_dates(calendar: pd.DataFrame) -> pd.DataFrame:
    """返回每个自然月最后一个开市交易日。"""

    dates = (
        _as_timestamp(calendar["cal_date"])
        .dropna()
        .drop_duplicates()
        .sort_values()
    )
    month_end = dates.groupby(dates.dt.to_period("M")).max()
    return pd.DataFrame({"date": month_end.to_numpy()})


def _listing_age_in_trading_days(
    dates: pd.Series,
    listing_dates: pd.Series,
    open_dates: pd.Series,
) -> pd.Series:
    """计算自上市日起的开市天数，并将上市日计入。"""

    open_values = (
        pd.to_datetime(open_dates)
        .sort_values()
        .to_numpy(dtype="datetime64[ns]")
    )
    decision_values = pd.to_datetime(dates).to_numpy(dtype="datetime64[ns]")
    listing_values = pd.to_datetime(listing_dates).to_numpy(dtype="datetime64[ns]")
    decision_position = np.searchsorted(open_values, decision_values, side="right")
    listing_position = np.searchsorted(open_values, listing_values, side="left")
    return pd.Series(
        decision_position - listing_position, index=dates.index
    ).clip(lower=0)


def attach_historical_industry(
    panel: pd.DataFrame,
    membership: pd.DataFrame,
) -> pd.DataFrame:
    """附加日期落在有效区间内的申万一级行业。"""

    if membership.empty:
        panel = panel.copy()
        panel["industry"] = pd.NA
        return panel

    intervals = membership.copy()
    intervals["in_date"] = _as_timestamp(intervals["in_date"])
    intervals["out_date"] = _as_timestamp(intervals["out_date"])
    intervals["out_date"] = intervals["out_date"].fillna(
        pd.Timestamp.max.normalize()
    )
    intervals = intervals.rename(
        columns={"ts_code": "stock_code", "l1_name": "industry"}
    )

    output: list[pd.DataFrame] = []
    for stock_code, group in panel.groupby("stock_code", sort=False):
        candidates = intervals.loc[intervals["stock_code"].eq(stock_code)]
        group = group.copy()
        group["industry"] = pd.NA
        for record in candidates.itertuples(index=False):
            mask = group["date"].between(
                record.in_date, record.out_date, inclusive="both"
            )
            group.loc[mask, "industry"] = record.industry
        output.append(group)
    return pd.concat(output, ignore_index=True) if output else panel.copy()


def build_monthly_universe(
    price_daily: pd.DataFrame,
    calendar: pd.DataFrame,
    stock_basic: pd.DataFrame,
    financial_panel: pd.DataFrame,
    *,
    st_records: Optional[pd.DataFrame] = None,
    suspension_records: Optional[pd.DataFrame] = None,
    industry_membership: Optional[pd.DataFrame] = None,
    config: UniverseConfig,
) -> pd.DataFrame:
    """构建月末时点正确且满足条件的股票池。

    返回表保留各项过滤标记和 ``is_eligible``，从而可审计每只股票在任一调仓日
    被剔除的原因。
    """

    rebalance_dates = _monthly_rebalance_dates(calendar)
    monthly = price_daily.merge(rebalance_dates, on="date", how="inner").copy()
    monthly = monthly.merge(
        stock_basic[["ts_code", "list_date", "delist_date"]].rename(
            columns={"ts_code": "stock_code"}
        ),
        on="stock_code",
        how="left",
    )
    monthly["list_date"] = _as_timestamp(monthly["list_date"])
    monthly["delist_date"] = _as_timestamp(monthly["delist_date"])
    monthly["is_listed"] = monthly["date"].ge(monthly["list_date"]) & (
        monthly["delist_date"].isna()
        | monthly["date"].le(monthly["delist_date"])
    )
    monthly["listing_trading_days"] = _listing_age_in_trading_days(
        monthly["date"],
        monthly["list_date"],
        _as_timestamp(calendar["cal_date"]),
    )

    monthly["is_st"] = False
    if st_records is not None and not st_records.empty:
        st_keys = (
            st_records[["ts_code", "trade_date"]]
            .drop_duplicates()
            .rename(columns={"ts_code": "stock_code", "trade_date": "date_key"})
        )
        st_keys["date"] = _as_timestamp(st_keys.pop("date_key"))
        monthly = monthly.merge(
            st_keys.assign(is_st=True),
            on=["stock_code", "date"],
            how="left",
            suffixes=("", "_from_source"),
        )
        monthly["is_st"] = monthly.pop("is_st_from_source").fillna(
            monthly["is_st"]
        )

    monthly["is_suspended"] = False
    if suspension_records is not None and not suspension_records.empty:
        suspension_keys = (
            suspension_records[["ts_code", "trade_date"]]
            .drop_duplicates()
            .rename(columns={"ts_code": "stock_code", "trade_date": "date_key"})
        )
        suspension_keys["date"] = _as_timestamp(suspension_keys.pop("date_key"))
        monthly = monthly.merge(
            suspension_keys.assign(is_suspended=True),
            on=["stock_code", "date"],
            how="left",
            suffixes=("", "_from_source"),
        )
        monthly["is_suspended"] = monthly.pop(
            "is_suspended_from_source"
        ).fillna(monthly["is_suspended"])

    monthly = attach_latest_reports(monthly, financial_panel)
    if industry_membership is not None:
        monthly = attach_historical_industry(monthly, industry_membership)
    else:
        monthly["industry"] = pd.NA

    core_columns = [
        "market_cap",
        "total_assets",
        "total_equity",
        "net_profit_ttm",
        "revenue_ttm",
    ]
    monthly["has_core_data"] = monthly[core_columns].notna().all(axis=1)
    monthly["is_buyable"] = ~monthly["is_one_price_limit_up"].fillna(False)
    monthly["passes_listing_age"] = monthly["listing_trading_days"].ge(
        config.min_listing_trading_days
    )
    monthly["passes_liquidity"] = True
    if config.liquidity_quantile is not None:
        cutoff = monthly.groupby("date")["amount"].transform(
            lambda values: values.quantile(config.liquidity_quantile)
        )
        monthly["passes_liquidity"] = monthly["amount"].ge(cutoff)

    monthly["is_eligible"] = (
        monthly["is_listed"]
        & ~monthly["is_st"].fillna(False)
        & ~monthly["is_suspended"].fillna(False)
        & monthly["is_buyable"]
        & monthly["passes_listing_age"]
        & monthly["passes_liquidity"]
        & monthly["has_core_data"]
    )
    start, end = pd.Timestamp(config.study_start), pd.Timestamp(config.study_end)
    return (
        monthly.loc[monthly["date"].between(start, end)]
        .sort_values(["date", "stock_code"])
        .reset_index(drop=True)
    )


def save_notebook01_outputs(
    price_daily: pd.DataFrame,
    universe_monthly: pd.DataFrame,
    *,
    processed_directory: Path,
) -> tuple[Path, Path]:
    """将 Notebook 01 的两个处理后 Parquet 文件写入 ``data/processed``。"""

    price_path = processed_directory / "price_daily.parquet"
    universe_path = processed_directory / "universe_monthly.parquet"
    _write_parquet_atomic(price_daily, price_path)
    _write_parquet_atomic(universe_monthly, universe_path)
    return price_path, universe_path


def run_preprocess_pipeline(
    config: UniverseConfig,
    *,
    raw_directory: Path = DEFAULT_RAW_DIRECTORY,
    processed_directory: Path = DEFAULT_PROCESSED_DIRECTORY,
) -> tuple[Path, Path]:
    """读取 ``data/raw`` 并运行完整的 Notebook 01 预处理流程。"""

    calendar = pd.read_parquet(raw_directory / "trade_calendar.parquet")
    stocks = pd.read_parquet(raw_directory / "stock_basic.parquet")
    income = pd.read_parquet(raw_directory / "income.parquet")
    balancesheet = pd.read_parquet(raw_directory / "balancesheet.parquet")
    membership = pd.read_parquet(
        raw_directory / "sw_industry_membership.parquet"
    )

    prices = build_price_daily(raw_directory)
    financials = prepare_financial_panel(income, balancesheet)
    universe = build_monthly_universe(
        prices,
        calendar,
        stocks,
        financials,
        st_records=read_date_partitions(raw_directory / "stock_st"),
        suspension_records=read_date_partitions(raw_directory / "suspend_d"),
        industry_membership=membership,
        config=config,
    )
    return save_notebook01_outputs(
        prices,
        universe,
        processed_directory=processed_directory,
    )
