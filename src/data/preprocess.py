"""预处理 Tushare 原始数据并构建 Notebook 01 的股票池。

本模块只读取 ``data/raw`` 中的原始数据，不调用 Tushare 接口。预处理结果统一写入
``data/processed``。

原始数据的形式：？

预处理数据的工作流：

1. 确定沪深股票范围：调用 ``select_sh_sz_stock_basic`` 从股票基础信息中保留
   上交所和深交所的人民币股票，并生成股票代码白名单。
2. 读取并过滤原始数据：调用 ``read_date_partitions`` 读取日线、复权因子、每日
   指标、涨跌停、ST 和停牌数据；调用 ``filter_stock_records`` 只保留白名单中的
   沪深股票，不改动 ``data/raw`` 原始文件。
3. 构建日频面板：调用 ``build_price_daily`` 合并行情相关接口，并统一成交量、
   成交额和市值单位。
4. 构建月末财务快照：调用 ``prepare_report_versions`` 保留同一报告期的历次公告
   和修订版本；调用 ``build_monthly_financial_snapshots`` 在每个调仓日先选择当时
   已公告的最新版本，再计算 TTM 指标并合并同报告期资产负债表，避免未来修订值
   泄漏到历史。
5. 匹配历史行业：调用 ``attach_historical_industry``，使用 ``in_date`` 和
   ``out_date`` 确定每个调仓日的申万一级行业。
6. 构建月度股票池：调用 ``build_monthly_universe``，处理上市状态、上市交易日数、
   ST、停牌、涨停可买性、流动性和核心字段完整性；同时输出
   ``has_gp_factor_data``，供后续毛利润质量因子按自身数据可得性筛选样本，银行和
   非银金融不会仅因不适用普通毛利润口径而被剔除出全局股票池。
7. 保存结果：调用 ``save_notebook01_outputs``，将 ``price_daily.parquet`` 和
   ``universe_monthly.parquet`` 写入 ``data/processed``。

如需从原始数据开始执行完整预处理流程，可直接调用 ``run_preprocess_pipeline``；
如只需检查或重跑某一步，则调用对应的单个函数。

要点：
- 时间要转换成时间戳timestamp
- 利用 drop_duplicates 去重
- 统一单位
"""

# annotations 使类型注解主要用于静态类型检查和代码说明，而不会在函数定义时急于解析。
# 这样更容易引用后面才定义的类，和编写递归或互相引用的类型。
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


# 工作目录和路径
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIRECTORY = PROJECT_ROOT / "data" / "raw"
DEFAULT_PROCESSED_DIRECTORY = PROJECT_ROOT / "data" / "processed"
# frozenset 是可查询、不可修改的集合，不使用 list 是为了防止运行过程中意外加入 BSE
# exchange: 交易所
SH_SZ_EXCHANGES = frozenset({"SSE", "SZSE"})
GROSS_PROFIT_NOT_APPLICABLE_INDUSTRIES = frozenset({"银行", "非银金融"})


# @dataclass 作用于它紧接着修饰的那个类，在定义时无需声明 __init__
# 它免除了重写一遍属性的麻烦，并且可以 print 以查看类对象
# frozen=True 表示对象创建后不能修改字段
@dataclass(frozen=True)
class UniverseConfig:
    """
    UniverseConfig 定义一次预处理和股票池筛选参数：
    研究起止日期；
    最低上市交易日数；
    流动性过滤分位数。
    """

    study_start: str
    study_end: str
    min_listing_trading_days: int = 120
    # 流动性过滤分位数，用来排除调仓日成交额最低的一部分股票
    # 如果设置了分位数，只有成交额超过分位数的股票会被设置为 monthly["passes_liquidity"] = True
    liquidity_quantile: Optional[float] = None

    # @dataclass 自动完成字段赋值后，会调用这个方法检查参数是否合法
    def __post_init__(self) -> None:
        if pd.Timestamp(self.study_start) > pd.Timestamp(self.study_end):
            raise ValueError("study_start 必须早于或等于 study_end")
        if self.min_listing_trading_days < 1:
            raise ValueError("min_listing_trading_days 必须为正数")
        if self.liquidity_quantile is not None and not 0 < self.liquidity_quantile < 1:
            raise ValueError("liquidity_quantile 必须位于 0 和 1 之间")


def _write_parquet_atomic(frame: pd.DataFrame, path: Path) -> None:
    """
    原子写入 Parquet，防止中断处理生成的残缺文件被复用。
    原理是生成一个完整的临时文件代替正式文件，保证程序只能看到替换前的完整旧文件和替换后的完整新文件，
    不会看到写了一半的新文件。
    """

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


def select_sh_sz_stock_basic(stock_basic: pd.DataFrame) -> pd.DataFrame:
    """从股票基础信息中选出沪深人民币股票。

    股票基础信息是预处理阶段的证券范围权威表。使用交易所和币种而不是仅根据
    代码后缀过滤，避免将非人民币证券误纳入研究范围。
    """

    required_columns = {"ts_code", "exchange"}

    ##### 错误报警
    # A.difference(B) 返回 A - B, 即在 A 中 而不在 B 中的元素的集合
    missing_columns = required_columns.difference(stock_basic.columns)
    if missing_columns:
        raise ValueError(
            f"stock_basic 缺少必需字段：{sorted(missing_columns)}"
        )
    #####

    mask = stock_basic["exchange"].isin(SH_SZ_EXCHANGES)
    if "curr_type" in stock_basic.columns:
        mask &= stock_basic["curr_type"].eq("CNY")
    selected = stock_basic.loc[mask].copy()
    if selected.empty:
        raise ValueError("stock_basic 中没有可用的沪深人民币股票")
    # drop_duplicates 根据 ts_code 排除重复股票，每只股票只保留一行
    return selected.drop_duplicates("ts_code", keep="last").sort_values("ts_code")


def filter_stock_records(
    frame: pd.DataFrame,
    allowed_stock_codes: Iterable[str],
    *,
    code_column: str = "ts_code",
) -> pd.DataFrame:
    """按股票代码白名单过滤一张原始或中间表。

    空表会保持原有字段直接返回；非空表必须包含指定的代码字段。该函数只返回
    过滤后的副本，不会修改 data/raw 中的任何文件。
    """

    if frame.empty:
        return frame.copy()
    if code_column not in frame.columns:
        raise ValueError(f"数据缺少股票代码字段：{code_column}")

    allowed = {str(code) for code in allowed_stock_codes if pd.notna(code)}
    if not allowed:
        raise ValueError("股票代码白名单不能为空")
    return frame.loc[frame[code_column].astype("string").isin(allowed)].copy()


def build_price_daily(
    raw_directory: Path,
    *,
    # * 后面的参数必须使用“参数名=参数值”的关键字形式传入，不能按位置传入。
    allowed_stock_codes: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """将日线行情、复权因子、市值和涨跌停价格合并为单一面板。

    如果传入 allowed_stock_codes，则在合并前对所有原始行情表进行统一过滤。
    在此统一 Tushare 的原始单位：成交量转为股、成交额转为元、市值转为元；
    原始文件保持不变。
    """

    required = ("daily", "adj_factor", "daily_basic", "stk_limit")
    # 收集表格。frames: dict
    frames = {name: read_date_partitions(raw_directory / name) for name in required}

    ##### 错误报警
    missing = [name for name, frame in frames.items() if frame.empty]
    if missing:
        raise FileNotFoundError(f"未找到以下接口的原始分区：{', '.join(missing)}")
    #####

    if allowed_stock_codes is not None:
        allowed = {
            str(code) for code in allowed_stock_codes if pd.notna(code)
        }
        if not allowed:
            raise ValueError("股票代码白名单不能为空")
        frames = {
            name: filter_stock_records(frame, allowed)
            for name, frame in frames.items()
        }
        if frames["daily"].empty:
            raise ValueError("按股票代码白名单过滤后日线行情为空")

    # 以 daily 为基础合并其它表格
    prices = frames["daily"].copy()
    key = ["ts_code", "trade_date"]
    for name in ("adj_factor", "daily_basic", "stk_limit"):
        # 注意去重的字段！每只股票每天保留一行
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
            # 当前交易日的前收盘参考价，它可能已经根据分红、送股、拆股等除权除息事件进行了调整
            # 不一定等于上一交易日的 close
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
    # 识别一字涨停股票，即全天最低成交价 ≥ 涨停价
    # 一字涨停股票的 is_buyable=False
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


def prepare_report_versions(
    frame: pd.DataFrame,
    *,
    value_columns: Iterable[str],
) -> pd.DataFrame:
    """清理一张财报表，同时保留同一报告期的历次公告版本。

    ``announcement_date`` 表示某个版本真正可用于研究的日期。这里只消除同一
    股票、报告期和公告日内的重复记录，不会跨公告日保留全样本最终修订版。
    """

    value_columns = tuple(value_columns)
    required_columns = {"ts_code", "ann_date", "end_date", *value_columns}
    missing_columns = required_columns.difference(frame.columns)
    if missing_columns:
        raise ValueError(f"财报数据缺少必需字段：{sorted(missing_columns)}")

    reports = frame.copy()
    reports["end_date"] = _as_timestamp(reports["end_date"])
    reports["ann_date"] = _as_timestamp(reports["ann_date"])
    if "f_ann_date" in reports:
        reports["f_ann_date"] = _as_timestamp(reports["f_ann_date"])
    else:
        reports["f_ann_date"] = pd.NaT
    reports["announcement_date"] = reports["f_ann_date"].fillna(
        reports["ann_date"]
    )

    if "report_type" in reports:
        report_type = pd.to_numeric(reports["report_type"], errors="coerce")
        reports = reports.loc[report_type.eq(1)].copy()
    reports = reports.dropna(
        subset=["ts_code", "end_date", "announcement_date"]
    )
    reports = reports.loc[
        reports["end_date"].le(reports["announcement_date"])
    ].copy()

    reports["_source_order"] = np.arange(len(reports))
    if "update_flag" in reports:
        reports["_update_priority"] = pd.to_numeric(
            reports["update_flag"], errors="coerce"
        ).fillna(-1)
    else:
        reports["_update_priority"] = -1
    reports = reports.sort_values(
        [
            "ts_code",
            "end_date",
            "announcement_date",
            "_update_priority",
            "_source_order",
        ]
    )
    reports = reports.drop_duplicates(
        ["ts_code", "end_date", "announcement_date"], keep="last"
    )
    return (
        reports.drop(columns=["_source_order", "_update_priority"])
        .sort_values(["ts_code", "end_date", "announcement_date"])
        .reset_index(drop=True)
    )


def select_latest_report_versions_as_of(
    report_versions: pd.DataFrame,
    as_of_date: pd.Timestamp,
) -> pd.DataFrame:
    """选择指定日期当时已经公告的各报告期最新版本。"""

    as_of_date = pd.Timestamp(as_of_date)
    available = report_versions.loc[
        report_versions["announcement_date"].le(as_of_date)
        & report_versions["end_date"].le(as_of_date)
    ].copy()
    return (
        available.sort_values(["ts_code", "end_date", "announcement_date"])
        .drop_duplicates(["ts_code", "end_date"], keep="last")
        .reset_index(drop=True)
    )


def _empty_financial_snapshot() -> pd.DataFrame:
    """返回具有稳定字段结构的空月末财务快照。"""

    columns = [
        "stock_code",
        "date",
        "end_date",
        "announcement_date",
        "financial_available_date",
        "current_income_announcement_date",
        "prior_same_announcement_date",
        "prior_annual_announcement_date",
        "balance_announcement_date",
        "revenue_ttm",
        "net_profit_ttm",
        "gross_profit_ttm",
        "total_assets",
        "total_equity",
    ]
    return pd.DataFrame(columns=columns)


def _calculate_financial_snapshot_as_of(
    income_as_of: pd.DataFrame,
    balance_as_of: pd.DataFrame,
    as_of_date: pd.Timestamp,
) -> pd.DataFrame:
    """使用指定日期当时可见的报表版本计算一张横截面财务快照。"""

    if income_as_of.empty:
        return _empty_financial_snapshot()

    result = income_as_of.copy()
    result["year"] = result["end_date"].dt.year
    result["month"] = result["end_date"].dt.month
    period_keys = ["ts_code", "year", "month"]
    duplicated_periods = result.duplicated(period_keys, keep=False)
    if duplicated_periods.any():
        examples = result.loc[duplicated_periods, "ts_code"].drop_duplicates().head(5)
        raise ValueError(
            "同一股票、年度和月份存在多个报告期，例如："
            + ", ".join(examples.astype(str))
        )

    lookup = result.set_index(period_keys)
    prior_same_index = pd.MultiIndex.from_arrays(
        [
            result["ts_code"],
            result["year"] - 1,
            result["month"],
        ],
        names=period_keys,
    )
    prior_annual_index = pd.MultiIndex.from_arrays(
        [
            result["ts_code"],
            result["year"] - 1,
            pd.Series(12, index=result.index),
        ],
        names=period_keys,
    )
    prior_same_announcement = lookup["announcement_date"].reindex(
        prior_same_index
    )
    prior_annual_announcement = lookup["announcement_date"].reindex(
        prior_annual_index
    )
    result["prior_same_announcement_date"] = prior_same_announcement.to_numpy()
    result["prior_annual_announcement_date"] = prior_annual_announcement.to_numpy()

    values = ["revenue", "oper_cost", "n_income_attr_p"]
    for value in values:
        prior_same = lookup[value].reindex(prior_same_index).to_numpy()
        prior_annual = lookup[value].reindex(prior_annual_index).to_numpy()
        ttm = result[value] + prior_annual - prior_same
        result[f"{value}_ttm"] = ttm.where(
            ~result["month"].eq(12), result[value]
        )

    annual_rows = result["month"].eq(12)
    result.loc[
        annual_rows,
        ["prior_same_announcement_date", "prior_annual_announcement_date"],
    ] = pd.NaT
    result = result.rename(
        columns={
            "announcement_date": "current_income_announcement_date",
            "n_income_attr_p_ttm": "net_profit_ttm",
        }
    )
    result["gross_profit_ttm"] = (
        result["revenue_ttm"] - result["oper_cost_ttm"]
    )
    result = (
        result.sort_values(["ts_code", "end_date"])
        .drop_duplicates("ts_code", keep="last")
    )

    balance_columns = [
        "ts_code",
        "end_date",
        "announcement_date",
        "total_assets",
        "total_hldr_eqy_exc_min_int",
    ]
    balance_current = balance_as_of[balance_columns].rename(
        columns={
            "announcement_date": "balance_announcement_date",
            "total_hldr_eqy_exc_min_int": "total_equity",
        }
    )
    snapshot = result.merge(
        balance_current,
        on=["ts_code", "end_date"],
        how="left",
        validate="one_to_one",
    )
    availability_columns = [
        "current_income_announcement_date",
        "prior_same_announcement_date",
        "prior_annual_announcement_date",
        "balance_announcement_date",
    ]
    snapshot["financial_available_date"] = snapshot[
        availability_columns
    ].max(axis=1)
    snapshot["announcement_date"] = snapshot["financial_available_date"]
    snapshot["date"] = pd.Timestamp(as_of_date)
    snapshot = snapshot.rename(columns={"ts_code": "stock_code"})

    columns = _empty_financial_snapshot().columns
    return snapshot.loc[:, columns].sort_values("stock_code").reset_index(drop=True)


def build_monthly_financial_snapshots(
    income: pd.DataFrame,
    balancesheet: pd.DataFrame,
    rebalance_dates: pd.DataFrame,
) -> pd.DataFrame:
    """为每个调仓日构建严格时点可得的财务指标快照。

    每个调仓日都会先选取当时已公告的利润表和资产负债表版本，再使用这些版本
    计算 TTM。后续修订只会影响修订公告日之后的快照，不会覆盖历史时点。
    """

    income_versions = prepare_report_versions(
        income,
        value_columns=["revenue", "oper_cost", "n_income_attr_p"],
    )
    balance_versions = prepare_report_versions(
        balancesheet,
        value_columns=["total_assets", "total_hldr_eqy_exc_min_int"],
    )
    dates = (
        pd.to_datetime(rebalance_dates["date"], errors="coerce")
        .dropna()
        .drop_duplicates()
        .sort_values()
    )

    snapshots: list[pd.DataFrame] = []
    for as_of_date in dates:
        income_as_of = select_latest_report_versions_as_of(
            income_versions, as_of_date
        )
        balance_as_of = select_latest_report_versions_as_of(
            balance_versions, as_of_date
        )
        snapshots.append(
            _calculate_financial_snapshot_as_of(
                income_as_of,
                balance_as_of,
                as_of_date,
            )
        )

    if not snapshots:
        return _empty_financial_snapshot()
    panel = pd.concat(snapshots, ignore_index=True)
    if panel.duplicated(["date", "stock_code"]).any():
        raise ValueError("月末财务快照存在重复的 date-stock_code 主键")
    future_information = panel["financial_available_date"].gt(panel["date"])
    if future_information.any():
        raise ValueError("月末财务快照使用了调仓日之后才公告的财报版本")
    return panel.sort_values(["date", "stock_code"]).reset_index(drop=True)


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
    financial_snapshots: pd.DataFrame,
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

    start, end = pd.Timestamp(config.study_start), pd.Timestamp(config.study_end)
    rebalance_dates = _monthly_rebalance_dates(calendar)
    rebalance_dates = rebalance_dates.loc[
        rebalance_dates["date"].between(start, end)
    ]
    monthly = price_daily.merge(rebalance_dates, on="date", how="inner").copy()

    stock_information = stock_basic[
        ["ts_code", "list_date", "delist_date"]
    ].rename(columns={"ts_code": "stock_code"})
    duplicated_codes = stock_information["stock_code"].duplicated(keep=False)
    if duplicated_codes.any():
        examples = (
            stock_information.loc[duplicated_codes, "stock_code"]
            .drop_duplicates()
            .head(5)
        )
        raise ValueError(
            "stock_basic 中存在重复股票代码，例如："
            + ", ".join(examples.astype(str))
        )

    known_codes = set(stock_information["stock_code"].dropna().astype(str))
    unknown_codes = sorted(
        set(monthly["stock_code"].dropna().astype(str)).difference(known_codes)
    )
    if unknown_codes:
        examples = ", ".join(unknown_codes[:5])
        raise ValueError(
            "price_daily 中存在 stock_basic 未收录的股票代码："
            f"共 {len(unknown_codes)} 个，例如 {examples}。"
            "请在构建日频面板时使用沪深股票白名单。"
        )

    monthly = monthly.merge(
        stock_information,
        on="stock_code",
        how="left",
        validate="many_to_one",
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
        monthly["is_st"] = (
            monthly.pop("is_st_from_source").eq(True) | monthly["is_st"]
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
        monthly["is_suspended"] = (
            monthly.pop("is_suspended_from_source").eq(True)
            | monthly["is_suspended"]
        )

    required_financial_columns = {"stock_code", "date", "financial_available_date"}
    missing_financial_columns = required_financial_columns.difference(
        financial_snapshots.columns
    )
    if missing_financial_columns:
        raise ValueError(
            "月末财务快照缺少必需字段："
            f"{sorted(missing_financial_columns)}"
        )
    if financial_snapshots.duplicated(["date", "stock_code"]).any():
        raise ValueError("月末财务快照存在重复的 date-stock_code 主键")
    monthly = monthly.merge(
        financial_snapshots,
        on=["stock_code", "date"],
        how="left",
        validate="one_to_one",
    )
    future_financials = monthly["financial_available_date"].gt(monthly["date"])
    if future_financials.any():
        raise ValueError("月度股票池包含调仓日之后才可得的财务数据")
    if industry_membership is not None:
        monthly = attach_historical_industry(monthly, industry_membership)
    else:
        monthly["industry"] = pd.NA

    monthly["report_age_days"] = (monthly["date"] - monthly["end_date"]).dt.days
    monthly["announcement_age_days"] = (
        monthly["date"] - monthly["financial_available_date"]
    ).dt.days
    monthly["is_gross_profit_applicable"] = ~monthly["industry"].isin(
        GROSS_PROFIT_NOT_APPLICABLE_INDUSTRIES
    )
    monthly["has_gross_profit_data"] = monthly[
        ["gross_profit_ttm", "total_assets"]
    ].notna().all(axis=1)
    monthly["has_gp_factor_data"] = (
        monthly["is_gross_profit_applicable"]
        & monthly["has_gross_profit_data"]
    )

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
    stocks = select_sh_sz_stock_basic(
        pd.read_parquet(raw_directory / "stock_basic.parquet")
    )
    allowed_stock_codes = set(stocks["ts_code"].dropna().astype(str))
    income = filter_stock_records(
        pd.read_parquet(raw_directory / "income.parquet"),
        allowed_stock_codes,
    )
    balancesheet = filter_stock_records(
        pd.read_parquet(raw_directory / "balancesheet.parquet"),
        allowed_stock_codes,
    )
    membership = filter_stock_records(
        pd.read_parquet(raw_directory / "sw_industry_membership.parquet"),
        allowed_stock_codes,
    )
    st_records = filter_stock_records(
        read_date_partitions(raw_directory / "stock_st"),
        allowed_stock_codes,
    )
    suspension_records = filter_stock_records(
        read_date_partitions(raw_directory / "suspend_d"),
        allowed_stock_codes,
    )

    prices = build_price_daily(
        raw_directory,
        allowed_stock_codes=allowed_stock_codes,
    )
    rebalance_dates = _monthly_rebalance_dates(calendar)
    study_start = pd.Timestamp(config.study_start)
    study_end = pd.Timestamp(config.study_end)
    rebalance_dates = rebalance_dates.loc[
        rebalance_dates["date"].between(study_start, study_end)
    ]
    financial_snapshots = build_monthly_financial_snapshots(
        income,
        balancesheet,
        rebalance_dates,
    )
    universe = build_monthly_universe(
        prices,
        calendar,
        stocks,
        financial_snapshots,
        st_records=st_records,
        suspension_records=suspension_records,
        industry_membership=membership,
        config=config,
    )
    return save_notebook01_outputs(
        prices,
        universe,
        processed_directory=processed_directory,
    )
