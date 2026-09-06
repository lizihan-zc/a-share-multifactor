"""从 Tushare 下载 Notebook 01 所需的原始数据。

本模块只负责调用 Tushare 接口，并将接口返回尽量原样写入 ``data/raw``。
股票池过滤和处理后数据构建位于 ``src/data/preprocess.py``。

下载原始数据的工作流：

1. 创建客户端：调用 ``create_tushare_client``，从显式参数或 ``.env`` 中读取
   ``TUSHARE_TOKEN``。
2. 下载基础数据：调用 ``download_trade_calendar`` 和 ``download_stock_basic``，
   分别保存交易日历与包含退市股票的基础信息。
3. 下载日频数据：调用 ``download_market_partitions``，按交易日分区下载 ``daily``、
   ``adj_factor``、``daily_basic``、``stk_limit``、``suspend_d`` 和 ``stock_st``。
4. 下载财务与行业数据：调用 ``download_financial_statements`` 取得利润表和资产负债表；
   调用 ``download_industry_membership`` 取得历史申万行业有效区间。
5. 如需执行完整下载流程，调用 ``run_download_pipeline``；如只需补充某类原始数据，
   则调用对应的单个下载函数。
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

import pandas as pd


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIRECTORY = PROJECT_ROOT / "data" / "raw"


MARKET_DATASETS: Mapping[str, tuple[str, Mapping[str, str]]] = {
    "daily": (
        "ts_code,trade_date,open,high,low,close,pre_close,vol,amount",
        {},
    ),
    "adj_factor": ("ts_code,trade_date,adj_factor", {}),
    "daily_basic": ("ts_code,trade_date,total_mv,limit_status", {}),
    "stk_limit": (
        "ts_code,trade_date,pre_close,up_limit,down_limit",
        {},
    ),
    "suspend_d": (
        "ts_code,trade_date,suspend_timing,suspend_type",
        {"suspend_type": "S"},
    ),
    "stock_st": ("ts_code,name,trade_date,type,type_name", {}),
}


@dataclass(frozen=True)
class DownloadConfig:
    """定义 Tushare 原始数据下载区间和证券范围。"""

    start_date: str
    end_date: str
    exchanges: tuple[str, ...] = ("SSE", "SZSE")
    list_statuses: tuple[str, ...] = ("L", "D", "P")

    def __post_init__(self) -> None:
        if pd.Timestamp(self.start_date) > pd.Timestamp(self.end_date):
            raise ValueError("start_date 必须早于或等于 end_date")


def create_tushare_client(token: Optional[str] = None) -> Any:
    """使用显式 Token 或 ``TUSHARE_TOKEN`` 创建 Tushare Pro 客户端。

    将 Tushare 延迟到此处导入，使模块在未安装 SDK 时仍可进行静态检查。
    """

    try:
        import tushare as ts
        from dotenv import load_dotenv
    except ImportError as exc:
        raise ImportError("下载数据前请先安装 tushare 和 python-dotenv。") from exc

    load_dotenv()
    token = token or os.environ.get("TUSHARE_TOKEN")
    if not token:
        raise ValueError("请设置 TUSHARE_TOKEN，或显式传入 token。")
    return ts.pro_api(token)


def call_with_retry(
    request: Callable[..., pd.DataFrame],
    /,
    *,
    attempts: int = 5,
    base_wait_seconds: float = 1.0,
    **kwargs: Any,
) -> pd.DataFrame:
    """调用单个 API；发生瞬时错误时使用指数退避重试。"""

    if attempts < 1:
        raise ValueError("attempts 至少应为 1")

    for attempt in range(attempts):
        try:
            return request(**kwargs)
        except Exception:
            if attempt == attempts - 1:
                raise
            wait = base_wait_seconds * (2**attempt)
            LOGGER.warning("Tushare 请求失败，将在 %.1f 秒后重试", wait)
            time.sleep(wait)

    raise RuntimeError("不应执行到此处")


def _write_parquet_atomic(frame: pd.DataFrame, path: Path) -> None:
    """原子写入 Parquet，防止中断下载生成的残缺文件被复用。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def download_trade_calendar(
    pro: Any,
    *,
    start_date: str,
    end_date: str,
    output_path: Path,
) -> pd.DataFrame:
    """下载上交所开市日，并将完整交易日历保存到 ``data/raw``。"""

    calendar = call_with_retry(
        pro.trade_cal,
        exchange="SSE",
        start_date=start_date,
        end_date=end_date,
        is_open="1",
        fields="exchange,cal_date,is_open,pretrade_date",
    )
    calendar = calendar.drop_duplicates("cal_date").sort_values("cal_date")
    _write_parquet_atomic(calendar, output_path)
    return calendar


def download_stock_basic(
    pro: Any,
    *,
    exchanges: Sequence[str],
    list_statuses: Sequence[str],
    output_path: Path,
) -> pd.DataFrame:
    """下载上市、退市及暂停上市 A 股的基础信息。

    必须下载全部状态：只使用当前上市股票会在历史回测中遗漏退市股票。
    """

    fields = (
        "ts_code,symbol,name,market,exchange,curr_type,list_status,"
        "list_date,delist_date"
    )
    parts: list[pd.DataFrame] = []
    for exchange in exchanges:
        for list_status in list_statuses:
            part = call_with_retry(
                pro.stock_basic,
                exchange=exchange,
                list_status=list_status,
                fields=fields,
            )
            parts.append(part)

    if not parts:
        raise ValueError("至少需要指定一个交易所和一个上市状态")

    stocks = pd.concat(parts, ignore_index=True)
    stocks = stocks.loc[stocks["curr_type"].eq("CNY")].copy()
    stocks = stocks.drop_duplicates("ts_code", keep="last").sort_values("ts_code")
    _write_parquet_atomic(stocks, output_path)
    return stocks


def download_market_partitions(
    pro: Any,
    trade_dates: Iterable[str],
    *,
    raw_directory: Path,
    api_names: Sequence[str] = tuple(MARKET_DATASETS),
    pause_seconds: float = 0.2,
) -> None:
    """将日频行情和状态接口下载为可断点续跑的日期分区。

    已存在的文件不会被改写，因此 API 或网络失败后可以安全重跑，并避免重复请求缓存数据。
    """

    unknown = set(api_names).difference(MARKET_DATASETS)
    if unknown:
        raise ValueError(f"不支持的行情数据集：{sorted(unknown)}")

    for trade_date in trade_dates:
        for api_name in api_names:
            fields, params = MARKET_DATASETS[api_name]
            output_path = (
                raw_directory / api_name / f"trade_date={trade_date}.parquet"
            )
            if output_path.exists():
                continue

            endpoint = getattr(pro, api_name)
            frame = call_with_retry(
                endpoint,
                trade_date=str(trade_date),
                fields=fields,
                **params,
            )
            _write_parquet_atomic(frame, output_path)
            if pause_seconds:
                time.sleep(pause_seconds)


def download_financial_statements(
    pro: Any,
    stock_codes: Iterable[str],
    *,
    start_date: str,
    end_date: str,
    raw_directory: Path,
    pause_seconds: float = 0.2,
) -> None:
    """按股票下载利润表和资产负债表。

    标准 Tushare 接口按股票提供历史财务数据。具备 VIP 权限的账户可使用
    ``income_vip`` 和 ``balancesheet_vip``，另行实现按报告期下载。
    """

    stock_codes = list(stock_codes)
    datasets = {
        "income": (
            "ts_code,ann_date,f_ann_date,end_date,report_type,comp_type,"
            "revenue,oper_cost,n_income_attr_p,update_flag"
        ),
        "balancesheet": (
            "ts_code,ann_date,f_ann_date,end_date,report_type,comp_type,"
            "total_assets,total_hldr_eqy_exc_min_int,"
            "total_hldr_eqy_inc_min_int,update_flag"
        ),
    }

    for api_name, fields in datasets.items():
        output_path = raw_directory / f"{api_name}.parquet"
        existing = (
            pd.read_parquet(output_path) if output_path.exists() else pd.DataFrame()
        )
        done = set(existing.get("ts_code", pd.Series(dtype="string")).dropna())
        parts = [existing] if not existing.empty else []

        endpoint = getattr(pro, api_name)
        for stock_code in stock_codes:
            if stock_code in done:
                continue
            frame = call_with_retry(
                endpoint,
                ts_code=stock_code,
                start_date=start_date,
                end_date=end_date,
                fields=fields,
            )
            parts.append(frame)
            if pause_seconds:
                time.sleep(pause_seconds)

        combined = (
            pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        )
        _write_parquet_atomic(combined, output_path)


def download_industry_membership(
    pro: Any,
    stock_codes: Iterable[str],
    *,
    output_path: Path,
    pause_seconds: float = 0.2,
) -> pd.DataFrame:
    """下载当前及历史申万行业成分的有效区间。

    同时保留当前（``is_new='Y'``）和历史（``is_new='N'``）记录，以便预处理阶段
    按历史调仓日确定股票所属行业。
    """

    parts: list[pd.DataFrame] = []
    fields = (
        "l1_code,l1_name,l2_code,l2_name,l3_code,l3_name,"
        "ts_code,name,in_date,out_date,is_new"
    )
    for stock_code in stock_codes:
        for is_new in ("Y", "N"):
            frame = call_with_retry(
                pro.index_member_all,
                ts_code=stock_code,
                is_new=is_new,
                fields=fields,
            )
            parts.append(frame)
            if pause_seconds:
                time.sleep(pause_seconds)

    membership = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if not membership.empty:
        membership = membership.drop_duplicates().sort_values(
            ["ts_code", "in_date"]
        )
    _write_parquet_atomic(membership, output_path)
    return membership


def run_download_pipeline(
    config: DownloadConfig,
    *,
    raw_directory: Path = DEFAULT_RAW_DIRECTORY,
    token: Optional[str] = None,
) -> dict[str, Path]:
    """运行完整下载流程，并将所有 Tushare 原始数据写入 ``data/raw``。"""

    pro = create_tushare_client(token)
    calendar_path = raw_directory / "trade_calendar.parquet"
    stock_basic_path = raw_directory / "stock_basic.parquet"
    industry_path = raw_directory / "sw_industry_membership.parquet"

    calendar = download_trade_calendar(
        pro,
        start_date=config.start_date,
        end_date=config.end_date,
        output_path=calendar_path,
    )
    stocks = download_stock_basic(
        pro,
        exchanges=config.exchanges,
        list_statuses=config.list_statuses,
        output_path=stock_basic_path,
    )
    download_market_partitions(
        pro,
        calendar["cal_date"].astype(str),
        raw_directory=raw_directory,
    )

    financial_start = (
        pd.Timestamp(config.start_date) - pd.DateOffset(years=1)
    ).strftime("%Y%m%d")
    download_financial_statements(
        pro,
        stocks["ts_code"],
        start_date=financial_start,
        end_date=config.end_date,
        raw_directory=raw_directory,
    )
    download_industry_membership(
        pro,
        stocks["ts_code"],
        output_path=industry_path,
    )

    return {
        "trade_calendar": calendar_path,
        "stock_basic": stock_basic_path,
        "daily_partitions": raw_directory / "daily",
        "adj_factor_partitions": raw_directory / "adj_factor",
        "daily_basic_partitions": raw_directory / "daily_basic",
        "stk_limit_partitions": raw_directory / "stk_limit",
        "suspend_partitions": raw_directory / "suspend_d",
        "stock_st_partitions": raw_directory / "stock_st",
        "income": raw_directory / "income.parquet",
        "balancesheet": raw_directory / "balancesheet.parquet",
        "industry_membership": industry_path,
    }
