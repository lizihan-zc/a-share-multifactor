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
4. 下载财务与行业数据：调用 ``download_financial_statements``，按股票分区取得利润表
   和资产负债表，并维护区间覆盖清单；调用 ``download_industry_membership`` 取得历史
   申万行业有效区间。
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

FINANCIAL_DATASETS: Mapping[str, str] = {
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
FINANCIAL_MANIFEST_NAME = "financial_download_manifest.parquet"
FINANCIAL_SCHEMA_VERSION = 1
FINANCIAL_MANIFEST_COLUMNS = (
    "dataset",
    "stock_code",
    "range_start",
    "range_end",
    "status",
    "row_count",
    "updated_at",
    "schema_version",
    "fields_signature",
    "source",
)


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


def _empty_financial_manifest() -> pd.DataFrame:
    """返回财务分区下载清单的稳定空表结构。"""

    return pd.DataFrame(
        {
            "dataset": pd.Series(dtype="string"),
            "stock_code": pd.Series(dtype="string"),
            "range_start": pd.Series(dtype="datetime64[ns]"),
            "range_end": pd.Series(dtype="datetime64[ns]"),
            "status": pd.Series(dtype="string"),
            "row_count": pd.Series(dtype="int64"),
            "updated_at": pd.Series(dtype="datetime64[ns]"),
            "schema_version": pd.Series(dtype="int64"),
            "fields_signature": pd.Series(dtype="string"),
            "source": pd.Series(dtype="string"),
        }
    )


def _load_financial_manifest(path: Path) -> pd.DataFrame:
    """读取并校验逐股财务下载清单。"""

    if not path.is_file():
        return _empty_financial_manifest()
    manifest = pd.read_parquet(path)
    missing = sorted(set(FINANCIAL_MANIFEST_COLUMNS).difference(manifest.columns))
    if missing:
        raise ValueError(f"财务下载清单缺少字段：{missing}")
    manifest = manifest.loc[:, list(FINANCIAL_MANIFEST_COLUMNS)].copy()
    manifest["range_start"] = pd.to_datetime(
        manifest["range_start"], errors="coerce"
    )
    manifest["range_end"] = pd.to_datetime(manifest["range_end"], errors="coerce")
    manifest["updated_at"] = pd.to_datetime(manifest["updated_at"], errors="coerce")
    invalid_range = (
        manifest["range_start"].isna()
        | manifest["range_end"].isna()
        | manifest["range_start"].gt(manifest["range_end"])
    )
    if invalid_range.any():
        raise ValueError("财务下载清单包含无效日期区间")
    return manifest


def _financial_partition_path(
    raw_directory: Path,
    dataset: str,
    stock_code: str,
) -> Path:
    """返回单个数据集、单只股票的 Parquet 分区路径。"""

    stock_code = str(stock_code).strip().upper()
    if (
        not stock_code
        or Path(stock_code).name != stock_code
        or stock_code in {".", ".."}
    ):
        raise ValueError(f"非法股票代码，无法构造分区路径：{stock_code!r}")
    return raw_directory / dataset / f"ts_code={stock_code}.parquet"


def _parse_tushare_dates(values: pd.Series) -> pd.Series:
    """兼容 Tushare 字符串日期以及 Parquet 中已有的时间戳。"""

    text = values.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    compact = pd.to_datetime(text, format="%Y%m%d", errors="coerce")
    fallback = pd.to_datetime(values, errors="coerce")
    return compact.fillna(fallback)


def _effective_announcement_date(frame: pd.DataFrame) -> pd.Series:
    """使用实际公告日，缺失时回退到公告日。"""

    if "ann_date" not in frame.columns:
        return pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns]")
    ann_date = _parse_tushare_dates(frame["ann_date"])
    if "f_ann_date" not in frame.columns:
        return ann_date
    actual = _parse_tushare_dates(frame["f_ann_date"])
    return actual.fillna(ann_date)


def _calculate_missing_date_ranges(
    requested_start: str,
    requested_end: str,
    completed_ranges: Iterable[tuple[pd.Timestamp, pd.Timestamp]],
) -> list[tuple[str, str]]:
    """从已完成区间中扣除当前请求，返回尚需请求的闭区间。"""

    start = pd.Timestamp(requested_start).normalize()
    end = pd.Timestamp(requested_end).normalize()
    if start > end:
        raise ValueError("start_date 必须早于或等于 end_date")

    clipped: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for range_start, range_end in completed_ranges:
        left = max(start, pd.Timestamp(range_start).normalize())
        right = min(end, pd.Timestamp(range_end).normalize())
        if left <= right:
            clipped.append((left, right))
    clipped.sort()

    merged: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    one_day = pd.Timedelta(days=1)
    for left, right in clipped:
        if not merged or left > merged[-1][1] + one_day:
            merged.append((left, right))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], right))

    missing: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    cursor = start
    for left, right in merged:
        if cursor < left:
            missing.append((cursor, left - one_day))
        cursor = max(cursor, right + one_day)
    if cursor <= end:
        missing.append((cursor, end))
    return [
        (left.strftime("%Y%m%d"), right.strftime("%Y%m%d"))
        for left, right in missing
    ]


def _merge_financial_records(
    existing: pd.DataFrame,
    downloaded: pd.DataFrame,
) -> pd.DataFrame:
    """合并新旧财报，并保留不同公告日和报表类型的历史版本。"""

    if existing.empty and downloaded.empty:
        columns = list(dict.fromkeys([*existing.columns, *downloaded.columns]))
        return pd.DataFrame(columns=columns)

    required = {"ts_code", "end_date", "ann_date", "report_type"}
    for name, frame in (("existing", existing), ("downloaded", downloaded)):
        if frame.empty:
            continue
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"{name} 财务数据缺少版本字段：{missing}")

    old = existing.copy()
    new = downloaded.copy()
    old["_download_priority"] = 0
    new["_download_priority"] = 1
    column_order = list(dict.fromkeys([*old.columns, *new.columns]))
    # 丢掉单个输入中全空的列后再拼接，避免 pandas 对全空列 dtype 推断的
    # 兼容性警告；随后按并集补回这些列，落盘结构不会改变。
    merge_parts = [
        part.dropna(axis="columns", how="all")
        for part in (old, new)
        if not part.empty
    ]
    combined = pd.concat(merge_parts, ignore_index=True, sort=False).reindex(
        columns=column_order
    )
    combined["_source_order"] = range(len(combined))
    combined["_announcement_key"] = _effective_announcement_date(combined)
    combined["_end_date_key"] = _parse_tushare_dates(combined["end_date"])
    combined["_report_type_key"] = combined["report_type"].astype("string")
    combined["_update_priority"] = pd.to_numeric(
        combined.get("update_flag", pd.Series(index=combined.index, dtype="float64")),
        errors="coerce",
    ).fillna(-1)
    key = ["ts_code", "_end_date_key", "_announcement_key", "_report_type_key"]
    combined = (
        combined.sort_values(
            [*key, "_update_priority", "_download_priority", "_source_order"],
            na_position="first",
        )
        .drop_duplicates(key, keep="last")
        .sort_values(["ts_code", "_end_date_key", "_announcement_key"])
        .drop(
            columns=[
                "_download_priority",
                "_source_order",
                "_announcement_key",
                "_end_date_key",
                "_report_type_key",
                "_update_priority",
            ]
        )
        .reset_index(drop=True)
    )
    return combined


def _manifest_completed_ranges(
    manifest: pd.DataFrame,
    *,
    dataset: str,
    stock_code: str,
    fields_signature: str,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """取得与当前字段版本相符的已完成下载区间。"""

    selected = manifest.loc[
        manifest["dataset"].eq(dataset)
        & manifest["stock_code"].eq(stock_code)
        & manifest["status"].isin(["complete", "inferred_existing"])
        & manifest["schema_version"].eq(FINANCIAL_SCHEMA_VERSION)
        & manifest["fields_signature"].eq(fields_signature)
    ]
    return list(zip(selected["range_start"], selected["range_end"]))


def _append_manifest_record(
    manifest: pd.DataFrame,
    *,
    dataset: str,
    stock_code: str,
    range_start: str,
    range_end: str,
    status: str,
    row_count: int,
    fields_signature: str,
    source: str,
) -> pd.DataFrame:
    """追加一条覆盖记录，并使同一区间的最新状态具有唯一性。"""

    record = pd.DataFrame(
        [
            {
                "dataset": dataset,
                "stock_code": stock_code,
                "range_start": pd.Timestamp(range_start),
                "range_end": pd.Timestamp(range_end),
                "status": status,
                "row_count": int(row_count),
                "updated_at": pd.Timestamp.now(tz="UTC").tz_localize(None),
                "schema_version": FINANCIAL_SCHEMA_VERSION,
                "fields_signature": fields_signature,
                "source": source,
            }
        ]
    )
    combined = pd.concat([manifest, record], ignore_index=True)
    deduplication_key = [
        "dataset",
        "stock_code",
        "range_start",
        "range_end",
        "schema_version",
        "fields_signature",
    ]
    return (
        combined.sort_values("updated_at")
        .drop_duplicates(deduplication_key, keep="last")
        .loc[:, list(FINANCIAL_MANIFEST_COLUMNS)]
        .reset_index(drop=True)
    )


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
    manifest_path: Optional[Path] = None,
    pause_seconds: float = 0.2,
) -> dict[str, Path]:
    """按股票增量下载利润表和资产负债表，并逐股原子保存。

    标准 Tushare 接口按股票提供历史财务数据。具备 VIP 权限的账户可使用
    ``income_vip`` 和 ``balancesheet_vip``，另行实现按报告期下载。当前实现以
    下载清单记录成功请求的闭区间，只补充未覆盖区间；API 返回空表也会记录为
    已完成，避免重复请求。

    旧版 ``income.parquet`` 和 ``balancesheet.parquet`` 会按股票逐步迁移到分区
    目录。由于旧文件没有下载清单，只能把每只股票已有公告日的最小值到最大值
    记为推断覆盖，区间两端仍会按当前请求补齐。
    """

    requested_start = pd.Timestamp(start_date).normalize()
    requested_end = pd.Timestamp(end_date).normalize()
    if requested_start > requested_end:
        raise ValueError("start_date 必须早于或等于 end_date")
    normalized_codes = [
        str(code).strip().upper() for code in stock_codes if pd.notna(code)
    ]
    normalized_codes = list(dict.fromkeys(code for code in normalized_codes if code))
    if not normalized_codes:
        raise ValueError("stock_codes 不能为空")

    raw_directory = Path(raw_directory)
    manifest_path = (
        raw_directory / FINANCIAL_MANIFEST_NAME
        if manifest_path is None
        else Path(manifest_path)
    )
    manifest = _load_financial_manifest(manifest_path)
    outputs: dict[str, Path] = {}

    for api_name, fields in FINANCIAL_DATASETS.items():
        fields_signature = fields
        field_columns = fields.split(",")
        partition_directory = raw_directory / api_name
        partition_directory.mkdir(parents=True, exist_ok=True)
        outputs[api_name] = partition_directory

        legacy_path = raw_directory / f"{api_name}.parquet"
        legacy = (
            pd.read_parquet(legacy_path)
            if legacy_path.is_file()
            else pd.DataFrame()
        )
        legacy_groups = None
        legacy_codes: set[str] = set()
        if not legacy.empty:
            if "ts_code" not in legacy.columns:
                raise ValueError(f"旧版 {legacy_path.name} 缺少 ts_code 字段")
            legacy["ts_code"] = (
                legacy["ts_code"].astype("string").str.strip().str.upper()
            )
            legacy_groups = legacy.groupby("ts_code", sort=False, observed=True)
            legacy_codes = set(legacy["ts_code"].dropna().astype(str))

        endpoint = getattr(pro, api_name)
        for stock_code in normalized_codes:
            partition_path = _financial_partition_path(
                raw_directory, api_name, stock_code
            )
            if partition_path.is_file():
                current = pd.read_parquet(partition_path)
                has_current_schema = set(field_columns).issubset(current.columns)
            elif stock_code in legacy_codes and legacy_groups is not None:
                legacy_stock = legacy_groups.get_group(stock_code).copy()
                has_current_schema = set(field_columns).issubset(
                    legacy_stock.columns
                )
                current = _merge_financial_records(
                    pd.DataFrame(columns=field_columns), legacy_stock
                )
                if has_current_schema:
                    _write_parquet_atomic(current, partition_path)
            else:
                current = pd.DataFrame(columns=field_columns)
                has_current_schema = True
            if "ts_code" in current.columns:
                current["ts_code"] = (
                    current["ts_code"].astype("string").str.strip().str.upper()
                )
                current_codes = set(current["ts_code"].dropna().astype(str))
                unexpected_codes = current_codes.difference({stock_code})
                if unexpected_codes:
                    raise ValueError(
                        f"{partition_path.name} 包含其他股票："
                        f"{sorted(unexpected_codes)[:5]}"
                    )
            # 旧版数据可能缺少后来新增的下载字段。保留旧行并补空列，但仍通过
            # has_current_schema=False 触发完整区间重取。
            for column in field_columns:
                if column not in current.columns:
                    current[column] = pd.NA

            completed_ranges = (
                _manifest_completed_ranges(
                    manifest,
                    dataset=api_name,
                    stock_code=stock_code,
                    fields_signature=fields_signature,
                )
                if partition_path.is_file() and has_current_schema
                else []
            )

            # 兼容没有清单的旧文件或写入数据后、更新清单前发生中断的情况。
            if not completed_ranges and not current.empty and has_current_schema:
                announcement_dates = _effective_announcement_date(current).dropna()
                if not announcement_dates.empty:
                    inferred_start = announcement_dates.min().normalize()
                    inferred_end = announcement_dates.max().normalize()
                    manifest = _append_manifest_record(
                        manifest,
                        dataset=api_name,
                        stock_code=stock_code,
                        range_start=inferred_start.strftime("%Y%m%d"),
                        range_end=inferred_end.strftime("%Y%m%d"),
                        status="inferred_existing",
                        row_count=len(current),
                        fields_signature=fields_signature,
                        source="existing_partition",
                    )
                    _write_parquet_atomic(manifest, manifest_path)
                    completed_ranges = [(inferred_start, inferred_end)]

            missing_ranges = _calculate_missing_date_ranges(
                requested_start.strftime("%Y%m%d"),
                requested_end.strftime("%Y%m%d"),
                completed_ranges,
            )
            for missing_start, missing_end in missing_ranges:
                downloaded = call_with_retry(
                    endpoint,
                    ts_code=stock_code,
                    start_date=missing_start,
                    end_date=missing_end,
                    fields=fields,
                )
                if not isinstance(downloaded, pd.DataFrame):
                    raise TypeError(f"{api_name} API 必须返回 pandas DataFrame")
                if downloaded.empty:
                    downloaded = pd.DataFrame(columns=field_columns)
                else:
                    missing_fields = sorted(
                        set(field_columns).difference(downloaded.columns)
                    )
                    if missing_fields:
                        raise ValueError(
                            f"{api_name} API 返回缺少字段：{missing_fields}"
                        )
                    downloaded["ts_code"] = (
                        downloaded["ts_code"]
                        .astype("string")
                        .str.strip()
                        .str.upper()
                    )
                    returned_codes = set(downloaded["ts_code"].dropna().astype(str))
                    if returned_codes != {stock_code}:
                        raise ValueError(
                            f"{api_name} API 返回的股票代码与请求不一致："
                            f"请求 {stock_code}，返回 {sorted(returned_codes)[:5]}"
                        )

                current = _merge_financial_records(current, downloaded)
                # 必须先提交数据，再提交清单；中断时最多重复请求，不会错误跳过。
                _write_parquet_atomic(current, partition_path)
                manifest = _append_manifest_record(
                    manifest,
                    dataset=api_name,
                    stock_code=stock_code,
                    range_start=missing_start,
                    range_end=missing_end,
                    status="complete",
                    row_count=len(downloaded),
                    fields_signature=fields_signature,
                    source="tushare_api",
                )
                _write_parquet_atomic(manifest, manifest_path)
                if pause_seconds:
                    time.sleep(pause_seconds)

    outputs["manifest"] = manifest_path
    return outputs


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
    financial_outputs = download_financial_statements(
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
        "income": financial_outputs["income"],
        "balancesheet": financial_outputs["balancesheet"],
        "financial_manifest": financial_outputs["manifest"],
        "industry_membership": industry_path,
    }
