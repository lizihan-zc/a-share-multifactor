"""逐股票财务分区下载和覆盖清单测试。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.preprocess import read_financial_partitions
from src.data.universe import (
    _calculate_missing_date_ranges,
    _merge_financial_records,
    download_financial_statements,
)


class FakeFinancialClient:
    """记录请求并返回结构稳定的单行财报。"""

    def __init__(self, *, empty: bool = False) -> None:
        self.empty = empty
        self.calls: list[dict[str, str]] = []

    def _request(self, dataset: str, **kwargs: str) -> pd.DataFrame:
        self.calls.append({"dataset": dataset, **kwargs})
        columns = kwargs["fields"].split(",")
        if self.empty:
            return pd.DataFrame(columns=columns)
        values = {column: pd.NA for column in columns}
        values.update(
            {
                "ts_code": kwargs["ts_code"],
                "ann_date": kwargs["start_date"],
                "f_ann_date": kwargs["start_date"],
                "end_date": kwargs["start_date"],
                "report_type": "1",
                "comp_type": "1",
                "update_flag": "0",
            }
        )
        if dataset == "income":
            values.update(
                {"revenue": 10.0, "oper_cost": 6.0, "n_income_attr_p": 2.0}
            )
        else:
            values.update(
                {
                    "total_assets": 100.0,
                    "total_hldr_eqy_exc_min_int": 40.0,
                    "total_hldr_eqy_inc_min_int": 45.0,
                }
            )
        return pd.DataFrame([values], columns=columns)

    def income(self, **kwargs: str) -> pd.DataFrame:
        return self._request("income", **kwargs)

    def balancesheet(self, **kwargs: str) -> pd.DataFrame:
        return self._request("balancesheet", **kwargs)


def test_missing_date_ranges_can_fill_both_sides() -> None:
    missing = _calculate_missing_date_ranges(
        "20190101",
        "20211231",
        [(pd.Timestamp("2020-01-01"), pd.Timestamp("2020-12-31"))],
    )
    assert missing == [("20190101", "20191231"), ("20210101", "20211231")]


def test_financial_download_reuses_manifest_and_only_extends_ranges(
    tmp_path: Path,
) -> None:
    client = FakeFinancialClient()
    outputs = download_financial_statements(
        client,
        ["000001.SZ"],
        start_date="20200101",
        end_date="20201231",
        raw_directory=tmp_path,
        pause_seconds=0,
    )

    assert len(client.calls) == 2
    assert outputs["income"].is_dir()
    assert outputs["balancesheet"].is_dir()
    assert outputs["manifest"].is_file()
    assert (tmp_path / "income" / "ts_code=000001.SZ.parquet").is_file()
    assert (tmp_path / "balancesheet" / "ts_code=000001.SZ.parquet").is_file()

    client.calls.clear()
    download_financial_statements(
        client,
        ["000001.SZ"],
        start_date="20200101",
        end_date="20201231",
        raw_directory=tmp_path,
        pause_seconds=0,
    )
    assert client.calls == []

    download_financial_statements(
        client,
        ["000001.SZ"],
        start_date="20190101",
        end_date="20211231",
        raw_directory=tmp_path,
        pause_seconds=0,
    )
    requested_ranges = {
        (call["dataset"], call["start_date"], call["end_date"])
        for call in client.calls
    }
    assert requested_ranges == {
        ("income", "20190101", "20191231"),
        ("income", "20210101", "20211231"),
        ("balancesheet", "20190101", "20191231"),
        ("balancesheet", "20210101", "20211231"),
    }


def test_empty_financial_response_is_checkpointed(tmp_path: Path) -> None:
    client = FakeFinancialClient(empty=True)
    download_financial_statements(
        client,
        ["000001.SZ"],
        start_date="20200101",
        end_date="20201231",
        raw_directory=tmp_path,
        pause_seconds=0,
    )
    assert len(client.calls) == 2

    client.calls.clear()
    download_financial_statements(
        client,
        ["000001.SZ"],
        start_date="20200101",
        end_date="20201231",
        raw_directory=tmp_path,
        pause_seconds=0,
    )
    assert client.calls == []


def test_legacy_schema_is_migrated_and_empty_response_is_checkpointed(
    tmp_path: Path,
) -> None:
    common = {
        "ts_code": "000001.SZ",
        "ann_date": "20200630",
        "f_ann_date": "20200630",
        "end_date": "20191231",
        "report_type": "1",
        "comp_type": "1",
        "update_flag": "0",
    }
    pd.DataFrame([{**common, "revenue": 10.0, "n_income_attr_p": 2.0}]).to_parquet(
        tmp_path / "income.parquet", index=False
    )
    pd.DataFrame(
        [
            {
                **common,
                "total_hldr_eqy_exc_min_int": 40.0,
                "total_hldr_eqy_inc_min_int": 45.0,
            }
        ]
    ).to_parquet(tmp_path / "balancesheet.parquet", index=False)

    client = FakeFinancialClient(empty=True)
    download_financial_statements(
        client,
        ["000001.SZ"],
        start_date="20200101",
        end_date="20201231",
        raw_directory=tmp_path,
        pause_seconds=0,
    )
    assert len(client.calls) == 2
    assert "oper_cost" in pd.read_parquet(
        tmp_path / "income" / "ts_code=000001.SZ.parquet"
    ).columns
    assert "total_assets" in pd.read_parquet(
        tmp_path / "balancesheet" / "ts_code=000001.SZ.parquet"
    ).columns

    client.calls.clear()
    download_financial_statements(
        client,
        ["000001.SZ"],
        start_date="20200101",
        end_date="20201231",
        raw_directory=tmp_path,
        pause_seconds=0,
    )
    assert client.calls == []


def test_financial_merge_keeps_versions_and_prefers_new_update() -> None:
    existing = pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "ann_date": "20200430",
                "f_ann_date": "20200430",
                "end_date": "20191231",
                "report_type": "1",
                "update_flag": "0",
                "revenue": 10.0,
            }
        ]
    )
    downloaded = pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "ann_date": "20200430",
                "f_ann_date": "20200430",
                "end_date": "20191231",
                "report_type": "1",
                "update_flag": "1",
                "revenue": 12.0,
            },
            {
                "ts_code": "000001.SZ",
                "ann_date": "20200515",
                "f_ann_date": "20200515",
                "end_date": "20191231",
                "report_type": "1",
                "update_flag": "1",
                "revenue": 13.0,
            },
        ]
    )

    merged = _merge_financial_records(existing, downloaded)
    assert len(merged) == 2
    assert merged.loc[merged["ann_date"].eq("20200430"), "revenue"].item() == 12.0
    assert merged.loc[merged["ann_date"].eq("20200515"), "revenue"].item() == 13.0


def test_financial_partition_reader_keeps_legacy_compatibility(
    tmp_path: Path,
) -> None:
    legacy = pd.DataFrame(
        [{"ts_code": "000001.SZ", "ann_date": "20200101", "value": 1.0}]
    )
    legacy_path = tmp_path / "income.parquet"
    legacy.to_parquet(legacy_path, index=False)
    partition_directory = tmp_path / "income"
    partition_directory.mkdir()
    pd.DataFrame(
        [
            {"ts_code": "000001.SZ", "ann_date": "20200101", "value": 1.0},
            {"ts_code": "000001.SZ", "ann_date": "20210101", "value": 2.0},
        ]
    ).to_parquet(
        partition_directory / "ts_code=000001.SZ.parquet", index=False
    )

    combined = read_financial_partitions(
        partition_directory, legacy_path=legacy_path
    )
    assert len(combined) == 2
    assert combined["ann_date"].tolist() == ["20200101", "20210101"]
