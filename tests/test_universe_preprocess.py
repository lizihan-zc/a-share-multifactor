"""Notebook 01 月度股票池财务适用性规则测试。"""

from __future__ import annotations

import pandas as pd

from src.data.preprocess import UniverseConfig, build_monthly_universe


def test_gp_inputs_do_not_control_whole_universe_eligibility() -> None:
    """验证 GP 不适用或数据缺失不会排除其他因子仍可用的股票。"""

    date = pd.Timestamp("2020-01-31")
    stock_codes = ["000001.SZ", "000002.SZ"]
    price_daily = pd.DataFrame(
        {
            "date": [date, date],
            "stock_code": stock_codes,
            "amount": [1_000_000.0, 1_000_000.0],
            "market_cap": [50_000_000.0, 60_000_000.0],
            "is_one_price_limit_up": [False, False],
        }
    )
    calendar = pd.DataFrame({"cal_date": ["20200131"]})
    stock_basic = pd.DataFrame(
        {
            "ts_code": stock_codes,
            "list_date": ["20100101", "20100101"],
            "delist_date": [pd.NA, pd.NA],
        }
    )
    financial_snapshots = pd.DataFrame(
        {
            "date": [date, date],
            "stock_code": stock_codes,
            "financial_available_date": [date, date],
            "end_date": pd.to_datetime(["2019-12-31", "2019-12-31"]),
            "revenue_ttm": [pd.NA, pd.NA],
            "net_profit_ttm": [10.0, 12.0],
            "gross_profit_ttm": [pd.NA, pd.NA],
            "total_assets": [100.0, 120.0],
            "total_equity": [50.0, 60.0],
        }
    )
    industry_membership = pd.DataFrame(
        {
            "ts_code": stock_codes,
            "l1_name": ["银行", "机械设备"],
            "in_date": ["20100101", "20100101"],
            "out_date": [pd.NA, pd.NA],
        }
    )

    result = build_monthly_universe(
        price_daily,
        calendar,
        stock_basic,
        financial_snapshots,
        industry_membership=industry_membership,
        config=UniverseConfig(
            study_start="20200101",
            study_end="20200131",
            min_listing_trading_days=1,
        ),
    ).set_index("stock_code")

    assert not result.loc["000001.SZ", "is_gross_profit_applicable"]
    assert result.loc["000002.SZ", "is_gross_profit_applicable"]
    assert not result["has_gp_factor_data"].any()
    assert result["has_core_data"].all()
    assert result["is_eligible"].all()
