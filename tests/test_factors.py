"""Notebook 03 因子计算函数的单元测试。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.factors import (
    calculate_amihud_illiquidity,
    calculate_book_to_price,
    calculate_earnings_to_price,
    calculate_gross_profitability,
    calculate_low_volatility,
    calculate_momentum_12_1,
    calculate_momentum_3m,
    calculate_roe,
    calculate_size,
    preprocess_factor_panel,
)


def make_daily(periods: int = 300) -> pd.DataFrame:
    """构造两只股票具有固定日收益率的合成日频行情。"""

    dates = pd.bdate_range("2020-01-01", periods=periods)
    parts = []
    for stock_code, daily_return in (("A", 0.01), ("B", -0.005)):
        step = np.arange(periods)
        parts.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "stock_code": stock_code,
                    "close": 100 * (1 + daily_return) ** step,
                    "adj_factor": 2.0,
                    "amount": 1_000_000.0,
                }
            )
        )
    return pd.concat(parts, ignore_index=True)


def test_accounting_ratios_and_size() -> None:
    """验证 EP、BP、GP、SIZE 公式以及非法分母的缺失值处理。"""

    panel = pd.DataFrame(
        {
            "net_profit_ttm": [10.0, -2.0, 1.0],
            "gross_profit_ttm": [30.0, 4.0, 1.0],
            "total_assets": [200.0, 50.0, 0.0],
            "total_equity": [100.0, -20.0, 10.0],
            "market_cap": [500.0, 40.0, 0.0],
        }
    )

    assert calculate_earnings_to_price(panel).iloc[:2].tolist() == [0.02, -0.05]
    assert calculate_book_to_price(panel).iloc[:2].tolist() == [0.2, -0.5]
    assert calculate_gross_profitability(panel).iloc[:2].tolist() == [0.15, 0.08]
    assert calculate_size(panel).iloc[0] == pytest.approx(np.log(500))
    assert calculate_earnings_to_price(panel).iloc[2:].isna().all()
    assert calculate_gross_profitability(panel).iloc[2:].isna().all()
    assert calculate_size(panel).iloc[2:].isna().all()


def test_roe_uses_prior_year_same_period_equity() -> None:
    """验证 ROE 使用当前权益与上年同期权益的平均值。"""

    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-04-30", "2021-04-30"]),
            "stock_code": ["A", "A"],
            "end_date": pd.to_datetime(["2020-03-31", "2021-03-31"]),
            "total_equity": [100.0, 120.0],
            "net_profit_ttm": [10.0, 22.0],
        }
    )

    result = calculate_roe(panel)
    assert np.isnan(result.iloc[0])
    assert result.iloc[1] == pytest.approx(22 / 110)


def test_roe_can_use_explicit_average_equity() -> None:
    """验证 ROE 可以直接使用调用方提供的平均权益字段。"""

    panel = pd.DataFrame(
        {"net_profit_ttm": [10.0, 2.0], "average_equity": [50.0, -1.0]}
    )
    result = calculate_roe(panel, average_equity_column="average_equity")
    assert result.iloc[0] == 0.2
    assert np.isnan(result.iloc[1])


def test_momentum_formulas_use_adjusted_prices() -> None:
    """验证两类动量因子的复权价格公式与交易日窗口。"""

    daily = make_daily()
    last_date = daily["date"].max()
    index = pd.DataFrame(
        {"date": [last_date, last_date], "stock_code": ["A", "B"]}
    )

    mom_12_1 = calculate_momentum_12_1(daily, index)
    mom_3m = calculate_momentum_3m(daily, index)
    assert mom_12_1.iloc[0] == pytest.approx(1.01 ** (252 - 21) - 1)
    assert mom_12_1.iloc[1] == pytest.approx(0.995 ** (252 - 21) - 1)
    assert mom_3m.iloc[0] == pytest.approx(1.01**63 - 1)
    assert mom_3m.iloc[1] == pytest.approx(0.995**63 - 1)


def test_exact_session_momentum_does_not_bridge_missing_quote() -> None:
    """验证动量因子遇到精确回看日行情缺失时不会跨期取值。"""

    daily = make_daily(periods=80)
    dates = pd.Index(daily["date"].drop_duplicates().sort_values())
    missing_key = (daily["stock_code"].eq("A")) & daily["date"].eq(dates[-64])
    daily = daily.loc[~missing_key]
    index = pd.DataFrame({"date": [dates[-1]], "stock_code": ["A"]})

    assert calculate_momentum_3m(daily, index).isna().all()


def test_low_volatility_and_amihud() -> None:
    """验证低波动和 Amihud 非流动性因子的滚动计算结果。"""

    daily = make_daily(periods=80)
    last_date = daily["date"].max()
    index = pd.DataFrame(
        {"date": [last_date, last_date], "stock_code": ["A", "B"]}
    )

    lowvol = calculate_low_volatility(daily, index, window=20)
    illiquidity = calculate_amihud_illiquidity(daily, index, window=20)
    assert lowvol.abs().max() < 1e-12
    assert illiquidity.iloc[0] == pytest.approx(0.01 / 1_000_000)
    assert illiquidity.iloc[1] == pytest.approx(0.005 / 1_000_000)


def test_rolling_factors_reject_market_session_gaps() -> None:
    """验证完整滚动窗口不会跨越股票行情中的交易日缺口。"""

    daily = make_daily(periods=80)
    dates = pd.Index(daily["date"].drop_duplicates().sort_values())
    missing_key = (daily["stock_code"].eq("A")) & daily["date"].eq(dates[-10])
    daily = daily.loc[~missing_key]
    index = pd.DataFrame({"date": [dates[-1]], "stock_code": ["A"]})

    assert calculate_low_volatility(daily, index, window=20).isna().all()
    assert calculate_amihud_illiquidity(daily, index, window=20).isna().all()


def test_preprocessing_is_cross_sectional_and_direction_aware() -> None:
    """验证预处理按日期分组，并按指定方向完成横截面标准化。"""

    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-31"] * 3 + ["2020-02-28"] * 3),
            "factor": [1.0, 2.0, 100.0, 10.0, 20.0, 30.0],
        }
    )
    processed = preprocess_factor_panel(
        panel,
        ["factor"],
        directions={"factor": -1},
        lower_quantile=0,
        upper_quantile=1,
    )

    by_date = processed.groupby("date")["factor"]
    assert np.allclose(by_date.mean(), 0)
    assert np.allclose(by_date.std(ddof=0), 1)
    assert processed.loc[0, "factor"] > processed.loc[2, "factor"]
