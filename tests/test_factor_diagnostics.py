"""Notebook 04 单因子诊断函数的单元测试。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.factors import (
    calculate_average_factor_correlation,
    calculate_factor_rank_autocorrelation,
    calculate_monthly_rank_ic,
    summarize_rank_ic,
)


def make_diagnostic_panel() -> pd.DataFrame:
    rows = []
    dates = pd.to_datetime(["2020-01-31", "2020-02-28", "2020-03-31"])
    for month, date in enumerate(dates):
        for stock in range(1, 6):
            rows.append(
                {
                    "date": date,
                    "stock_code": f"S{stock}",
                    "good": stock + month * 0.01,
                    "bad": -stock,
                    "same_information": 2 * stock,
                    "future_return_1m": stock / 100,
                }
            )
    return pd.DataFrame(rows)


def test_monthly_rank_ic_and_summary() -> None:
    panel = make_diagnostic_panel()
    monthly = calculate_monthly_rank_ic(
        panel, ["good", "bad"], min_observations=5
    )

    assert monthly["good"].tolist() == pytest.approx([1.0, 1.0, 1.0])
    assert monthly["bad"].tolist() == pytest.approx([-1.0, -1.0, -1.0])

    varying = pd.DataFrame(
        {"factor": [0.10, -0.05, 0.20]},
        index=pd.date_range("2020-01-31", periods=3, freq="ME"),
    )
    summary = summarize_rank_ic(varying)
    expected_mean = varying["factor"].mean()
    expected_std = varying["factor"].std(ddof=1)
    assert summary.loc["factor", "Mean Rank IC"] == pytest.approx(expected_mean)
    assert summary.loc["factor", "IC Std"] == pytest.approx(expected_std)
    assert summary.loc["factor", "ICIR"] == pytest.approx(
        expected_mean / expected_std
    )
    assert summary.loc["factor", "Positive IC %"] == pytest.approx(2 / 3)
    assert summary.loc["factor", "IC t-stat"] == pytest.approx(
        expected_mean / (expected_std / np.sqrt(3))
    )


def test_rank_ic_uses_pairwise_complete_observations() -> None:
    panel = make_diagnostic_panel()
    panel.loc[
        panel["stock_code"].eq("S5") & panel["date"].eq(panel["date"].min()),
        "good",
    ] = np.nan

    monthly = calculate_monthly_rank_ic(
        panel, ["good"], min_observations=5
    )
    assert np.isnan(monthly.iloc[0, 0])
    assert monthly.iloc[1:, 0].tolist() == pytest.approx([1.0, 1.0])


def test_average_factor_correlation_is_monthly_then_time_averaged() -> None:
    panel = make_diagnostic_panel()
    correlation = calculate_average_factor_correlation(
        panel,
        ["good", "bad", "same_information"],
        min_observations=5,
    )

    assert correlation.loc["good", "same_information"] == pytest.approx(1.0)
    assert correlation.loc["good", "bad"] == pytest.approx(-1.0)
    assert correlation.equals(correlation.T)


def test_factor_rank_autocorrelation_uses_common_stocks() -> None:
    panel = make_diagnostic_panel()
    second_date = panel["date"].sort_values().unique()[1]
    panel = panel.loc[
        ~(panel["date"].eq(second_date) & panel["stock_code"].eq("S5"))
    ]
    autocorrelation = calculate_factor_rank_autocorrelation(
        panel, ["good", "bad"], min_observations=4
    )

    assert len(autocorrelation) == 4
    assert autocorrelation["rank_autocorrelation"].tolist() == pytest.approx(
        [1.0, 1.0, 1.0, 1.0]
    )
    assert autocorrelation["n_stocks"].tolist() == [4, 4, 4, 4]
