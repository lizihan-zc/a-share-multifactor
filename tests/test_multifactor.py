"""多因子测试重点：手算权重、收益可得时点、缺失值与持仓的因果顺序。"""

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from src.multifactor import (
    build_quantile_holdings,
    calculate_holdings_returns,
    calculate_rolling_ic_weights,
    calculate_target_turnover,
    combine_factor_scores,
)


def make_ic_history():
    dates = pd.date_range("2020-01-31", periods=7, freq="ME")
    ic = pd.DataFrame({
        "a": [0.2, 0.4, -0.2, 0.6, 0.8, 0.1, 0.3],
        "b": [-0.1, -0.3, 0.4, 0.2, -0.1, 0.3, -0.2],
    }, index=dates)
    available = pd.Series(dates, index=dates).shift(-1)
    return dates, ic, available


def test_rolling_weights_use_realized_ic_and_roll_out_old_months():
    dates, ic, available = make_ic_history()
    weights, audit = calculate_rolling_ic_weights(ic, available, dates, window=2)

    assert weights.iloc[:2].isna().all().all()
    # 三月收盘后，一月和二月的持有期收益已实现，平均 IC 为 0.3、-0.2。
    assert weights.iloc[2].tolist() == pytest.approx([0.6, -0.4])
    # 四月只使用二月和三月，旧的一月已经滚出窗口。
    assert weights.iloc[3].tolist() == pytest.approx([2 / 3, 1 / 3])
    assert weights.dropna().abs().sum(axis=1).tolist() == pytest.approx([1] * 5)
    assert audit.loc[dates[2], "history_start"] == dates[0]
    assert audit.loc[dates[2], "history_end"] == dates[1]
    assert audit.loc[dates[2], "latest_available_date"] == dates[2]
    assert audit.loc[dates[2], "min_valid_ic_months"] == 2


def test_future_and_current_unrealized_ic_cannot_change_past_weights():
    dates, ic, available = make_ic_history()
    expected, _ = calculate_rolling_ic_weights(ic, available, dates, window=2)
    changed = ic.copy()
    # 三月信号的未来收益要到四月才知道，连三月自己的 IC 也不能用于三月权重。
    changed.loc[dates[2]:] = [[9, -8]] * len(changed.loc[dates[2]:])
    actual, _ = calculate_rolling_ic_weights(changed, available, dates, window=2)
    assert_frame_equal(actual.iloc[:3], expected.iloc[:3])
    assert not actual.iloc[3:].equals(expected.iloc[3:])


def test_explicit_ic_available_date_handles_delayed_labels():
    dates, ic, available = make_ic_history()
    available.iloc[1] = dates[3]
    weights, audit = calculate_rolling_ic_weights(ic, available, dates, window=2)
    assert weights.iloc[2].isna().all()
    assert audit.loc[dates[2], "n_periods"] == 1
    assert weights.iloc[3].notna().all()


def test_missing_ic_month_stays_in_window_and_zero_means_stay_missing():
    dates, ic, available = make_ic_history()
    ic.iloc[1, 0] = np.nan
    weights, _ = calculate_rolling_ic_weights(ic, available, dates, window=2)
    assert weights.iloc[2:4].isna().all().all()
    relaxed, _ = calculate_rolling_ic_weights(
        ic, available, dates, window=2, min_history=1
    )
    assert relaxed.iloc[2].tolist() == pytest.approx([0.5, -0.5])
    # 正负 IC 恰好抵消时，不能凭空构造一组权重。
    zero_ic = ic.copy()
    zero_ic.iloc[:2] = [[0.2, -0.1], [-0.2, 0.1]]
    zero_weights, audit = calculate_rolling_ic_weights(
        zero_ic, available, dates, window=2
    )
    assert zero_weights.iloc[2].isna().all()
    assert not audit.loc[dates[2], "is_ready"]


def test_window_counts_calendar_months_instead_of_skipping_nan_months():
    dates, ic, available = make_ic_history()
    ic.iloc[2] = np.nan
    weights, audit = calculate_rolling_ic_weights(ic, available, dates, window=2)
    # 即使更早有足够有效 IC，也不能跳过三月向前补成两个有效观测。
    assert weights.iloc[3].isna().all()
    assert audit.loc[dates[3], "history_start"] == dates[1]
    assert audit.loc[dates[3], "n_periods"] == 2


def test_rejects_ic_available_before_its_return_horizon():
    dates, ic, available = make_ic_history()
    available.iloc[0] = dates[0]
    with pytest.raises(ValueError, match="IC 可用日"):
        calculate_rolling_ic_weights(ic, available, dates, window=2)


def test_equal_and_weighted_scores_align_rows_and_do_not_skip_missing_factors():
    dates = pd.to_datetime(["2020-02-28", "2020-01-31"])
    panel = pd.DataFrame({
        "date": [dates[0], dates[1], dates[0], dates[0], dates[0]],
        "is_eligible": [True, True, True, False, True],
        "a": [2.0, 4.0, np.nan, 3.0, 1.0],
        "b": [4.0, 2.0, 1.0, 1.0, 2.0],
    }, index=[11, 5, 9, 8, 3])
    equal = combine_factor_scores(panel, ["a", "b"])
    assert equal.iloc[:2].tolist() == pytest.approx([3, 3])
    assert equal.iloc[2:4].isna().all()
    weights = pd.DataFrame({"a": [0.75, 1.0], "b": [-0.25, 0.0]}, index=dates)
    weighted = combine_factor_scores(panel, ["a", "b"], weights, score_name="ic")
    assert weighted.iloc[:2].tolist() == pytest.approx([0.5, 4])
    assert weighted.index.equals(panel.index)
    # 即使缺失因子恰好是零权重，也坚持相同的完整因子样本定义。
    weights.loc[dates[0]] = [0, 1]
    assert np.isnan(combine_factor_scores(panel, ["a", "b"], weights).iloc[2])
    missing_month = weights.drop(index=dates[1])
    assert np.isnan(combine_factor_scores(panel, ["a", "b"], missing_month).iloc[1])


def make_score_panel():
    rows = []
    for date in pd.date_range("2020-01-31", periods=3, freq="ME"):
        for stock in range(10):
            rows.append({
                "date": date,
                "stock_code": f"S{stock:02d}",
                "is_eligible": True,
                "a": float(stock),
                "b": float(stock),
                "future_return_1m": stock / 100,
                "is_in_label_sample": True,
            })
    panel = pd.DataFrame(rows)
    panel["score"] = combine_factor_scores(panel, ["a", "b"])
    return panel


def test_signals_and_holdings_are_independent_of_future_label_availability():
    panel = make_score_panel()
    expected = build_quantile_holdings(panel, ["score"], min_observations=5)
    changed = panel.copy()
    changed.loc[changed.date.eq(changed.date.max()), "is_in_label_sample"] = False
    changed.loc[changed.stock_code.eq("S09"), "future_return_1m"] = np.nan
    assert_series_equal(
        combine_factor_scores(changed, ["a", "b"]),
        combine_factor_scores(panel, ["a", "b"]),
    )
    actual = build_quantile_holdings(changed, ["score"], min_observations=5)
    assert_frame_equal(actual, expected)
    # 缺失收益的 S09 仍在最高组，S07 不会事后被补进来。
    assert set(actual.loc[actual.quantile_number.eq(5), "stock_code"]) == {"S08", "S09"}
    returns = calculate_holdings_returns(actual, changed)
    first_top = returns.loc[returns.date.eq(panel.date.min()) & returns.quantile_number.eq(5)].iloc[0]
    assert first_top.portfolio_return == pytest.approx(0.08)
    assert first_top.label_coverage == pytest.approx(0.5)
    assert first_top.n_stocks == 2
    assert first_top.n_observed == 1
    assert returns.loc[returns.date.eq(panel.date.max()), "portfolio_return"].isna().all()


def test_quantiles_are_equal_weight_and_ties_have_stable_stock_order():
    panel = make_score_panel()
    panel["score"] = 1.0
    holdings = build_quantile_holdings(
        panel.sample(frac=1, random_state=7), ["score"], min_observations=5
    )
    assert holdings.groupby(["date", "quantile_number"]).weight.sum().tolist() == pytest.approx([1] * 15)
    assert set(holdings.loc[holdings.quantile_number.eq(5), "stock_code"]) == {"S08", "S09"}


def test_target_turnover_hand_calculation_and_monthly_gaps():
    dates = pd.date_range("2020-01-31", periods=5, freq="ME")
    holdings = pd.DataFrame({
        "date": [dates[0], dates[0], dates[1], dates[1], dates[3], dates[4]],
        "stock_code": ["A", "B", "B", "C", "A", "D"],
        "factor": ["score"] * 6,
        "quantile_number": [5] * 6,
        "weight": [0.5, 0.5, 0.5, 0.5, 1.0, 1.0],
    })
    result = calculate_target_turnover(holdings, dates).set_index("date")
    assert np.isnan(result.loc[dates[0], "target_turnover"])
    assert result.loc[dates[1], "target_turnover"] == pytest.approx(0.5)
    assert result.loc[dates[2:4], "target_turnover"].isna().all()
    assert result.loc[dates[4], "target_turnover"] == pytest.approx(1.0)
    unchanged = make_score_panel()
    same_holdings = build_quantile_holdings(unchanged, ["score"], min_observations=5)
    same_turnover = calculate_target_turnover(same_holdings, pd.DatetimeIndex(unchanged.date.unique()))
    assert same_turnover.target_turnover.iloc[1:].tolist() == [0.0, 0.0]
