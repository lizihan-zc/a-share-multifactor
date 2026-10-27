"""账户现金守恒、执行时点、价格漂移及约束的手算案例。"""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from src.portfolio.backtest import (
    MarketData, build_portfolio_targets, is_opening_suspension,
    run_portfolio_backtest, summarize_backtest,
)


def market(opening=None, close=None, factors=None):
    opening = np.array(opening if opening is not None else [[10., 10.]] * 5)
    close = np.array(close if close is not None else opening, dtype=float)
    factors = np.ones_like(opening) if factors is None else np.array(factors)
    dates = pd.date_range('2024-01-01', periods=len(opening), freq='B')
    return MarketData(
        dates, pd.Index(['A', 'B'], name='stock_code'), opening,
        opening * factors, close * factors,
        pd.DataFrame(close * factors).ffill().to_numpy(),
        np.full_like(opening, 100.), np.full_like(opening, 1.),
        np.full_like(opening, 1000.), np.zeros_like(opening, dtype=bool),
        np.ones(len(opening), dtype=bool), np.full(2, np.datetime64('NaT'), dtype='datetime64[ns]'),
        dates[[0, 2, -1]],
    )


def targets(m, rows, positions):
    return pd.DataFrame(rows, index=m.dates[positions], columns=m.stocks)


def test_next_open_and_overnight_gap_go_to_old_holdings():
    m = market([[10.,10.],[10.,10.],[20.,10.],[20.,10.]],
               [[10.,10.],[10.,10.],[20.,20.],[20.,20.]])
    r = run_portfolio_backtest(m, targets(m, [[1,0],[0,1]], [0,1]), initial_cash=100)
    assert r.daily.index.min() == m.dates[1]
    assert r.daily.nav.iloc[0] == pytest.approx(100)
    assert r.daily.nav.iloc[1] == pytest.approx(400)  # 旧A开盘翻倍，新B日内翻倍
    assert r.trades.loc[r.trades.date.eq(m.dates[2]) & r.trades.side.eq('sell'), 'filled_notional'].sum() == 200


def test_drift_turnover_and_adjusted_units():
    m = market(close=[[10.,10.],[20.,10.],[20.,10.],[20.,10.],[20.,10.]],
               opening=[[10.,10.],[10.,10.],[20.,10.],[20.,10.],[20.,10.]])
    r = run_portfolio_backtest(m, targets(m, [[.5,.5],[0,1]], [0,1]), initial_cash=100)
    assert r.daily.nav.iloc[0] == 150
    assert r.daily.one_way_turnover.iloc[1] == pytest.approx(200 / 150 / 2)
    # 复权因子翻倍、原始价格减半，不产生损益。
    m2 = market([[10.,10.],[10.,10.],[5.,10.],[5.,10.]],
                factors=[[1,1],[1,1],[2,1],[2,1]])
    r2 = run_portfolio_backtest(m2, targets(m2, [[1,0]], [0]), initial_cash=100)
    np.testing.assert_allclose(r2.daily.nav, 100)


def test_fees_cash_conservation_and_first_day_drawdown():
    m = market()
    r = run_portfolio_backtest(m, targets(m, [[1,0],[0,1]], [0,1]), initial_cash=100, cost_bps=100)
    assert r.daily.nav.iloc[0] == pytest.approx(100 / 1.01)
    assert r.daily.nav.iloc[-1] + r.trades.cost.sum() == pytest.approx(100)
    assert (r.daily.cash_weight >= 0).all()
    assert summarize_backtest(r, 100)['Max Drawdown'] == pytest.approx(r.daily.nav.iloc[-1] / 100 - 1)
    assert r.daily.cost.sum() == pytest.approx((r.trades.filled_notional * .01).sum())


def test_limit_up_cancels_buy_and_limit_down_retries_sell():
    m = market()
    m.up_limit[1,0] = 10
    no_buy = run_portfolio_backtest(m, targets(m, [[1,0]], [0]), initial_cash=100, enforce_limits=True)
    assert (no_buy.daily.nav == 100).all()
    assert no_buy.trades.reason.tolist() == ['limit_up']  # 买单不延后重试
    m.up_limit[1,0] = 100
    m.down_limit[2,0] = 10
    r = run_portfolio_backtest(m, targets(m, [[1,0],[0,1]], [0,1]), initial_cash=100, enforce_limits=True)
    blocked = r.trades.loc[r.trades.reason.eq('limit_down')]
    assert len(blocked) == 1 and blocked.filled_notional.iloc[0] == 0
    delayed = r.trades.loc[(r.trades.side == 'sell') & (r.trades.filled_notional > 0)]
    assert delayed.date.tolist() == [m.dates[3]]
    # 受阻卖单释放的现金不追买已取消的B买单。
    assert r.daily.cash_weight.iloc[-1] == 1


def test_capacity_uses_previous_day_adv_and_cash_is_not_redistributed():
    m = market()
    m.adv20[0,0] = 100
    m.adv20[1:,0] = 100_000  # 执行日收盘才知道，不能用于当天交易
    r = run_portfolio_backtest(m, targets(m, [[1,0]], [0]), initial_cash=100, max_adv_participation=.1)
    assert r.trades.filled_notional.sum() == 10
    assert r.daily.cash_weight.iloc[0] == .9
    assert r.trades.reason.tolist() == ['adv_cap']


def test_missing_prices_are_stale_marked_then_delisting_written_off():
    m = market()
    m.open[2:,0] = np.nan
    m.adjusted_open[2:,0] = np.nan
    m.close[2:,0] = np.nan
    m.mark[2:,0] = 10
    m.delist_dates[0] = m.dates[4].to_datetime64()
    r = run_portfolio_backtest(m, targets(m, [[.5,.5],[0,1]], [0,1]), initial_cash=100)
    assert r.daily.stale_weight.iloc[1] == .5
    assert r.daily.nav.iloc[-1] == 50
    assert r.daily.writeoff_value.sum() == 50
    assert r.daily.writeoff_count.sum() == 1
    assert r.daily.max_stale_days.max() == 2


def test_missing_suspension_or_limit_data_blocks_trade():
    m = market()
    m.suspension_covered[1] = False
    r = run_portfolio_backtest(m, targets(m, [[1,0]], [0]), initial_cash=100)
    assert r.trades.filled_notional.sum() == 0
    assert r.trades.reason.iloc[0] == 'suspended_or_uncovered'
    m.suspension_covered[1] = True
    m.up_limit[1,0] = np.nan
    r = run_portfolio_backtest(m, targets(m, [[1,0]], [0]), initial_cash=100, enforce_limits=True)
    assert r.trades.reason.iloc[0] == 'missing_limit'


def test_only_opening_suspension_blocks():
    assert is_opening_suspension(None)
    assert is_opening_suspension('09:30-10:00')
    assert not is_opening_suspension('09:31-10:01')
    assert not is_opening_suspension('13:00-15:00')


def test_signal_independence_from_labels_and_future_prices():
    m = market()
    m = replace(m, dates=pd.DatetimeIndex(['2024-01-31','2024-02-01','2024-02-29','2024-03-01','2024-03-29']))
    panel = pd.DataFrame({'date': np.repeat(m.dates[[0,2,4]], 2), 'stock_code':['A','B'] * 3,
                          'is_eligible':True, 'score':[1.,1.,2.,1.,1.,2.], 'future_return_1m':np.nan})
    original = build_portfolio_targets(panel, m, 'score', top_fraction=.5)
    panel['future_return_1m'] = [1,999,2,999,999,999]
    pd.testing.assert_frame_equal(original, build_portfolio_targets(panel, m, 'score', top_fraction=.5))
    assert original.iloc[0].tolist() == [1,0]  # 同分按代码稳定排序
    sparse = build_portfolio_targets(panel, m, 'score', top_fraction=.5, rebalance_every=2)
    assert sparse.index.tolist() == [m.dates[0], m.dates[4]]
    r = run_portfolio_backtest(m, original, initial_cash=100)
    assert r.pending_signals.tolist() == [m.dates[-1]]
    # 扰动第二次执行及之后的价格不影响此前的账户。
    changed_open = m.adjusted_open.copy(); changed_open[3:] *= 2
    changed_close = m.close.copy(); changed_close[3:] *= 2
    changed_mark = m.mark.copy(); changed_mark[3:] *= 2
    changed = replace(m, adjusted_open=changed_open, close=changed_close, mark=changed_mark)
    other = run_portfolio_backtest(changed, original, initial_cash=100)
    pd.testing.assert_frame_equal(r.daily.iloc[:2], other.daily.iloc[:2])


def test_invalid_weights_and_missing_month_are_rejected():
    m = market()
    with pytest.raises(ValueError):
        run_portfolio_backtest(m, targets(m, [[1.1,0]], [0]))
    panel = pd.DataFrame({'date':pd.to_datetime(['2024-01-01','2024-03-01']),
                          'stock_code':['A','A'], 'is_eligible':True, 'score':1.})
    with pytest.raises(ValueError, match='完整月历'):
        build_portfolio_targets(panel, m, 'score')


def test_full_liquidation_does_not_leave_roundoff_positions():
    m = market(opening=[[13.27,10.]] * 5)
    m.delist_dates[0] = m.dates[4].to_datetime64()
    r = run_portfolio_backtest(m, targets(m, [[1,0],[0,1]], [0,1]), initial_cash=10_000_000)
    assert r.daily.n_stocks.eq(1).all()
    assert r.daily.writeoff_count.sum() == 0
    assert r.writeoffs.empty
