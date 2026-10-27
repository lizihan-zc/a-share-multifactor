"""08/09：下一交易日开盘执行、逐日估值的多头现金账户研究回测。

交易以复权价格计价的连续单位模拟，不是整数股／逐笔撮合。费用按实际双边
成交额收取；没有足够现金时等比例缩减买单。所有历史均值仅使用当时已知数据。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import pandas as pd


@dataclass
class MarketData:
    dates: pd.DatetimeIndex
    stocks: pd.Index
    open: np.ndarray                  # 原始开盘价，用于涨跌停判断
    adjusted_open: np.ndarray
    close: np.ndarray                 # 当天真实可观察的复权收盘价，未填充
    mark: np.ndarray                  # 只向前填充的复权收盘估值
    up_limit: np.ndarray
    down_limit: np.ndarray
    adv20: np.ndarray                 # 当日收盘已知的过去20个市场交易日成交额均值（元）
    suspended: np.ndarray             # 开盘时处于停牌，盘中才开始的不拦截
    suspension_covered: np.ndarray
    delist_dates: np.ndarray
    signal_dates: pd.DatetimeIndex


@dataclass
class BacktestResult:
    daily: pd.DataFrame
    trades: pd.DataFrame              # 包括零成交的受阻请求
    holdings: pd.DataFrame            # 每个月首个执行日收盘的实际持仓
    pending_signals: pd.DatetimeIndex
    writeoffs: pd.DataFrame           # 逐只股票核销审计，不伪造成卖出


def is_opening_suspension(timing) -> bool:
    """空时间按全天停牌；只检查包含09:30的区间，未知格式保守拦截。"""
    if pd.isna(timing) or not str(timing).strip():
        return True
    intervals = re.findall(r'(\d{2}:\d{2})\s*-\s*(\d{2}:\d{2})', str(timing))
    if not intervals:
        return True
    return any(start <= '09:30' <= end for start, end in intervals)


def load_market_data(project_root: Path, signal_panel: pd.DataFrame) -> MarketData:
    """读取本地缓存；交易日来自市场日历，不以单只股票的下一条记录替代。"""
    root = Path(project_root)
    raw = root / 'data' / 'raw'
    calendar = pd.read_parquet(raw / 'trade_calendar.parquet')
    dates = pd.DatetimeIndex(pd.to_datetime(
        calendar.loc[calendar.is_open.eq(1), 'cal_date'].astype(str), format='%Y%m%d'
    ).unique()).sort_values()
    signal_dates = pd.DatetimeIndex(sorted(signal_panel.date.unique()))
    # 预留至少20个交易日计算ADV；终点只到真实缓存终点。
    first = max(0, dates.searchsorted(signal_dates.min()) - 25)
    dates = dates[first:]
    stocks = pd.Index(sorted(signal_panel.stock_code.unique()), name='stock_code')
    cols = ['date', 'stock_code', 'open', 'close', 'adj_factor', 'amount',
            'up_limit_price', 'down_limit_price']
    prices = pd.read_parquet(root / 'data/processed/price_daily_clean.parquet',
                             columns=cols, filters=[('date', '>=', dates.min())])
    prices = prices.loc[prices.stock_code.isin(stocks)]
    if prices.duplicated(['date', 'stock_code']).any():
        raise ValueError('行情存在重复主键')
    if prices.date.max() < dates.max():
        raise ValueError('交易日历终点超过行情缓存，请补齐或显式截断日历')
    prices['adjusted_open'] = prices.open * prices.adj_factor
    prices['adjusted_close'] = prices.close * prices.adj_factor

    def matrix(col):
        return prices.pivot(index='date', columns='stock_code', values=col).reindex(
            index=dates, columns=stocks).to_numpy(dtype=float)

    opening = matrix('open')
    adjusted_open = matrix('adjusted_open')
    close = matrix('adjusted_close')
    mark = pd.DataFrame(close).where(lambda x: x.gt(0)).ffill().to_numpy()
    # 缺行情日成交额记0用于日历日均值；滚动窗口不跳过停牌日、不跨越未来。
    amount = pd.DataFrame(matrix('amount')).clip(lower=0).fillna(0)
    adv20 = amount.rolling(20, min_periods=20).mean().to_numpy()
    suspended = np.zeros(opening.shape, dtype=bool)
    covered = np.zeros(len(dates), dtype=bool)
    for i, day in enumerate(dates):
        path = raw / 'suspend_d' / f'trade_date={day:%Y%m%d}.parquet'
        if not path.exists():
            continue
        records = pd.read_parquet(path)
        covered[i] = True
        opening_stops = records.suspend_type.eq('S') & records.suspend_timing.map(is_opening_suspension)
        indices = stocks.get_indexer(records.loc[opening_stops, 'ts_code'])
        suspended[i, indices[indices >= 0]] = True
    basic = pd.read_parquet(raw / 'stock_basic.parquet')
    if basic.ts_code.duplicated().any():
        raise ValueError('stock_basic包含重复股票')
    delist = pd.to_datetime(basic.set_index('ts_code').delist_date, format='%Y%m%d', errors='coerce')
    return MarketData(dates, stocks, opening, adjusted_open, close, mark,
                      matrix('up_limit_price'), matrix('down_limit_price'), adv20,
                      suspended, covered, delist.reindex(stocks).to_numpy(), signal_dates)


def build_portfolio_targets(
    panel: pd.DataFrame, market: MarketData, score_column: str | None,
    *, top_fraction: float = 0.2, rebalance_every: int = 1,
    min_adv20: float = 0.0,
) -> pd.DataFrame:
    """只读取信号日评分与资格；None表示同一候选股票池等权基准。"""
    if not 0 < top_fraction <= 1 or rebalance_every not in (1, 2, 3):
        raise ValueError('top_fraction应在(0,1]，rebalance_every应为1/2/3')
    if not np.isfinite(min_adv20) or min_adv20 < 0:
        raise ValueError('min_adv20必须是非负有限数')
    if panel.duplicated(['date', 'stock_code']).any():
        raise ValueError('信号面板存在重复主键')
    dates = pd.DatetimeIndex(sorted(panel.date.unique()))
    if not dates.to_period('M').equals(pd.period_range(dates[0], dates[-1], freq='M')):
        raise ValueError('输入必须保留完整月历，不能跳过缺信号月份')
    output = pd.DataFrame(0.0, index=dates[::rebalance_every], columns=market.stocks)
    for date in output.index:
        cross = panel.loc[panel.date.eq(date) & panel.is_eligible.fillna(False)].copy()
        if score_column is not None:
            cross = cross.loc[np.isfinite(pd.to_numeric(cross[score_column], errors='coerce'))]
        if min_adv20 > 0:
            loc = market.dates.get_indexer([date])[0]
            if loc < 0:
                raise ValueError('信号日不在市场交易日历中')
            adv = pd.Series(market.adv20[loc], index=market.stocks)
            cross = cross.loc[cross.stock_code.map(adv).ge(min_adv20)]
        if cross.empty:
            # 明确的空目标，转现金；不把本月删除后跨月持有。
            continue
        if score_column is not None:
            cross = cross.sort_values([score_column, 'stock_code'], ascending=[False, True], kind='mergesort')
            cross = cross.head(max(1, int(np.ceil(len(cross) * top_fraction))))
        if not cross.stock_code.isin(market.stocks).all():
            raise ValueError('信号股票不在行情矩阵中')
        output.loc[date, cross.stock_code] = 1.0 / len(cross)
    output.index.name = 'signal_date'
    return output


def run_portfolio_backtest(
    market: MarketData, targets: pd.DataFrame, *, initial_cash: float = 10_000_000,
    cost_bps: float = 0.0, enforce_limits: bool = False,
    max_adv_participation: float | None = None,
) -> BacktestResult:
    """开盘先卖后买；受阻卖单逐日重试，未完成买单取消，剩余资金留现金。

    ADV容量上限使用前一交易日已知ADV20；目标金额以交易前NAV计算。费用预算
    会令买单略低于目标，不借款。缺少停牌分区时禁止两侧交易；约束版本缺少对应
    涨跌停价时也禁止该侧交易。退市日剩余仓位零回收核销是显式压力假设。
    """
    if not np.isfinite(initial_cash) or initial_cash <= 0:
        raise ValueError('initial_cash必须是正有限数')
    if not np.isfinite(cost_bps) or not 0 <= cost_bps < 10_000:
        raise ValueError('cost_bps应在[0,10000)')
    if max_adv_participation is not None and not 0 < max_adv_participation <= 1:
        raise ValueError('ADV比例应在(0,1]')
    targets = targets.sort_index().reindex(columns=market.stocks)
    if targets.empty or targets.index.has_duplicates or targets.isna().any().any():
        raise ValueError('目标为空、日期重复或股票列缺失')
    w = targets.to_numpy(dtype=float)
    if not np.isfinite(w).all() or (w < 0).any() or (w.sum(axis=1) > 1 + 1e-10).any():
        raise ValueError('仅支持非负、权重和不超过1的目标')
    if not targets.index.isin(market.dates).all():
        raise ValueError('目标信号日不在交易日历中')
    execution_indices = market.dates.searchsorted(targets.index, side='right')
    pending = targets.index[execution_indices == len(market.dates)]
    schedule = {int(i): j for j, i in enumerate(execution_indices) if i < len(market.dates)}
    if not schedule:
        raise ValueError('没有信号对应的下一交易日')
    snapshot_map = {int(i): d for d, i in zip(market.signal_dates, market.dates.searchsorted(market.signal_dates, side='right'))
                    if i < len(market.dates)}
    qty = np.zeros(len(market.stocks))
    pending_sales = np.zeros_like(qty)
    stale_days = np.zeros(len(qty), dtype=int)
    cash = float(initial_cash)
    previous_nav = initial_cash
    fee_rate = cost_bps / 10_000
    daily, trades, snapshots, writeoffs = [], [], [], []
    for t in range(min(schedule), len(market.dates)):
        date = market.dates[t]
        opening = market.adjusted_open[t]
        valid_open = np.isfinite(opening) & (opening > 0) & np.isfinite(market.open[t]) & (market.open[t] > 0)
        # 没开盘价的旧仓只用昨天已知价格估算目标，不倒用当日收盘价。
        prior = market.mark[t - 1] if t else np.full_like(qty, np.nan)
        open_mark = np.where(valid_open, opening, prior)
        if np.any((qty > 0) & ~np.isfinite(open_mark)):
            raise ValueError(f'{date}: 旧仓没有可用历史估值')
        open_value = qty * np.nan_to_num(open_mark)
        nav_before = cash + open_value.sum()
        if nav_before <= 0:
            raise ValueError('账户净值非正')
        desired_buy = np.zeros_like(qty)
        if t in schedule:
            target_value = nav_before * w[schedule[t]]
            pending_sales = np.divide(np.maximum(open_value - target_value, 0), open_mark,
                                      out=np.zeros_like(qty), where=np.isfinite(open_mark) & (open_mark > 0))
            desired_buy = np.maximum(target_value - open_value, 0)
        fees = buy_notional = sell_notional = 0.0
        filled_units_today = np.zeros_like(qty)
        for side in ('sell', 'buy'):
            requested = pending_sales * np.nan_to_num(open_mark) if side == 'sell' else desired_buy
            requested = np.where(requested > 1e-8, requested, 0)
            if not np.any(requested):
                continue
            reasons = np.full(len(qty), 'filled', dtype=object)
            allowed = valid_open.copy()
            reasons[~valid_open] = 'missing_open'
            unavailable = ~market.suspension_covered[t] | market.suspended[t]
            reasons[allowed & unavailable] = 'suspended_or_uncovered'
            allowed &= ~unavailable
            if enforce_limits:
                limit = market.down_limit[t] if side == 'sell' else market.up_limit[t]
                known_limit = np.isfinite(limit) & (limit > 0)
                reasons[allowed & ~known_limit] = 'missing_limit'
                at_limit = market.open[t] <= limit + 1e-6 if side == 'sell' else market.open[t] >= limit - 1e-6
                reasons[allowed & known_limit & at_limit] = 'limit_down' if side == 'sell' else 'limit_up'
                allowed &= known_limit & ~at_limit
            fill = np.where(allowed, requested, 0.0)
            if max_adv_participation is not None:
                adv = market.adv20[t - 1] if t else np.zeros_like(qty)
                capacity = np.where(np.isfinite(adv), np.maximum(adv, 0), 0) * max_adv_participation
                # 同一天同一股票两侧累计成交不超过容量。
                capacity = np.maximum(capacity - filled_units_today * np.nan_to_num(opening), 0)
                limited = allowed & (fill > capacity + 1e-8)
                fill = np.minimum(fill, capacity)
                reasons[limited] = 'adv_cap'
            if side == 'buy' and fill.sum() * (1 + fee_rate) > cash:
                active = fill > 0
                fill *= max(cash, 0) / (fill.sum() * (1 + fee_rate))
                reasons[active & (reasons == 'filled')] = 'cash_scaled'
            units = np.divide(fill, opening, out=np.zeros_like(qty), where=valid_open)
            if side == 'sell':
                # 全量卖出的乘除舍入不能留下1e-16单位的“旧仓”，否则会产生
                # 虚假的持仓数、陈旧估值天数与退市核销次数。
                full_exit = (units > 0) & np.isclose(units, qty, rtol=1e-12, atol=0)
                units[full_exit] = qty[full_exit]
                fill[full_exit] = units[full_exit] * opening[full_exit]
            fee = fill * fee_rate
            if side == 'sell':
                qty -= units
                pending_sales = np.maximum(pending_sales - units, 0)
                cash += float((fill - fee).sum())
                sell_notional = float(fill.sum())
            else:
                qty += units
                cash -= float((fill + fee).sum())
                buy_notional = float(fill.sum())
            filled_units_today += units
            fees += float(fee.sum())
            for j in np.flatnonzero(requested):
                trades.append((date, market.stocks[j], side, requested[j], fill[j], fee[j], reasons[j]))
        if cash < -1e-6 or np.any(qty < -1e-9):
            raise AssertionError('账户发生透支或负持仓')
        cash = max(cash, 0.0)
        qty = np.maximum(qty, 0.0)
        expired = ~np.isnat(market.delist_dates) & (market.delist_dates <= date.to_datetime64()) & (qty > 0)
        writeoff = float((qty[expired] * np.nan_to_num(market.mark[t, expired])).sum())
        for j in np.flatnonzero(expired):
            writeoffs.append((date, market.stocks[j], qty[j], market.mark[t, j],
                              qty[j] * market.mark[t, j]))
        qty[expired] = 0
        pending_sales[expired] = 0
        stale = ~np.isfinite(market.close[t]) | (market.close[t] <= 0)
        stale_days = np.where(stale & (qty > 0), stale_days + 1, 0)
        if np.any((qty > 0) & ~np.isfinite(market.mark[t])):
            raise ValueError(f'{date}: 持仓无可用收盘估值')
        value = qty * np.nan_to_num(market.mark[t])
        nav = float(cash + value.sum())
        if nav <= 0:
            raise ValueError('账户净值非正')
        daily.append((date, nav, nav / previous_nav - 1, cash / nav,
                      buy_notional, sell_notional, fees, (buy_notional + sell_notional) / nav_before,
                      0.5 * (buy_notional + sell_notional) / nav_before,
                      value[stale].sum() / nav, int(stale_days.max()), writeoff, int(expired.sum()),
                      int(np.count_nonzero(qty)), bool(t in schedule)))
        if t in snapshot_map:
            for j in np.flatnonzero(qty > 0):
                snapshots.append((date, snapshot_map[t], market.stocks[j], value[j], value[j] / nav))
        previous_nav = nav
    daily_columns = ['date', 'nav', 'return', 'cash_weight', 'buy_notional', 'sell_notional', 'cost',
                     'traded_fraction', 'one_way_turnover', 'stale_weight', 'max_stale_days',
                     'writeoff_value', 'writeoff_count', 'n_stocks', 'is_rebalance']
    return BacktestResult(
        pd.DataFrame(daily, columns=daily_columns).set_index('date'),
        pd.DataFrame(trades, columns=['date', 'stock_code', 'side', 'requested_notional', 'filled_notional', 'cost', 'reason']),
        pd.DataFrame(snapshots, columns=['date', 'signal_date', 'stock_code', 'value', 'weight']),
        pd.DatetimeIndex(pending),
        pd.DataFrame(writeoffs, columns=['date', 'stock_code', 'adjusted_units', 'last_mark', 'writeoff_value']),
    )


def summarize_backtest(result: BacktestResult, initial_cash: float) -> pd.Series:
    """日频绩效：CAGR、样本波动、算术均值Sharpe（Rf=0）；初始净值参与回撤。"""
    d = result.daily
    r = d['return']
    std = r.std(ddof=1)
    wealth = pd.concat([pd.Series([initial_cash]), d.nav.reset_index(drop=True)], ignore_index=True)
    elapsed_months = len(d.index.to_period('M').unique())
    return pd.Series({
        'Annualized Return': (d.nav.iloc[-1] / initial_cash) ** (252 / len(d)) - 1,
        'Annualized Volatility': std * np.sqrt(252),
        'Sharpe': r.mean() / std * np.sqrt(252) if std > 0 else np.nan,
        'Max Drawdown': (wealth / wealth.cummax() - 1).min(),
        'Mean Monthly Turnover': d.one_way_turnover.sum() / elapsed_months,
        'Total Cost / Initial AUM': d.cost.sum() / initial_cash,
        'Mean Cash Weight': d.cash_weight.mean(),
        'Max Stale Weight': d.stale_weight.max(),
        'Max Stale Days': d.max_stale_days.max(),
        'Writeoff Count': d.writeoff_count.sum(),
        'Writeoff / Initial AUM': d.writeoff_value.sum() / initial_cash,
        'Days': len(d),
    })
