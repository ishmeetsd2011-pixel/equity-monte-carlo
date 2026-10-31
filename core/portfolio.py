"""Portfolio accounting with calendar-month cash flows and rebalancing."""
from __future__ import annotations
from dataclasses import dataclass
from collections.abc import Callable
import numpy as np
import pandas as pd
from core.models import Array, SimulationConfig, daily_log_returns

REBALANCE_MONTHS = {"None": 0, "Monthly": 1, "Quarterly": 3, "Annually": 12}
MAX_RESULT_BYTES = 600_000_000


@dataclass
class PortfolioResult:
    """Values and cash-flow-neutral per-path diagnostics."""
    values: Array
    dates: pd.DatetimeIndex
    invested: Array
    max_drawdown: Array
    annual_return: Array
    annual_volatility: Array
    sharpe: Array


def weights_from_shares(shares: Array, latest_prices: Array) -> tuple[Array, float]:
    """Convert share counts and base-currency adjusted prices into allocations."""
    amounts = np.asarray(shares, dtype=float) * np.asarray(latest_prices, dtype=float)
    if amounts.ndim != 1 or not np.isfinite(amounts).all() or (amounts < 0).any() or amounts.sum() <= 0:
        raise ValueError("Share counts must be nonnegative with a positive total market value.")
    return amounts / amounts.sum(), float(amounts.sum())


def simulate_portfolio(returns: Array, weights: Array, investment: float,
                       config: SimulationConfig, start_date: str | pd.Timestamp,
                       rebalance: str = "None", monthly_contribution: float = 0,
                       risk_free: float = 0.0,
                       progress: Callable[[float], None] | None = None) -> PortfolioResult:
    """Simulate long-only holdings in base currency, vectorised over all paths.

    Future sessions are business-day approximations. Contributions arrive after
    returns on the first simulated business day of each subsequent calendar month.
    They buy target weights. Rebalancing occurs after that contribution, at each
    chosen month interval. Drawdowns and return metrics use time-weighted NAV.
    """
    config.validate()
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (np.asarray(returns).shape[1],) or not np.isfinite(weights).all() or (weights < 0).any() or not np.isclose(weights.sum(), 1):
        raise ValueError("Weights must match the assets, be nonnegative, and sum to one.")
    if not np.isfinite([investment, monthly_contribution, risk_free]).all() or investment <= 0 or monthly_contribution < 0 or risk_free <= -1:
        raise ValueError("Investment must be positive; contribution nonnegative; risk-free rate above -100%.")
    if rebalance not in REBALANCE_MONTHS:
        raise ValueError("Unknown rebalance interval.")
    if (config.days + 1) * config.paths * 8 > MAX_RESULT_BYTES:
        raise ValueError("Result exceeds 600 MB memory budget. Reduce paths or horizon.")
    dates = pd.bdate_range(pd.Timestamp(start_date) + pd.offsets.BDay(1), periods=config.days)
    dates = pd.DatetimeIndex([pd.Timestamp(start_date), *dates])
    values = np.empty((config.days + 1, config.paths))
    values[0] = investment
    invested = np.full(config.days + 1, investment, dtype=float)
    holdings = np.broadcast_to(investment * weights, (config.paths, len(weights))).copy()
    nav = np.ones(config.paths)
    peak = nav.copy()
    drawdown = np.zeros(config.paths)
    sum_return = np.zeros(config.paths)
    sum_square = np.zeros(config.paths)
    period = REBALANCE_MONTHS[rebalance]
    initial_month = dates[0].year * 12 + dates[0].month
    for day, increments in enumerate(daily_log_returns(returns, config), 1):
        holdings *= np.exp(increments)
        total = holdings.sum(axis=1)
        growth = total / values[day - 1]
        daily_return = growth - 1
        nav *= growth
        peak = np.maximum(peak, nav)
        drawdown = np.maximum(drawdown, 1 - nav / peak)
        sum_return += daily_return
        sum_square += daily_return**2
        month = dates[day].year * 12 + dates[day].month
        new_month = dates[day].month != dates[day - 1].month or dates[day].year != dates[day - 1].year
        invested[day] = invested[day - 1]
        if new_month:
            holdings += monthly_contribution * weights
            total += monthly_contribution
            invested[day] += monthly_contribution
            if period and (month - initial_month) % period == 0:
                holdings = total[:, None] * weights
        values[day] = total
        if not np.isfinite(total).all() or (total <= 0).any():
            raise ValueError("Simulation overflow/underflow. Reduce horizon or extreme drift inputs.")
        if progress and (day % max(1, config.days // 50) == 0 or day == config.days):
            progress(day / config.days)
    daily_mean = sum_return / config.days
    variance = np.maximum(0, (sum_square - config.days * daily_mean**2) / max(1, config.days - 1))
    volatility = np.sqrt(variance * config.annualisation)
    daily_rf = (1 + risk_free)**(1 / config.annualisation) - 1
    sharpe = np.divide((daily_mean - daily_rf) * config.annualisation, volatility,
                       out=np.full_like(volatility, np.nan), where=volatility > 1e-12)
    return PortfolioResult(values, dates, invested, drawdown,
                           nav**(config.annualisation / config.days) - 1, volatility, sharpe)
