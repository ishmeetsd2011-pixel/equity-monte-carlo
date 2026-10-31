"""Deterministic portfolio accounting and known-distribution risk checks."""
import numpy as np
import pandas as pd
import pytest
from core.metrics import value_at_risk, summarize, percentile_frame, terminal_frame
from core.models import SimulationConfig
from core.portfolio import simulate_portfolio, weights_from_shares
from core.report import html_report
from app.charts import fan_chart


def test_var_cvar_known_distribution():
    losses = np.arange(1, 101, dtype=float)
    assert value_at_risk(losses, .95) == pytest.approx((95.05, 98.0))
    assert value_at_risk(losses, .99) == pytest.approx((99.01, 100.))
    assert value_at_risk(np.array([-5., -5.])) == (-5., -5.)
    with pytest.raises(ValueError):
        value_at_risk(np.array([]))


def test_contributions_not_returns_or_drawdown():
    config = SimulationConfig(model="Bootstrap", paths=10, days=45)
    result = simulate_portfolio(np.zeros((100, 2)), [.5, .5], 1000, config, "2025-01-15", "Monthly", 100)
    assert result.invested[-1] == 1200
    np.testing.assert_allclose(result.values[-1], 1200)
    np.testing.assert_allclose(result.max_drawdown, 0)
    np.testing.assert_allclose(result.annual_return, 0)
    metrics = summarize(result, 1200)
    assert metrics["Probability of loss"] == 0
    assert metrics["Probability of reaching target"] == 1
    assert len(percentile_frame(result)) == 46
    assert len(terminal_frame(result)) == 10
    report = html_report(metrics, {"ticker": "<script>"}, [fan_chart(result, "INR")])
    assert "&lt;script&gt;" in report
    assert "plotly" in report


def test_buy_hold_vs_rebalancing():
    returns = np.tile([.01, -.01], (100, 1))
    config = SimulationConfig(model="Bootstrap", paths=3, days=60)
    hold = simulate_portfolio(returns, [.5, .5], 1000, config, "2025-01-01")
    monthly = simulate_portfolio(returns, [.5, .5], 1000, config, "2025-01-01", "Monthly")
    expected = 500 * (np.exp(.6) + np.exp(-.6))
    np.testing.assert_allclose(hold.values[-1], expected)
    assert (hold.values[-1] > monthly.values[-1]).all()


def test_cash_flow_neutral_drawdown():
    returns = np.full((100, 1), np.log(.99))
    result = simulate_portfolio(returns, [1], 1000, SimulationConfig(model="Bootstrap", paths=3, days=30), "2025-01-15", monthly_contribution=10000)
    np.testing.assert_allclose(result.max_drawdown, 1 - .99**30)
    assert (result.values[-1] > 1000).all()


def test_shares_and_invalid_inputs():
    weights, total = weights_from_shares(np.array([2, 1]), np.array([100, 300]))
    np.testing.assert_allclose(weights, [.4, .6])
    assert total == 500
    for weights in [[.2, .2], [-1, 2], [1], [np.nan, 0]]:
        with pytest.raises(ValueError):
            simulate_portfolio(np.zeros((100, 2)), weights, 1000, SimulationConfig(paths=10), "2025-01-01")
    with pytest.raises(ValueError, match="memory"):
        simulate_portfolio(np.zeros((100, 2)), [.5, .5], 1000, SimulationConfig(paths=100000, days=10000), "2025-01-01")
