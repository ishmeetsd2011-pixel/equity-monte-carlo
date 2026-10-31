"""Statistical convergence and reproducibility checks, independent of Yahoo."""
import numpy as np
import pytest
from core.models import SimulationConfig, daily_log_returns, estimate, safe_cholesky, simulate_prices


@pytest.fixture
def returns():
    """Seeded bivariate history."""
    rng = np.random.default_rng(1)
    return rng.multivariate_normal([.0003, .0002], [[.0001, .00006], [.00006, .000225]], size=3000)


def test_gbm_mean_variance(returns):
    config = SimulationConfig(paths=100000, days=40, seed=123)
    mu, covariance = estimate(returns[:, :1])
    terminal = np.ones((config.paths, 1))
    for increment in daily_log_returns(returns[:, :1], config):
        terminal *= np.exp(increment)
    expected_mean = np.exp(40 * (mu[0] + covariance[0, 0] / 2))
    expected_var = np.exp(80 * mu[0] + 40 * covariance[0, 0]) * np.expm1(40 * covariance[0, 0])
    assert terminal.mean() == pytest.approx(expected_mean, rel=.001)
    assert terminal.var() == pytest.approx(expected_var, rel=.025)


def test_cholesky_reproduces_correlation(returns):
    config = SimulationConfig(paths=100000, days=1, seed=11)
    samples = next(daily_log_returns(returns, config))
    np.testing.assert_allclose(np.corrcoef(samples.T), np.corrcoef(returns.T), atol=.012)


@pytest.mark.parametrize("model", ["GBM", "Student-t", "Bootstrap", "GARCH"])
def test_seed_reproducibility_and_finite(returns, model):
    config = SimulationConfig(model=model, paths=100, days=10, block_size=4)
    first = simulate_prices(returns, np.array([100., 200.]), config)
    second = simulate_prices(returns, np.array([100., 200.]), config)
    np.testing.assert_array_equal(first, second)
    assert np.isfinite(first).all()
    assert (first > 0).all()


def test_psd_fallback():
    for covariance in [np.ones((2, 2)), np.array([[1., 2.], [2., 1.]])]:
        factor = safe_cholesky(covariance)
        assert np.linalg.eigvalsh(factor @ factor.T).min() > 0


def test_antithetic_pairs(returns):
    config = SimulationConfig(paths=1000, days=1, antithetic=True)
    draws = next(daily_log_returns(returns, config))
    np.testing.assert_allclose(draws.mean(axis=0), returns.mean(axis=0), atol=1e-16)


def test_bootstrap_preserves_rows_and_blocks(returns):
    config = SimulationConfig(model="Bootstrap", paths=100, days=5, block_size=5)
    draws = np.asarray(list(daily_log_returns(returns, config)))
    starts = np.argmin(np.abs(returns[:, 0, None] - draws[0, :, 0]), axis=0)
    for day in range(5):
        np.testing.assert_allclose(draws[day], returns[(starts + day) % len(returns)])


def test_shrinkage_and_bad_inputs(returns):
    _, covariance = estimate(returns, True)
    assert np.linalg.eigvalsh(covariance).min() > 0
    with pytest.raises(ValueError):
        list(daily_log_returns(np.full((100, 2), np.nan), SimulationConfig()))
    with pytest.raises(ValueError):
        SimulationConfig(model="nope").validate()
    with pytest.raises(ValueError):
        SimulationConfig(model="Bootstrap", antithetic=True).validate()
