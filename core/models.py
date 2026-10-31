"""Vectorised Monte Carlo innovations; loops run over dates, never paths."""
from __future__ import annotations
from dataclasses import dataclass
from collections.abc import Iterator
import warnings
import numpy as np
from numpy.typing import NDArray
from scipy.stats import t

Array = NDArray[np.float64]
MODELS = ("GBM", "Student-t", "Bootstrap", "GARCH")


@dataclass(frozen=True)
class SimulationConfig:
    """Daily simulation settings; drift override is annual expected log growth."""
    model: str = "GBM"
    paths: int = 10_000
    days: int = 252
    seed: int = 42
    annualisation: int = 252
    antithetic: bool = False
    block_size: int = 1
    shrinkage: bool = False
    annual_log_drift: float | None = None

    def validate(self) -> None:
        """Validate bounds before allocating simulation memory."""
        if self.model not in MODELS:
            raise ValueError(f"Unknown model: {self.model}")
        if not 1 <= self.paths <= 100_000 or not 1 <= self.days <= 12_600:
            raise ValueError("Paths must be 1–100,000 and days 1–12,600.")
        if not 1 <= self.annualisation <= 366 or self.block_size < 1:
            raise ValueError("Invalid annualisation or block size.")
        if self.annual_log_drift is not None and not np.isfinite(self.annual_log_drift):
            raise ValueError("Drift must be finite.")
        if self.antithetic and self.model == "Bootstrap":
            raise ValueError("Antithetic variates are unavailable for historical bootstrap.")


def nearest_psd(matrix: Array, floor: float = 1e-12) -> Array:
    """Project a symmetric matrix onto the PSD cone with a positive eigenvalue floor."""
    symmetric = (matrix + matrix.T) / 2
    values, vectors = np.linalg.eigh(symmetric)
    return (vectors * np.maximum(values, floor)) @ vectors.T


def safe_cholesky(matrix: Array) -> Array:
    """Factor covariance, repairing singular or indefinite estimates as needed."""
    matrix = np.atleast_2d(np.asarray(matrix, dtype=float))
    if matrix.shape[0] != matrix.shape[1] or not np.isfinite(matrix).all():
        raise ValueError("Covariance must be a finite square matrix.")
    try:
        return np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError:
        return np.linalg.cholesky(nearest_psd(matrix))


def estimate(returns: Array, shrinkage: bool = False) -> tuple[Array, Array]:
    """Estimate daily log means and covariance, optionally with Ledoit–Wolf."""
    returns = np.asarray(returns, dtype=float)
    if returns.ndim != 2 or returns.shape[0] < 30 or returns.shape[1] < 1 or not np.isfinite(returns).all():
        raise ValueError("Need at least 30 finite observations in a dates × assets matrix.")
    if shrinkage:
        from sklearn.covariance import LedoitWolf
        covariance = LedoitWolf().fit(returns).covariance_
    else:
        covariance = np.atleast_2d(np.cov(returns, rowvar=False))
    return returns.mean(axis=0), covariance


def fit_garch(returns: Array) -> tuple[Array, Array, Array, Array, Array, Array]:
    """Fit univariate GARCH(1,1) with arch, retaining final conditional states."""
    from arch import arch_model
    parameters = []
    residuals = []
    for column in returns.T:
        if column.std() < 1e-10:
            residuals.append(np.zeros(len(column)))
            parameters.append((0.0, 0.0, 0.0, 0.0, 0.0))
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted = arch_model(column * 100, mean="Constant", vol="GARCH", p=1, q=1,
                                rescale=False).fit(disp="off", show_warning=False)
        if fitted.convergence_flag != 0:
            raise ValueError("GARCH fit did not converge; extend lookback or use another model.")
        p = fitted.params
        if p["alpha[1]"] + p["beta[1]"] >= 0.99999:
            raise ValueError("GARCH fit is non-stationary; extend lookback or use another model.")
        residuals.append(np.asarray(fitted.std_resid))
        parameters.append((p["omega"] / 10000, p["alpha[1]"], p["beta[1]"],
                           float(fitted.conditional_volatility[-1]) ** 2 / 10000,
                           float(fitted.resid[-1]) / 100))
    return (*tuple(np.asarray(parameters).T), np.asarray(residuals).T)


def daily_log_returns(returns: Array, config: SimulationConfig) -> Iterator[Array]:
    """Yield paths × assets daily log returns with preserved cross-asset dependence.

    Student-t uses a shared chi-square scale (elliptical multivariate t).
    Bootstrap samples whole rows in circular blocks. GARCH uses constant
    correlation of standardized residuals and path-specific conditional variance.
    """
    config.validate()
    returns = np.asarray(returns, dtype=float)
    mean, covariance = estimate(returns, config.shrinkage)
    drift = mean if config.annual_log_drift is None else np.full_like(mean, config.annual_log_drift / config.annualisation)
    rng = np.random.default_rng(config.seed)
    n, assets = config.paths, returns.shape[1]
    factor = safe_cholesky(covariance)
    sample_count = (n + 1) // 2 if config.antithetic else n
    df = None
    if config.model == "Student-t":
        standardized = ((returns - mean) / np.maximum(returns.std(axis=0, ddof=1), 1e-12)).ravel()
        df = max(2.1, min(200.0, float(t.fit(standardized, floc=0)[0])))
    if config.model == "GARCH":
        omega, alpha, beta, last_var, last_residual, residual_matrix = fit_garch(returns)
        # Refit residual correlation, not raw-return correlation.
        _, residual_cov = estimate(residual_matrix, config.shrinkage)
        scales = np.sqrt(np.maximum(np.diag(residual_cov), 1e-12))
        correlation = residual_cov / np.outer(scales, scales)
        np.fill_diagonal(correlation, 1)
        factor = safe_cholesky(correlation)
        variance = np.broadcast_to(last_var, (n, assets)).copy()
        residual = np.broadcast_to(last_residual, (n, assets)).copy()
    for day in range(config.days):
        if config.model == "Bootstrap":
            if day % config.block_size == 0:
                starts = rng.integers(0, len(returns), size=n)
            yield returns[(starts + day % config.block_size) % len(returns)] - mean + drift
            continue
        z = rng.standard_normal((sample_count, assets))
        if df is not None:
            z *= np.sqrt((df - 2) / rng.chisquare(df, size=(sample_count, 1)))
        if config.antithetic:
            z = np.concatenate((z, -z), axis=0)[:n]
        shocks = np.einsum("ij,kj->ik", z, factor, optimize=False)
        if config.model == "GARCH":
            variance = omega + alpha * residual**2 + beta * variance
            residual = np.sqrt(np.maximum(variance, 0)) * shocks
            yield drift + residual
        else:
            yield drift + shocks


def simulate_prices(returns: Array, initial: Array, config: SimulationConfig) -> Array:
    """Materialise prices for small research jobs; UI uses streaming portfolio values."""
    initial = np.asarray(initial, dtype=float)
    if initial.shape != (np.asarray(returns).shape[1],) or (initial <= 0).any():
        raise ValueError("Initial prices must be positive and match the assets.")
    paths = np.empty((config.days + 1, config.paths, len(initial)))
    paths[0] = initial
    for day, increments in enumerate(daily_log_returns(returns, config), 1):
        paths[day] = paths[day - 1] * np.exp(increments)
    return paths
