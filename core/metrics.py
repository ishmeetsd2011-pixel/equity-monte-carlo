"""Cash-flow-aware risk measures, with losses positive and gains negative."""
import numpy as np
import pandas as pd
from core.models import Array
from core.portfolio import PortfolioResult


def value_at_risk(losses: Array, confidence: float = .95) -> tuple[float, float]:
    """Empirical loss quantile and expected loss at or above that quantile."""
    losses = np.asarray(losses, dtype=float)
    if losses.ndim != 1 or losses.size == 0 or not np.isfinite(losses).all() or not 0 < confidence < 1:
        raise ValueError("Provide finite losses and confidence strictly between zero and one.")
    var = float(np.quantile(losses, confidence))
    return var, float(losses[losses >= var].mean())


def summarize(result: PortfolioResult, target: float) -> pd.Series:
    """Summarize monetary terminal risk and time-weighted return diagnostics."""
    if not np.isfinite(target) or target < 0:
        raise ValueError("Target must be nonnegative and finite.")
    terminal = result.values[-1]
    losses = result.invested[-1] - terminal
    output = {
        "Total paid in": result.invested[-1],
        "Expected terminal value": terminal.mean(), "Median terminal value": np.median(terminal),
        "Terminal standard deviation": terminal.std(ddof=1) if len(terminal) > 1 else 0,
        "Probability of loss": (losses > 0).mean(),
        "Probability of reaching target": (terminal >= target).mean(),
        "Expected max drawdown": result.max_drawdown.mean(),
        "95th percentile max drawdown": np.quantile(result.max_drawdown, .95),
        "Mean annualised time-weighted return": result.annual_return.mean(),
        "Mean annualised volatility": result.annual_volatility.mean(),
        "Mean Sharpe ratio": np.nanmean(result.sharpe) if np.isfinite(result.sharpe).any() else np.nan,
    }
    for confidence in (.95, .99):
        var, cvar = value_at_risk(losses, confidence)
        output[f"VaR {confidence:.0%}"] = var
        output[f"CVaR {confidence:.0%}"] = cvar
    return pd.Series(output, name="Value")


def percentile_frame(result: PortfolioResult) -> pd.DataFrame:
    """Daily 5/25/50/75/95 percentiles for display/export."""
    frame = pd.DataFrame(np.percentile(result.values, [5, 25, 50, 75, 95], axis=1).T,
                         index=result.dates, columns=["P05", "P25", "P50", "P75", "P95"])
    frame["Paid in"] = result.invested
    return frame


def terminal_frame(result: PortfolioResult) -> pd.DataFrame:
    """Per-path terminal and drawdown results for download."""
    return pd.DataFrame({"terminal_value": result.values[-1], "net_profit": result.values[-1] - result.invested[-1],
                         "max_drawdown": result.max_drawdown, "annual_return": result.annual_return,
                         "annual_volatility": result.annual_volatility, "sharpe": result.sharpe})
