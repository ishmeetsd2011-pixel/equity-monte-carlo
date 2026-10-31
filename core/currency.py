"""Currency identification and adjusted-price conversion."""
import numpy as np
import pandas as pd

INDIAN_INDICES = {"^NSEI", "^BSESN"}


def asset_currency(ticker: str) -> str:
    """Identify the quote currency for the supported equity universes."""
    return "INR" if ticker.endswith((".NS", ".BO")) or ticker in INDIAN_INDICES else "USD"


def convert_prices(prices: pd.DataFrame, usd_inr: pd.Series, base: str) -> pd.DataFrame:
    """Convert every historical observation using contemporaneous INR per USD."""
    if base not in {"USD", "INR"}:
        raise ValueError("Base currency must be USD or INR.")
    fx = usd_inr.reindex(prices.index).ffill()
    if fx.isna().any() or not np.isfinite(fx).all() or (fx <= 0).any():
        raise ValueError("USD/INR history is missing or invalid for the price dates.")
    result = prices.copy()
    for ticker in prices:
        if asset_currency(ticker) != base:
            result[ticker] = prices[ticker] * fx if base == "INR" else prices[ticker] / fx
    return result
