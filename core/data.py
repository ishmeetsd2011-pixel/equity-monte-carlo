"""Yahoo adjusted data with atomic on-disk CSV caching and calendar alignment."""
from __future__ import annotations
import hashlib
import os
import re
import tempfile
from pathlib import Path
from collections.abc import Callable
import numpy as np
import pandas as pd

MIN_OBSERVATIONS = 60
MAX_STALE_DAYS = 10


class DataError(ValueError):
    """Actionable market-data error."""


def normalize_tickers(text: str, market: str = "Mixed / US") -> list[str]:
    """Parse unique symbols, adding an Indian exchange suffix when selected."""
    symbols = list(dict.fromkeys(re.split(r"[,\s]+", text.strip().upper())))
    if not symbols or not symbols[0]:
        raise DataError("Enter at least one ticker.")
    result = []
    for symbol in symbols:
        if not re.fullmatch(r"[A-Z0-9^][A-Z0-9.^=\-]{0,24}", symbol):
            raise DataError(f"Invalid ticker: {symbol!r}.")
        if market in {"NSE", "BSE"} and not symbol.startswith("^") and "." not in symbol:
            symbol += ".NS" if market == "NSE" else ".BO"
        result.append(symbol)
    return list(dict.fromkeys(result))


def _download(ticker: str, start: str, end: str) -> pd.Series:
    """Use auto-adjusted Close (splits and dividends)."""
    import yfinance as yf
    frame = yf.download(ticker, start=start, end=end, auto_adjust=True,
                        progress=False, threads=False, timeout=20, multi_level_index=False)
    if frame is None or frame.empty or "Close" not in frame:
        raise DataError(f"No data for {ticker}; check exchange suffix, listing dates or Yahoo availability.")
    return frame["Close"]


def fetch_prices(tickers: list[str], start: str, end: str, refresh: bool = False,
                 cache_dir: str | Path = ".cache/prices",
                 downloader: Callable | None = None) -> pd.DataFrame:
    """Load native histories; fail explicitly if any requested symbol is unavailable.

    The union of observed sessions is forward-filled, never backward-filled.
    Initial non-overlap is removed. Stale tails exceeding ten calendar days fail
    instead of silently treating a delisted stock as a constant-price asset.
    End is exclusive, following Yahoo's API.
    """
    if not tickers or pd.Timestamp(start) >= pd.Timestamp(end):
        raise DataError("Provide tickers and a start date before the end date.")
    directory = Path(cache_dir)
    directory.mkdir(parents=True, exist_ok=True)
    series = []
    for ticker in tickers:
        key = hashlib.sha256(f"{ticker}|{start}|{end}|adjusted-v1".encode()).hexdigest()
        path = directory / f"{key}.csv"
        try:
            if path.exists() and not refresh:
                item = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
            else:
                item = (downloader or _download)(ticker, start, end).copy()
                if isinstance(item, pd.DataFrame):
                    item = item.iloc[:, 0]
                item.index = pd.to_datetime(item.index).tz_localize(None).normalize()
                item = pd.to_numeric(item, errors="coerce").dropna().sort_index()
                item = item[~item.index.duplicated(keep="last")]
                if len(item) < MIN_OBSERVATIONS or (item <= 0).any() or not np.isfinite(item).all():
                    raise DataError(f"{ticker}: need at least {MIN_OBSERVATIONS} valid positive daily prices.")
                with tempfile.NamedTemporaryFile(dir=directory, suffix=".csv", delete=False) as handle:
                    temporary = handle.name
                item.rename(ticker).to_csv(temporary)
                os.replace(temporary, path)
            item.name = ticker
            if len(item) < MIN_OBSERVATIONS or (item <= 0).any() or not np.isfinite(item).all():
                raise DataError(f"{ticker}: invalid cached history; use Refresh data.")
            if (pd.Timestamp(end) - item.index[-1]).days > MAX_STALE_DAYS:
                raise DataError(f"{ticker}: stale last quote ({item.index[-1].date()}); possible delisting or API gap.")
            series.append(item)
        except DataError:
            raise
        except Exception as exc:
            raise DataError(f"Could not load {ticker}: {exc}. Check symbol/network and retry Refresh data.") from exc
    raw = pd.concat(series, axis=1).sort_index()
    # Long internal gaps are not ordinary exchange holidays.
    for item in series:
        if item.index.to_series().diff().dt.days.max() > MAX_STALE_DAYS:
            raise DataError(f"{item.name}: history contains a gap longer than {MAX_STALE_DAYS} days.")
    aligned = raw.ffill().dropna()
    if len(aligned) < MIN_OBSERVATIONS:
        raise DataError("Insufficient overlapping history across tickers and FX.")
    return aligned


def fetch_government_yield(base: str) -> tuple[float, str]:
    """Retrieve the latest available 10Y government yield, in percentage points.

    Returns an explicit indicative fallback on provider failure. Requests uses its
    maintained certificate bundle; TLS verification remains enabled.
    """
    from datetime import date
    import io
    import requests
    if base not in {"INR", "USD"}:
        raise ValueError("Yield currency must be INR or USD.")
    series = "INDIRLTLT01STM" if base == "INR" else "DGS10"
    try:
        response = requests.get("https://fred.stlouisfed.org/graph/fredgraph.csv",
                                params={"id": series, "cosd": "2020-01-01", "coed": str(date.today())}, timeout=5)
        response.raise_for_status()
        frame = pd.read_csv(io.StringIO(response.text))
        valid = pd.to_numeric(frame[series], errors="coerce").dropna()
        if valid.empty or not np.isfinite(valid.iloc[-1]):
            raise ValueError("No valid yield observations")
        return float(valid.iloc[-1]), f"FRED {series}, observation {frame.iloc[valid.index[-1], 0]}; verify freshness."
    except Exception:
        return (6.5 if base == "INR" else 4.0), "Indicative 10Y government-yield fallback; live retrieval unavailable. Enter a current rate."
