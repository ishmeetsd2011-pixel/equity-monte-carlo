"""Data validation, cache and mixed-currency conversions."""
import numpy as np
import pandas as pd
import pytest
from core.data import DataError, fetch_prices, normalize_tickers
from core.currency import asset_currency, convert_prices


def test_currency_conversion():
    dates = pd.date_range("2025-01-01", periods=2)
    prices = pd.DataFrame({"AAPL": [100., 110.], "TCS.NS": [800., 900.], "^NSEI": [8000., 9000.]}, index=dates)
    fx = pd.Series([80., 90.], index=dates)
    inr = convert_prices(prices, fx, "INR")
    usd = convert_prices(prices, fx, "USD")
    np.testing.assert_allclose(inr.AAPL, [8000., 9900.])
    np.testing.assert_allclose(usd["TCS.NS"], [10., 10.])
    np.testing.assert_allclose(inr["TCS.NS"], prices["TCS.NS"])
    assert asset_currency("ABC.BO") == "INR"
    with pytest.raises(ValueError):
        convert_prices(prices, fx * 0, "USD")
    with pytest.raises(ValueError):
        convert_prices(prices, fx, "EUR")


def test_tickers():
    assert normalize_tickers("reliance, TCS, reliance", "NSE") == ["RELIANCE.NS", "TCS.NS"]
    assert normalize_tickers("abc", "BSE") == ["ABC.BO"]
    assert normalize_tickers("AAPL, ^GSPC, TCS.NS") == ["AAPL", "^GSPC", "TCS.NS"]
    for invalid in ["", "BAD!", "../../file"]:
        with pytest.raises(DataError):
            normalize_tickers(invalid)


def test_bad_ticker_empty_and_api_failure(tmp_path):
    def empty(*args):
        return pd.Series(dtype=float)
    def failure(*args):
        raise RuntimeError("API unavailable")
    for download in [empty, failure]:
        with pytest.raises(DataError, match="MISSING"):
            fetch_prices(["MISSING"], "2025-01-01", "2025-06-01", cache_dir=tmp_path, downloader=download)


def test_cache_refresh_and_alignment(tmp_path):
    dates = pd.bdate_range("2025-01-01", "2025-06-02")
    calls = []
    def download(ticker, *args):
        calls.append(ticker)
        series = pd.Series(np.arange(len(dates)) + 100., index=dates)
        return series.drop(dates[30]) if ticker == "B" else series
    first = fetch_prices(["A", "B"], "2025-01-01", "2025-06-03", cache_dir=tmp_path, downloader=download)
    assert first.loc[dates[30], "B"] == first.loc[dates[29], "B"]
    second = fetch_prices(["A", "B"], "2025-01-01", "2025-06-03", cache_dir=tmp_path, downloader=download)
    pd.testing.assert_frame_equal(first, second, check_freq=False)
    assert len(calls) == 2
    fetch_prices(["A"], "2025-01-01", "2025-06-03", True, tmp_path, download)
    assert len(calls) == 3


def test_delisted_stale_and_invalid_dates(tmp_path):
    def stale(*args):
        return pd.Series(100., index=pd.bdate_range("2025-01-01", "2025-04-01"))
    with pytest.raises(DataError, match="stale"):
        fetch_prices(["OLD"], "2025-01-01", "2025-06-01", cache_dir=tmp_path, downloader=stale)
    with pytest.raises(DataError):
        fetch_prices(["A"], "2026-01-01", "2025-01-01", cache_dir=tmp_path)


def test_government_yield_success_and_fallback(monkeypatch):
    import requests
    from core.data import fetch_government_yield
    class Response:
        text = "observation_date,DGS10\n2026-10-28,4.0\n2026-10-29,4.1\n2026-10-30,.\n"
        def raise_for_status(self):
            pass
    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: Response())
    rate, source = fetch_government_yield("USD")
    assert rate == 4.1
    assert "2026-10-29" in source
    def fail(*args, **kwargs):
        raise requests.ConnectionError("offline")
    monkeypatch.setattr(requests, "get", fail)
    rate, source = fetch_government_yield("INR")
    assert rate == 6.5
    assert "fallback" in source
    with pytest.raises(ValueError):
        fetch_government_yield("EUR")
