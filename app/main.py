"""Run from the repository root: streamlit run app/main.py."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataclasses import asdict
from datetime import date, timedelta
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from core.currency import convert_prices
from core.data import DataError, fetch_prices, normalize_tickers, fetch_government_yield
from core.models import MODELS, SimulationConfig
from core.portfolio import REBALANCE_MONTHS, simulate_portfolio, weights_from_shares
from core.metrics import summarize, percentile_frame, terminal_frame
from core.report import DISCLAIMER, html_report
from app.charts import fan_chart, terminal_chart

st.set_page_config(page_title="Equity Monte Carlo Lab", page_icon="📈", layout="wide")


@st.cache_data(ttl=3600, show_spinner=False)
def load_data(tickers: tuple[str, ...], start: str, end: str, refresh_token: int) -> pd.DataFrame:
    """Cache downloaded/aligned frames for an hour; token forces disk refresh."""
    return fetch_prices(list(tickers), start, end, refresh=refresh_token > 0)


@st.cache_data(show_spinner=False)
def demo_data(tickers: tuple[str, ...], start: str, end: str) -> pd.DataFrame:
    """Explicitly synthetic offline histories for trying the application."""
    dates = pd.bdate_range(start, pd.Timestamp(end) - pd.Timedelta(days=1))
    rng = np.random.default_rng(2026)
    common = rng.normal(0, .007, (len(dates), 1))
    increments = .0002 + common + rng.normal(0, .01, (len(dates), len(tickers)))
    frame = pd.DataFrame(100 * np.exp(np.cumsum(increments, axis=0)), index=dates, columns=tickers)
    frame["INR=X"] = 83 * np.exp(np.cumsum(rng.normal(0, .002, len(dates))))
    return frame


@st.cache_data(ttl=86400, show_spinner=False)
def government_yield(base: str) -> tuple[float, str]:
    """Read FRED 10Y government benchmark series, with a labelled fallback."""
    return fetch_government_yield(base)


def parse_numbers(text: str, count: int) -> np.ndarray:
    """Parse and validate a comma-separated allocation vector."""
    try:
        values = np.array([float(value.strip()) for value in text.split(",")])
    except ValueError as exc:
        raise ValueError("Enter comma-separated numbers for weights or shares.") from exc
    if len(values) != count or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError(f"Provide {count} nonnegative finite weights or share counts.")
    return values


st.title("Equity Monte Carlo Lab")
st.caption("Indian and US equities • mixed-currency portfolios • reproducible scenarios")
st.warning(DISCLAIMER)
with st.sidebar:
    st.header("Portfolio & simulation")
    market = st.selectbox("Ticker market", ["Mixed / US", "NSE", "BSE"])
    ticker_text = st.text_input("Tickers", "AAPL, RELIANCE.NS")
    st.caption("Mixed portfolios: include .NS / .BO explicitly. Indices: ^NSEI, ^BSESN, ^GSPC.")
    sizing = st.radio("Position input", ["Weights", "Share counts"], horizontal=True)
    allocation_text = st.text_input("Weights (sum 1) or shares", "0.5, 0.5")
    investment = st.number_input("Initial investment (base currency)", min_value=1.0, value=100000.0)
    base = st.selectbox("Base currency", ["INR", "USD"])
    st.caption("Share mode uses share market value as initial investment; fractional adjusted-share equivalents.")
    model = st.selectbox("Model", MODELS)
    paths = st.number_input("Paths", 1000, 100000, 10000, 1000)
    unit = st.selectbox("Horizon unit", ["Days", "Months", "Years"])
    horizon = st.number_input("Horizon", 1, 12600 if unit == "Days" else 600 if unit == "Months" else 50, 252 if unit == "Days" else 12 if unit == "Months" else 1)
    annualisation = st.number_input("Sessions per year", 1, 366, 248 if market in {"NSE", "BSE"} else 252)
    lookback = st.slider("Lookback (years)", 1, 15, 5)
    seed = st.number_input("Random seed", 0, 2**32 - 1, 42)
    antithetic = st.checkbox("Antithetic variates", disabled=model == "Bootstrap")
    shrinkage = st.checkbox("Ledoit–Wolf covariance shrinkage")
    block_size = st.number_input("Bootstrap block length (sessions)", 1, 252, 5)
    rebalance = st.selectbox("Rebalance", list(REBALANCE_MONTHS))
    sip = st.number_input("Monthly contribution (base currency)", min_value=0.0, value=0.0)
    override = st.checkbox("Override historical drift")
    annual_log_drift = None
    if override:
        st.warning("Historical drift is noisy. A manual expected return is a scenario assumption.")
        expected = st.number_input("Annual compounded growth assumption (%)", -99.0, 300.0, 8.0)
        annual_log_drift = float(np.log1p(expected / 100))
    st.caption("Drift override sets annual expected log growth for each base-currency asset; it is not expected arithmetic return.")
    # Fetch on request to keep first paint fast and support offline mode.
    if st.button("Load 10Y government yield"):
        rate, source = government_yield(base)
        st.session_state[f"rf_{base}"] = rate
        st.session_state[f"rf_source_{base}"] = source
    risk_free = st.number_input("Annual risk-free rate (%)", min_value=-99.0, max_value=100.0,
                                value=6.5 if base == "INR" else 4.0, key=f"rf_{base}")
    st.caption(st.session_state.get(f"rf_source_{base}", "Indicative India 10Y G-sec / US 10Y Treasury proxy; editable, not a live quote."))
    target = st.number_input("Target terminal value", min_value=0.0, value=150000.0)
    benchmark = st.selectbox("Compare benchmark", ["None", "Nifty 50", "S&P 500"])
    alternate = st.text_input("Alternative portfolio weights (optional)", "")
    demo = st.checkbox("Synthetic offline demo data", False)
    refresh = st.button("Refresh data", help="Download new data on the next run, replacing local cache.")
    if refresh:
        load_data.clear()
        st.session_state["refresh_requested"] = True
        st.success("Refresh queued for next run.")
    run = st.button("Run simulation", type="primary")

if run:
    try:
        tickers = normalize_tickers(ticker_text, market)
        if "INR=X" in tickers:
            raise ValueError("Enter equity/index tickers only; USD/INR is fetched automatically.")
        benchmark_ticker = {"None": None, "Nifty 50": "^NSEI", "S&P 500": "^GSPC"}[benchmark]
        all_tickers = tuple(dict.fromkeys(tickers + ([benchmark_ticker] if benchmark_ticker else []) + ["INR=X"]))
        days = max(1, round(horizon * (1 if unit == "Days" else annualisation / 12 if unit == "Months" else annualisation)))
        number_runs = 1 + bool(alternate.strip()) + bool(benchmark_ticker)
        if (days + 1) * paths * 8 * number_runs > 600_000_000:
            raise ValueError("Combined outputs exceed the 600 MB budget. Reduce paths, horizon or comparisons.")
        config = SimulationConfig(model, int(paths), days, int(seed), int(annualisation),
                                  bool(antithetic and model != "Bootstrap"), int(block_size), shrinkage, annual_log_drift)
        start = (date.today() - timedelta(days=round(lookback * 365.25))).isoformat()
        end = date.today().isoformat()
        with st.spinner("Loading and aligning adjusted daily prices…"):
            token = int(st.session_state.pop("refresh_requested", False))
            raw = demo_data(all_tickers, start, end) if demo else load_data(all_tickers, start, end, token)
            prices = convert_prices(raw.drop(columns="INR=X"), raw["INR=X"], base)
            returns = np.log(prices / prices.shift()).dropna()
        weights = parse_numbers(allocation_text, len(tickers))
        if sizing == "Share counts":
            weights, investment = weights_from_shares(weights, prices[tickers].iloc[-1].to_numpy())
        progress = st.progress(0.0, text="Simulating vectorised paths…")
        result = simulate_portfolio(returns[tickers].to_numpy(), weights, investment, config, prices.index[-1],
                                    rebalance, sip, risk_free / 100, progress.progress)
        comparisons = {"Portfolio": result}
        if alternate.strip():
            comparisons["Alternative"] = simulate_portfolio(returns[tickers].to_numpy(), parse_numbers(alternate, len(tickers)),
                                                            investment, config, prices.index[-1], rebalance, sip, risk_free / 100)
        if benchmark_ticker:
            comparisons[benchmark] = simulate_portfolio(returns[[benchmark_ticker]].to_numpy(), np.array([1.]), investment,
                                                        config, prices.index[-1], "None", sip, risk_free / 100)
        progress.empty()
        settings = {**asdict(config), "tickers": tickers, "weights": weights.tolist(), "initial_investment": investment,
                    "base_currency": base, "rebalance": rebalance, "monthly_contribution": sip,
                    "risk_free": risk_free / 100, "target": target, "last_observation": str(prices.index[-1].date()),
                    "lookback_start": start, "synthetic_data": demo,
                    "benchmark": benchmark, "alternative_weights": alternate,
                    "risk_free_source": st.session_state.get(f"rf_source_{base}", "Indicative editable 10Y government-yield proxy")}
        st.session_state["run"] = dict(result=result, prices=prices, raw=raw, returns=returns, settings=settings, comparisons=comparisons)
    except (ValueError, DataError, OSError) as exc:
        st.error(str(exc))
    except Exception as exc:
        st.error(f"Run failed: {exc}. Retry with fewer paths or another model; use offline demo to check the installation.")

if "run" not in st.session_state:
    st.info("Set the portfolio in the sidebar, then run. Choose synthetic offline demo to explore without Yahoo access.")
else:
    saved = st.session_state["run"]
    result, settings = saved["result"], saved["settings"]
    currency = settings["base_currency"]
    metrics = summarize(result, settings["target"])
    st.caption("Displayed results use the last completed run. Apply changed sidebar inputs with Run simulation.")
    if settings["synthetic_data"]:
        st.error("SYNTHETIC DEMO — these are generated histories, not market observations.")
    data_tab, simulation_tab, risk_tab, compare_tab, export_tab = st.tabs(["Data", "Simulation", "Risk Metrics", "Compare", "Export"])
    with data_tab:
        st.write(f"Last observation: {settings['last_observation']} · base currency: {currency}")
        st.dataframe(saved["prices"].tail(20))
        st.line_chart(saved["prices"] / saved["prices"].iloc[0] * 100)
        st.plotly_chart(px.imshow(saved["returns"].corr(), zmin=-1, zmax=1, color_continuous_scale="RdBu_r", text_auto=".2f", title="Base-currency daily log-return correlation"), width="stretch")
        st.caption("Union of observed sessions; exchange holidays forward-filled. Initial missing values removed; no backward fill. Indian and US closes are asynchronous. Converted returns jointly include equity and FX changes; FX is not simulated separately.")
        with st.expander("Native adjusted prices and USD/INR"):
            st.dataframe(saved["raw"].tail(20))
    with simulation_tab:
        if settings["model"] == "Student-t":
            st.warning("Student-t log shocks have no finite exponential moments. Sample means and tail losses may be unstable; use medians/quantiles and stress scenarios.")
        st.plotly_chart(fan_chart(result, currency), width="stretch")
        st.plotly_chart(terminal_chart(result, currency), width="stretch")
    with risk_tab:
        st.dataframe(metrics.to_frame(), width="stretch")
        st.plotly_chart(px.histogram(x=result.max_drawdown, nbins=60, labels={"x": "Maximum drawdown (fraction)"}, title="Cash-flow-neutral maximum drawdowns"), width="stretch")
        st.caption("Probabilities/drawdowns/returns/volatility are fractions. VaR/CVaR are base-currency losses relative to all cash paid in; negative values indicate gains. Annual returns and drawdowns are time-weighted, removing contributions. Sharpe uses arithmetic daily excess returns and realised volatility.")
    with compare_tab:
        comparison_metrics = pd.DataFrame({name: summarize(item, settings["target"]) for name, item in saved["comparisons"].items()})
        st.dataframe(comparison_metrics, width="stretch")
        chart = go.Figure()
        for name, item in saved["comparisons"].items():
            frame = percentile_frame(item)
            chart.add_trace(go.Scatter(x=frame.index, y=frame.P50, name=name))
        st.plotly_chart(chart.update_layout(title="Median portfolio outcomes", yaxis_title=currency), width="stretch")
        st.caption("Comparisons use identical starting capital, monthly contributions, model and horizon. Benchmark is in the same base currency. Marginal Monte Carlo forecasts are compared; these are not paired relative-performance probabilities.")
    with export_tab:
        st.download_button("Terminal paths CSV", terminal_frame(result).to_csv(index=False), "terminal_paths.csv", "text/csv")
        st.download_button("Percentile bands CSV", percentile_frame(result).to_csv(), "percentiles.csv", "text/csv")
        st.download_button("Risk metrics CSV", metrics.to_csv(), "risk_metrics.csv", "text/csv")
        st.download_button("Comparison metrics CSV", comparison_metrics.to_csv(), "comparisons.csv", "text/csv")
        report = html_report(metrics, settings, [fan_chart(result, currency), terminal_chart(result, currency), chart])
        st.download_button("Standalone HTML summary report", report, "monte_carlo_report.html", "text/html")
        st.caption("HTML includes interactive charts and can be printed to PDF in your browser.")
