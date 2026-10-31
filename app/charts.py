"""Plotly chart builders shared by the UI and standalone HTML report."""
import plotly.graph_objects as go
from core.metrics import percentile_frame, value_at_risk
from core.portfolio import PortfolioResult


def fan_chart(result: PortfolioResult, currency: str) -> go.Figure:
    """Plot percentile bands and a bounded sample of paths."""
    frame = percentile_frame(result)
    figure = go.Figure()
    for path in range(min(25, result.values.shape[1])):
        figure.add_trace(go.Scatter(x=result.dates, y=result.values[:, path], mode="lines",
                                   line=dict(color="rgba(100,120,140,.12)", width=1), showlegend=False))
    for lower, upper, color, name in [("P05", "P95", "rgba(50,120,220,.15)", "5–95%"),
                                      ("P25", "P75", "rgba(50,120,220,.3)", "25–75%")]:
        figure.add_trace(go.Scatter(x=frame.index, y=frame[lower], line=dict(width=0), showlegend=False))
        figure.add_trace(go.Scatter(x=frame.index, y=frame[upper], fill="tonexty", fillcolor=color,
                                   line=dict(width=0), name=name))
    figure.add_trace(go.Scatter(x=frame.index, y=frame.P50, name="Median", line=dict(color="#2563eb")))
    figure.add_trace(go.Scatter(x=frame.index, y=frame["Paid in"], name="Paid in", line=dict(dash="dash")))
    return figure.update_layout(title="Simulated portfolio outcomes", yaxis_title=currency, xaxis_title="Approximate future business dates")


def terminal_chart(result: PortfolioResult, currency: str) -> go.Figure:
    """Terminal histogram with mean, median and monetary 95% VaR threshold."""
    import numpy as np
    terminal = result.values[-1]
    var, _ = value_at_risk(result.invested[-1] - terminal)
    figure = go.Figure(go.Histogram(x=terminal, nbinsx=70, name="Paths"))
    for value, name, color in [(terminal.mean(), "Mean", "green"), (np.median(terminal), "Median", "blue"),
                                (result.invested[-1] - var, "95% VaR terminal threshold", "red")]:
        figure.add_vline(x=value, line_color=color, annotation_text=name, annotation_textangle=-90)
    return figure.update_layout(title="Terminal value distribution", xaxis_title=currency, yaxis_title="Paths")
