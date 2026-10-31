"""Standalone HTML reporting (no external JavaScript dependency)."""
from html import escape
import json
from collections.abc import Mapping
import pandas as pd
import plotly.graph_objects as go

DISCLAIMER = "Educational tool, not investment advice. Historical data does not guarantee future results."


def html_report(metrics: pd.Series, settings: Mapping, figures: list[go.Figure]) -> str:
    """Export escaped settings, risk measures and interactive embedded charts."""
    charts = "".join(figure.to_html(full_html=False, include_plotlyjs=True if i == 0 else False)
                     for i, figure in enumerate(figures))
    return ("<!doctype html><html><head><meta charset='utf-8'><title>Monte Carlo report</title>"
            "<style>body{font-family:system-ui;max-width:1200px;margin:40px auto;padding:16px}"
            "td,th{padding:6px;text-align:left}pre{white-space:pre-wrap}</style></head><body>"
            f"<h1>Equity Monte Carlo forecast</h1><p>{DISCLAIMER}</p>"
            f"<h2>Run settings</h2><pre>{escape(json.dumps(dict(settings), indent=2, default=str))}</pre>"
            f"<h2>Metrics</h2>{metrics.to_frame().to_html(escape=True)}{charts}</body></html>")
