"""Streamlit startup and fully offline run smoke test."""
from pathlib import Path
from streamlit.testing.v1 import AppTest


def test_app_offline_run():
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app/main.py"), default_timeout=60).run()
    assert not app.exception
    next(widget for widget in app.checkbox if widget.label == "Synthetic offline demo data").check()
    next(widget for widget in app.number_input if widget.label == "Paths").set_value(1000)
    next(widget for widget in app.button if widget.label == "Run simulation").click().run()
    assert not app.exception
    assert len(app.tabs) == 5
    assert "run" in app.session_state
    assert len(app.session_state["run"]["result"].values) == 253


def test_app_share_comparisons_and_validation():
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app/main.py"), default_timeout=60).run()
    next(widget for widget in app.checkbox if widget.label == "Synthetic offline demo data").check()
    next(widget for widget in app.number_input if widget.label == "Paths").set_value(1000)
    next(widget for widget in app.radio if widget.label == "Position input").set_value("Share counts")
    next(widget for widget in app.selectbox if widget.label == "Compare benchmark").set_value("Nifty 50")
    next(widget for widget in app.text_input if widget.label == "Alternative portfolio weights (optional)").set_value("0.25, 0.75")
    next(widget for widget in app.number_input if widget.label == "Monthly contribution (base currency)").set_value(500.)
    next(widget for widget in app.button if widget.label == "Run simulation").click().run()
    assert not app.exception
    assert len(app.session_state["run"]["comparisons"]) == 3
    assert app.session_state["run"]["result"].invested[-1] > app.session_state["run"]["result"].invested[0]
    next(widget for widget in app.text_input if widget.label == "Tickers").set_value("BAD!")
    next(widget for widget in app.button if widget.label == "Run simulation").click().run()
    assert not app.exception
    assert any("Invalid ticker" in item.value for item in app.error)
