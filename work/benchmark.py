"""Reproducible full-accounting benchmark, excluding download and rendering."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from time import perf_counter
import numpy as np
from core.models import SimulationConfig
from core.portfolio import simulate_portfolio

rng = np.random.default_rng(123)
history = rng.normal(.0003, .012, (1260, 5))
for model in ("GBM", "Student-t", "Bootstrap", "GARCH"):
    start = perf_counter()
    result = simulate_portfolio(history, np.full(5, .2), 100000,
                                SimulationConfig(model=model, paths=10000, days=252, block_size=5),
                                "2026-10-30", "Monthly", 1000)
    duration = perf_counter() - start
    print(f"{model}: {duration:.3f}s, output {result.values.nbytes / 1e6:.1f} MB")
