"""Unit tests for the technical analyzer."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from src.analysis import TechnicalAnalyzer


def build_sample_dataframe(periods: int = 120) -> pd.DataFrame:
    start = datetime(2023, 1, 1)
    index = pd.date_range(start=start, periods=periods, freq="H", tz="UTC")

    base = np.linspace(100, 140, periods)
    noise = np.random.default_rng(42).normal(scale=2.5, size=periods)
    close = base + noise
    open_ = close + np.random.default_rng(43).normal(scale=1.2, size=periods)
    high = np.maximum(open_, close) + np.random.default_rng(44).uniform(0, 2, periods)
    low = np.minimum(open_, close) - np.random.default_rng(45).uniform(0, 2, periods)
    volume = np.abs(np.random.default_rng(46).normal(loc=1_000_000, scale=150_000, size=periods))

    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=index,
    )


def test_technical_analyzer_generates_scores():
    config = {
        "timeframes": ["1h", "4h", "1d"],
        "indicators": {
            "rsi": {"enabled": True, "period": 14, "weight": 0.2},
            "macd": {"enabled": True, "fast": 12, "slow": 26, "signal": 9, "weight": 0.2},
            "bollinger_bands": {"enabled": True, "period": 20, "std_dev": 2, "weight": 0.2},
            "volume_analysis": {"enabled": True, "sma_period": 20, "weight": 0.2},
            "support_resistance": {"enabled": True, "lookback_periods": 60, "weight": 0.2},
        },
    }

    analyzer = TechnicalAnalyzer(config)
    df = build_sample_dataframe()

    results = analyzer.evaluate_timeframes(df)

    assert set(results.keys()) == {"1h", "4h", "1d"}
    for timeframe, result in results.items():
        assert result.composite_score >= 0
        assert result.bias in {"bullish", "bearish", "neutral"}
        assert result.indicator_scores, f"No indicator scores for timeframe {timeframe}"
