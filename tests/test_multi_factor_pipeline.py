"""Tests covering multi-factor analyzers and pipeline integration."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import math

import pytest

yaml = pytest.importorskip("yaml")

from src.analysis import OnChainAnalyzer
from src.pipeline import QuantScanner


def _build_ohlcv_samples(points: int = 240) -> dict:
    base_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
    timestamps = [int((base_time + timedelta(hours=i)).timestamp() * 1000) for i in range(points)]
    prices = [100 + math.sin(i / 12) * 5 + i * 0.05 for i in range(points)]

    return {
        "timestamps": timestamps,
        "open": prices,
        "high": [price * 1.01 for price in prices],
        "low": [price * 0.99 for price in prices],
        "close": [price * 1.002 for price in prices],
        "volume": [150_000 + (i % 24) * 1_000 for i in range(points)],
    }


def test_onchain_analyzer_generates_signals():
    config = {
        "whale_tracking": {
            "enabled": True,
            "whale_threshold": 50_000,
            "accumulation_weight": 0.4,
        },
        "holder_analysis": {
            "enabled": True,
            "concentration_weight": 0.3,
        },
        "transaction_analysis": {
            "enabled": True,
            "velocity_weight": 0.2,
            "smart_money_weight": 0.1,
        },
        "liquidity": {
            "enabled": True,
            "pool_depth_weight": 0.4,
            "lock_status_weight": 0.1,
            "min_depth_usd": 120_000,
        },
    }

    asset = {
        "market_cap": 1_500_000,
        "total_volume": 400_000,
        "price_change_24h": 18,
        "watchlist_portfolio_users": 5_000,
        "developer": {"commit_count_4_weeks": 220},
        "community": {
            "reddit_subscribers": 30_000,
            "telegram_users": 4_500,
        },
    }

    analyzer = OnChainAnalyzer(config)
    result = analyzer.analyze(asset)

    assert result.score > 40
    assert "whale accumulation detected" in result.signals
    assert result.metrics["whale_volume_ratio"] > 0


class _StaticAdapter:
    def __init__(self, asset: dict, ohlcv: dict) -> None:
        self._asset = asset
        self._ohlcv = ohlcv

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def get_asset_data(self, coin_id: str) -> dict:
        return self._asset

    async def get_ohlcv(self, coin_id: str, days: int = 90, vs_currency: str = "usd") -> dict:
        return self._ohlcv


@pytest.mark.asyncio
async def test_quant_scanner_produces_multi_factor_scores(tmp_path):
    config = {
        "profile": {"name": "unit-test"},
        "data_sources": {"price_data": {"primary": "test"}},
        "technical_analysis": {
            "timeframes": ["1h", "4h", "1d"],
            "indicators": {
                "rsi": {"enabled": True, "period": 14, "weight": 0.2},
                "macd": {"enabled": True, "weight": 0.2},
                "bollinger_bands": {"enabled": True, "period": 20, "weight": 0.2},
                "volume_analysis": {"enabled": True, "sma_period": 20, "weight": 0.2},
                "support_resistance": {"enabled": True, "lookback_periods": 60, "weight": 0.2},
            },
        },
        "onchain_analysis": {
            "whale_tracking": {"enabled": True, "whale_threshold": 40_000, "accumulation_weight": 0.3},
            "holder_analysis": {"enabled": True, "concentration_weight": 0.2},
            "transaction_analysis": {
                "enabled": True,
                "velocity_weight": 0.2,
                "smart_money_weight": 0.1,
            },
            "liquidity": {
                "enabled": True,
                "pool_depth_weight": 0.2,
                "lock_status_weight": 0.1,
                "min_depth_usd": 80_000,
            },
        },
        "market_structure": {
            "liquidity_metrics": {"min_depth_usd": 60_000, "weight": 0.6},
            "exchange_distribution": {"cex_weight": 0.2, "dex_weight": 0.2},
        },
        "sentiment_analysis": {
            "twitter": {"enabled": True, "sentiment_weight": 0.3, "engagement_weight": 0.2},
            "reddit": {"enabled": True, "sentiment_weight": 0.2},
            "news": {"enabled": True, "catalyst_weight": 0.3},
        },
        "risk_assessment": {
            "contract_security": {"audit_weight": 0.3, "require_audit": False, "honeypot_check": True},
            "rug_pull_indicators": {
                "liquidity_lock_weight": 0.2,
                "team_doxxed_weight": 0.1,
                "holder_distribution_weight": 0.2,
                "contract_ownership_weight": 0.1,
            },
            "volatility_metrics": {
                "weight": 0.2,
                "max_daily_volatility": 0.6,
                "drawdown_tolerance": 0.5,
            },
        },
        "analyzer_weights": {
            "technical": 0.3,
            "onchain": 0.25,
            "market_structure": 0.2,
            "sentiment": 0.15,
            "risk_assessment": 0.1,
        },
        "scoring": {
            "score_ranges": {
                "high": [70, 100],
                "medium": [40, 69],
                "low": [0, 39],
            }
        },
    }

    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    asset = {
        "id": "test-coin",
        "symbol": "TST",
        "name": "Test Coin",
        "market_cap": 2_000_000,
        "market_cap_rank": 250,
        "total_volume": 250_000,
        "price_usd": 1.25,
        "price_change_24h": 12.0,
        "price_change_30d": -15.0,
        "watchlist_portfolio_users": 10_000,
        "community": {
            "twitter_followers": 320_000,
            "reddit_subscribers": 55_000,
            "reddit_accounts_active_48h": 6_000,
            "telegram_users": 25_000,
        },
        "developer": {
            "commit_count_4_weeks": 180,
            "total_issues": 45,
            "github_stars": 1_500,
        },
        "sentiment_votes_up_percentage": 78,
    }

    adapter = _StaticAdapter(asset, _build_ohlcv_samples())
    scanner = QuantScanner(str(config_path), adapter=adapter)

    results = await scanner.analyze_assets(["test-coin"], days=30)

    assert results, "Scanner should produce at least one signal"
    signal = results[0]
    assert set(signal.modules.keys()) >= {"onchain", "market_structure", "sentiment", "risk_assessment"}
    assert signal.analyzer_breakdown["technical"] >= 0
    assert signal.composite_score >= 0
    assert signal.signals, "Combined signal set should not be empty"

