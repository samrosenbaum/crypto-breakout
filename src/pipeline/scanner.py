"""Quantitative crypto asset screening pipeline."""

from __future__ import annotations

import asyncio
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import structlog

from src.analysis import (
    MarketStructureAnalyzer,
    OnChainAnalyzer,
    RiskAnalyzer,
    SentimentAnalyzer,
    TechnicalAnalyzer,
)
from src.adapters import BaseAdapter, get_adapter
from src.config_loader import ConfigLoader
from src.models import AssetSignal, ModuleResult, TechnicalAnalysisResult

logger = structlog.get_logger()


class QuantScanner:
    """Coordinate data ingestion, analysis, and signal generation."""

    def __init__(self, config_path: str, adapter: Optional[BaseAdapter] = None) -> None:
        self.config_loader = ConfigLoader(config_path)
        self.config = self.config_loader.load()
        self.adapter = adapter

        price_source = self.config.get("data_sources", {}).get("price_data", {})
        self.adapter_name = price_source.get("primary", "coingecko")
        adapter_kwargs = {}
        if api_key := price_source.get("api_key"):
            adapter_kwargs["api_key"] = api_key
        self.adapter_kwargs = adapter_kwargs

        technical_config = self.config.get("technical_analysis", {})
        self.technical_analyzer = TechnicalAnalyzer(technical_config)
        self.onchain_analyzer = OnChainAnalyzer(self.config.get("onchain_analysis"))
        self.market_structure_analyzer = MarketStructureAnalyzer(
            self.config.get("market_structure")
        )
        self.sentiment_analyzer = SentimentAnalyzer(self.config.get("sentiment_analysis"))
        self.risk_analyzer = RiskAnalyzer(self.config.get("risk_assessment"))
        self.analyzer_weights = self.config.get("analyzer_weights", {"technical": 1.0})
        self.score_ranges = self.config.get("scoring", {}).get("score_ranges", {})

    async def analyze_assets(self, coin_ids: Iterable[str], days: int = 90) -> List[AssetSignal]:
        """Run the full quant pipeline for the provided assets."""

        coin_ids = [coin_id for coin_id in coin_ids if coin_id]
        if not coin_ids:
            return []

        adapter = self.adapter or get_adapter(self.adapter_name, **self.adapter_kwargs)

        if self.adapter is None:
            async with adapter as managed_adapter:
                return await self._analyze_with_adapter(managed_adapter, coin_ids, days)

        return await self._analyze_with_adapter(adapter, coin_ids, days)

    async def _analyze_with_adapter(
        self, adapter: BaseAdapter, coin_ids: Iterable[str], days: int
    ) -> List[AssetSignal]:
        tasks = [self._analyze_asset(adapter, coin_id, days) for coin_id in coin_ids]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        signals: List[AssetSignal] = []
        for coin_id, result in zip(coin_ids, results):
            if isinstance(result, Exception):
                logger.error("quant_scanner.analysis_failed", asset=coin_id, error=str(result))
                continue
            if result is None:
                continue
            signals.append(result)

        signals.sort(key=lambda item: item.composite_score, reverse=True)
        return signals

    async def _analyze_asset(
        self, adapter: BaseAdapter, coin_id: str, days: int
    ) -> Optional[AssetSignal]:
        try:
            asset = await adapter.get_asset_data(coin_id)
            ohlcv = await adapter.get_ohlcv(coin_id, days=days)
        except Exception as exc:  # noqa: BLE001 - We want to bubble the error with context
            logger.error("quant_scanner.data_fetch_failed", asset=coin_id, error=str(exc))
            return None

        frame = self._ohlcv_to_dataframe(ohlcv)
        if frame.empty:
            logger.warning("quant_scanner.empty_market_data", asset=coin_id)
            return None

        technical_results = self.technical_analyzer.evaluate_timeframes(frame)
        if not technical_results:
            logger.warning("quant_scanner.no_technical_signal", asset=coin_id)
            return None

        module_scores: Dict[str, float] = {}
        technical_score = self._aggregate_timeframe_results(technical_results)
        if technical_score is not None:
            module_scores["technical"] = technical_score

        module_results = {}
        for name, analyzer in (
            ("onchain", self.onchain_analyzer),
            ("market_structure", self.market_structure_analyzer),
            ("sentiment", self.sentiment_analyzer),
            ("risk_assessment", self.risk_analyzer),
        ):
            result = self._evaluate_module(analyzer, asset)
            if result is None:
                continue
            module_results[name] = result
            module_scores[name] = result.score

        composite_score = self._combine_module_scores(module_scores)
        confidence = self._score_to_label(composite_score)
        aggregated_signals = self._collect_signals(technical_results, module_results)

        analyzer_breakdown = {
            name: module_scores.get(name, 0.0) for name in self.analyzer_weights.keys()
        }

        metadata = {
            "asset": {
                "market_cap": asset.get("market_cap"),
                "market_cap_rank": asset.get("market_cap_rank"),
                "total_volume": asset.get("total_volume"),
                "price_usd": asset.get("price_usd"),
            },
            "risk_profile": self.config.get("profile", {}).get("name"),
        }

        return AssetSignal(
            asset_id=asset.get("id", coin_id),
            symbol=asset.get("symbol", coin_id.upper()),
            name=asset.get("name", coin_id.title()),
            composite_score=composite_score,
            confidence=confidence,
            signals=aggregated_signals,
            analyzer_breakdown=analyzer_breakdown,
            technical=technical_results,
            modules=module_results,
            metadata=metadata,
        )

    def _evaluate_module(self, analyzer, asset: Dict[str, Any]) -> Optional[ModuleResult]:
        if analyzer is None:
            return None

        try:
            result = analyzer.analyze(asset)
        except Exception as exc:  # noqa: BLE001 - provide context upstream
            logger.error("quant_scanner.module_failed", module=analyzer.__class__.__name__, error=str(exc))
            return None

        if not isinstance(result, ModuleResult):
            return None

        return result

    def _ohlcv_to_dataframe(self, data: Dict[str, List]) -> pd.DataFrame:
        if not data:
            return pd.DataFrame()

        timestamps = data.get("timestamps")
        if not timestamps:
            return pd.DataFrame()

        frame = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(data.get("timestamps"), unit="ms", utc=True),
                "open": pd.to_numeric(data.get("open", []), errors="coerce"),
                "high": pd.to_numeric(data.get("high", []), errors="coerce"),
                "low": pd.to_numeric(data.get("low", []), errors="coerce"),
                "close": pd.to_numeric(data.get("close", []), errors="coerce"),
            }
        )

        if "volume" in data:
            frame["volume"] = pd.to_numeric(data.get("volume", []), errors="coerce")

        frame = frame.set_index("timestamp").sort_index()
        return frame.dropna(subset=["open", "high", "low", "close"])

    def _aggregate_timeframe_results(
        self, results: Dict[str, TechnicalAnalysisResult]
    ) -> Optional[float]:
        if not results:
            return None

        frames = [tf for tf in self.technical_analyzer.timeframes if tf in results]
        if not frames:
            frames = list(results.keys())

        weights = np.linspace(1, len(frames), len(frames))
        weights = weights / weights.sum()

        aggregated = 0.0
        for weight, timeframe in zip(weights, frames, strict=False):
            aggregated += results[timeframe].composite_score * weight

        return aggregated

    def _combine_module_scores(self, module_scores: Dict[str, float]) -> float:
        weighted = 0.0
        total_weight = 0.0

        for name, score in module_scores.items():
            weight = float(self.analyzer_weights.get(name, 0.0))
            if weight <= 0:
                continue
            weighted += score * weight
            total_weight += weight

        if total_weight == 0:
            return 0.0

        return weighted / total_weight

    def _collect_signals(
        self,
        technical_results: Dict[str, TechnicalAnalysisResult],
        module_results: Dict[str, ModuleResult],
    ) -> List[str]:
        aggregated: List[str] = []
        for timeframe, result in technical_results.items():
            aggregated.extend(result.signals or [f"{timeframe} neutral"])
        for name, result in module_results.items():
            aggregated.extend(result.signals or [f"{name} neutral"])
        return sorted(set(aggregated))

    def _score_to_label(self, score: float) -> str:
        if not self.score_ranges:
            return "unclassified"

        for label, bounds in self.score_ranges.items():
            if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
                continue
            lower, upper = bounds
            if lower <= score <= upper:
                return label

        # If the score is outside declared bands, pick the closest range.
        closest_label = "unclassified"
        smallest_distance = float("inf")
        for label, bounds in self.score_ranges.items():
            if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
                continue
            lower, upper = bounds
            if score < lower:
                distance = lower - score
            else:
                distance = score - upper
            if distance < smallest_distance:
                smallest_distance = distance
                closest_label = label

        return closest_label
