"""Dataclasses describing analysis and signal outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass(slots=True)
class IndicatorScore:
    """Result of a single indicator evaluation."""

    name: str
    value: float
    score: float
    weight: float
    timeframe: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ModuleResult:
    """Outcome of a non-technical analysis module."""

    name: str
    score: float
    signals: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the module into a JSON-compatible dictionary."""

        return {
            "name": self.name,
            "score": self.score,
            "signals": self.signals,
            "metrics": self.metrics,
        }


@dataclass(slots=True)
class TechnicalAnalysisResult:
    """Aggregate of technical indicators for a timeframe."""

    timeframe: str
    composite_score: float
    indicator_scores: List[IndicatorScore]
    signals: List[str] = field(default_factory=list)
    bias: str = "neutral"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the result to a JSON-compatible dictionary."""

        return {
            "timeframe": self.timeframe,
            "composite_score": self.composite_score,
            "indicator_scores": [
                {
                    "name": indicator.name,
                    "value": indicator.value,
                    "score": indicator.score,
                    "weight": indicator.weight,
                    "timeframe": indicator.timeframe,
                    "metadata": indicator.metadata,
                }
                for indicator in self.indicator_scores
            ],
            "signals": self.signals,
            "bias": self.bias,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class AssetSignal:
    """Composite signal for a digital asset."""

    asset_id: str
    symbol: str
    name: str
    composite_score: float
    confidence: str
    signals: List[str]
    analyzer_breakdown: Dict[str, float]
    technical: Dict[str, TechnicalAnalysisResult] = field(default_factory=dict)
    modules: Dict[str, ModuleResult] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the signal into a JSON-compatible dictionary."""

        return {
            "asset_id": self.asset_id,
            "symbol": self.symbol,
            "name": self.name,
            "composite_score": self.composite_score,
            "confidence": self.confidence,
            "signals": self.signals,
            "analyzer_breakdown": self.analyzer_breakdown,
            "technical": {
                timeframe: result.to_dict()
                for timeframe, result in self.technical.items()
            },
            "modules": {name: module.to_dict() for name, module in self.modules.items()},
            "metadata": self.metadata,
        }
