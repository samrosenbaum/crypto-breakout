"""Technical indicator evaluation for digital assets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import structlog

from src.models import IndicatorScore, TechnicalAnalysisResult

logger = structlog.get_logger()


@dataclass(frozen=True)
class _IndicatorConfig:
    enabled: bool
    weight: float


class TechnicalAnalyzer:
    """Evaluate market structure using classical technical indicators."""

    _RESAMPLE_MAP = {
        "5m": "5T",
        "15m": "15T",
        "1h": "1H",
        "4h": "4H",
        "1d": "1D",
    }

    def __init__(self, config: Dict[str, Dict]) -> None:
        self.config = config
        self.timeframes = config.get("timeframes", ["1d"])
        self.indicator_config = self._load_indicator_config(config.get("indicators", {}))

    def evaluate_timeframes(self, data: pd.DataFrame) -> Dict[str, TechnicalAnalysisResult]:
        """Evaluate configured indicators across timeframes."""

        results: Dict[str, TechnicalAnalysisResult] = {}
        for timeframe in self.timeframes:
            prepared = self._prepare_timeframe(data, timeframe)
            if prepared is None or prepared.empty:
                logger.warning(
                    "technical.analyzer.insufficient_data", timeframe=timeframe, bars=len(data)
                )
                continue

            results[timeframe] = self._analyze_single(prepared, timeframe)

        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _load_indicator_config(self, config: Dict[str, Dict]) -> Dict[str, _IndicatorConfig]:
        indicator_config: Dict[str, _IndicatorConfig] = {}
        for name, params in config.items():
            indicator_config[name] = _IndicatorConfig(
                enabled=bool(params.get("enabled", True)),
                weight=float(params.get("weight", 0.0)),
            )
        return indicator_config

    def _prepare_timeframe(self, data: pd.DataFrame, timeframe: str) -> pd.DataFrame | None:
        if timeframe not in self._RESAMPLE_MAP:
            logger.warning("technical.analyzer.unsupported_timeframe", timeframe=timeframe)
            return None

        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be a DatetimeIndex for technical analysis")

        rule = self._RESAMPLE_MAP[timeframe]
        agg_map = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
        }
        if "volume" in data.columns:
            agg_map["volume"] = "sum"

        resampled = data.resample(rule).agg(agg_map).dropna(subset=["open", "high", "low", "close"])

        if "volume" not in resampled.columns:
            resampled["volume"] = np.nan

        return resampled

    def _analyze_single(self, data: pd.DataFrame, timeframe: str) -> TechnicalAnalysisResult:
        indicator_scores: List[IndicatorScore] = []
        signals: List[str] = []

        close = data["close"]
        volume = data["volume"] if "volume" in data.columns else pd.Series(dtype="float64")

        weights_sum = sum(cfg.weight for cfg in self.indicator_config.values() if cfg.enabled)
        composite = 0.0

        if self._is_enabled("rsi"):
            score, indicator, rsi_signal = self._evaluate_rsi(close, timeframe)
            indicator_scores.append(indicator)
            composite += score
            signals.extend(rsi_signal)

        if self._is_enabled("macd"):
            score, indicator, macd_signal = self._evaluate_macd(close, timeframe)
            indicator_scores.append(indicator)
            composite += score
            signals.extend(macd_signal)

        if self._is_enabled("bollinger_bands"):
            score, indicator, bb_signal = self._evaluate_bollinger(close, timeframe)
            indicator_scores.append(indicator)
            composite += score
            signals.extend(bb_signal)

        if self._is_enabled("volume_analysis") and not volume.empty and volume.notna().any():
            score, indicator, vol_signal = self._evaluate_volume(volume, timeframe)
            indicator_scores.append(indicator)
            composite += score
            signals.extend(vol_signal)

        if self._is_enabled("support_resistance"):
            score, indicator, sr_signal = self._evaluate_support_resistance(data, timeframe)
            indicator_scores.append(indicator)
            composite += score
            signals.extend(sr_signal)

        if weights_sum > 0:
            composite_score = composite / weights_sum
        else:
            composite_score = 0.0

        bias = self._bias_from_score(composite_score)

        return TechnicalAnalysisResult(
            timeframe=timeframe,
            composite_score=composite_score,
            indicator_scores=indicator_scores,
            signals=sorted(set(signals)),
            bias=bias,
            metadata={"bars": len(data)},
        )

    def _is_enabled(self, name: str) -> bool:
        cfg = self.indicator_config.get(name, _IndicatorConfig(enabled=False, weight=0))
        return cfg.enabled and cfg.weight > 0

    # Indicator evaluation -------------------------------------------------
    def _evaluate_rsi(
        self, close: pd.Series, timeframe: str
    ) -> Tuple[float, IndicatorScore, List[str]]:
        params = self.config["indicators"].get("rsi", {})
        period = int(params.get("period", 14))
        oversold = float(params.get("oversold", 30))
        overbought = float(params.get("overbought", 70))
        weight = float(params.get("weight", 0))

        if len(close) < period + 1:
            rsi_value = np.nan
        else:
            delta = close.diff()
            gains = delta.clip(lower=0)
            losses = -delta.clip(upper=0)
            avg_gain = gains.ewm(alpha=1 / period, adjust=False).mean()
            avg_loss = losses.ewm(alpha=1 / period, adjust=False).mean()
            rs = avg_gain / avg_loss.replace({0: np.nan})
            rsi = 100 - (100 / (1 + rs))
            rsi_value = float(rsi.iloc[-1])

        score = self._score_rsi(rsi_value, oversold, overbought) * weight

        signals: List[str] = []
        if not np.isnan(rsi_value):
            if rsi_value < oversold:
                signals.append(f"{timeframe} RSI oversold")
            elif rsi_value > overbought:
                signals.append(f"{timeframe} RSI overbought")

        indicator = IndicatorScore(
            name="RSI",
            value=float(rsi_value) if not np.isnan(rsi_value) else float("nan"),
            score=score / weight if weight else 0.0,
            weight=weight,
            timeframe=timeframe,
            metadata={"period": period, "oversold": oversold, "overbought": overbought},
        )

        return score, indicator, signals

    def _score_rsi(self, value: float, oversold: float, overbought: float) -> float:
        if np.isnan(value):
            return 0.0

        if value < oversold:
            return min(100.0, 100 - ((oversold - value) / oversold) * 20)
        if value > overbought:
            return max(0.0, 100 - ((value - overbought) / (100 - overbought)) * 100)

        distance = abs(value - 50)
        return max(0.0, 100 - (distance / 50) * 100)

    def _evaluate_macd(
        self, close: pd.Series, timeframe: str
    ) -> Tuple[float, IndicatorScore, List[str]]:
        params = self.config["indicators"].get("macd", {})
        fast = int(params.get("fast", 12))
        slow = int(params.get("slow", 26))
        signal = int(params.get("signal", 9))
        weight = float(params.get("weight", 0))

        if len(close) < slow + signal:
            macd_value = signal_value = hist_value = np.nan
        else:
            ema_fast = close.ewm(span=fast, adjust=False).mean()
            ema_slow = close.ewm(span=slow, adjust=False).mean()
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=signal, adjust=False).mean()
            hist = macd_line - signal_line
            macd_value = float(macd_line.iloc[-1])
            signal_value = float(signal_line.iloc[-1])
            hist_value = float(hist.iloc[-1])

        score = self._score_macd(hist_value) * weight

        signals: List[str] = []
        if not np.isnan(hist_value):
            if hist_value > 0:
                signals.append(f"{timeframe} MACD bullish")
            elif hist_value < 0:
                signals.append(f"{timeframe} MACD bearish")

        indicator = IndicatorScore(
            name="MACD",
            value=macd_value if not np.isnan(macd_value) else float("nan"),
            score=score / weight if weight else 0.0,
            weight=weight,
            timeframe=timeframe,
            metadata={
                "fast": fast,
                "slow": slow,
                "signal": signal,
                "signal_value": signal_value,
                "histogram": hist_value,
            },
        )

        return score, indicator, signals

    def _score_macd(self, hist_value: float) -> float:
        if np.isnan(hist_value):
            return 0.0

        # Normalize the histogram by using an arctangent to bound values.
        normalized = (np.arctan(hist_value) / (np.pi / 2) + 1) / 2
        return normalized * 100

    def _evaluate_bollinger(
        self, close: pd.Series, timeframe: str
    ) -> Tuple[float, IndicatorScore, List[str]]:
        params = self.config["indicators"].get("bollinger_bands", {})
        period = int(params.get("period", 20))
        std_dev = float(params.get("std_dev", 2))
        weight = float(params.get("weight", 0))

        if len(close) < period:
            bb_position = np.nan
            upper = lower = middle = np.nan
        else:
            middle = close.rolling(window=period).mean()
            std = close.rolling(window=period).std(ddof=0)
            upper = middle + std_dev * std
            lower = middle - std_dev * std
            bb_position = (close - lower) / (upper - lower)
            bb_position = float(bb_position.iloc[-1])
            upper = float(upper.iloc[-1])
            lower = float(lower.iloc[-1])
            middle = float(middle.iloc[-1])

        score = self._score_bollinger(bb_position) * weight

        signals: List[str] = []
        if not np.isnan(bb_position):
            if bb_position < 0.1:
                signals.append(f"{timeframe} price near lower band")
            elif bb_position > 0.9:
                signals.append(f"{timeframe} price near upper band")

        indicator = IndicatorScore(
            name="Bollinger Bands",
            value=bb_position if not np.isnan(bb_position) else float("nan"),
            score=score / weight if weight else 0.0,
            weight=weight,
            timeframe=timeframe,
            metadata={
                "period": period,
                "std_dev": std_dev,
                "upper": upper,
                "lower": lower,
                "middle": middle,
            },
        )

        return score, indicator, signals

    def _score_bollinger(self, position: float) -> float:
        if np.isnan(position):
            return 0.0

        # Favor prices near the lower band (potential entries) and penalize
        # extreme upper band moves to avoid chasing pumps.
        if position <= 0.5:
            return (1 - position) * 100
        return max(0.0, (1 - position) * 60)

    def _evaluate_volume(
        self, volume: pd.Series, timeframe: str
    ) -> Tuple[float, IndicatorScore, List[str]]:
        params = self.config["indicators"].get("volume_analysis", {})
        period = int(params.get("sma_period", 20))
        spike_threshold = float(params.get("spike_threshold", 2.0))
        weight = float(params.get("weight", 0))

        if len(volume.dropna()) < period:
            ratio = np.nan
        else:
            sma = volume.rolling(window=period).mean()
            ratio = float((volume.iloc[-1] / sma.iloc[-1]) if sma.iloc[-1] else np.nan)

        score = self._score_volume(ratio, spike_threshold) * weight

        signals: List[str] = []
        if not np.isnan(ratio):
            if ratio >= spike_threshold:
                signals.append(f"{timeframe} volume breakout")
            elif ratio < 0.5:
                signals.append(f"{timeframe} volume drought")

        indicator = IndicatorScore(
            name="Volume Surge",
            value=ratio if not np.isnan(ratio) else float("nan"),
            score=score / weight if weight else 0.0,
            weight=weight,
            timeframe=timeframe,
            metadata={"period": period, "spike_threshold": spike_threshold},
        )

        return score, indicator, signals

    def _score_volume(self, ratio: float, threshold: float) -> float:
        if np.isnan(ratio):
            return 0.0

        if ratio >= threshold:
            return min(100.0, (ratio / threshold) * 100)

        return max(0.0, (ratio / threshold) * 60)

    def _evaluate_support_resistance(
        self, data: pd.DataFrame, timeframe: str
    ) -> Tuple[float, IndicatorScore, List[str]]:
        params = self.config["indicators"].get("support_resistance", {})
        lookback = int(params.get("lookback_periods", 100))
        weight = float(params.get("weight", 0))

        window = data.tail(lookback)
        if window.empty:
            return 0.0, IndicatorScore("Support/Resistance", float("nan"), 0.0, weight, timeframe), []

        recent_low = float(window["low"].min())
        recent_high = float(window["high"].max())
        close = float(window["close"].iloc[-1])

        support_distance = (close - recent_low) / recent_low if recent_low else np.nan
        resistance_distance = (recent_high - close) / recent_high if recent_high else np.nan

        score = self._score_support_resistance(support_distance, resistance_distance) * weight

        signals: List[str] = []
        if not np.isnan(support_distance) and support_distance < 0.05:
            signals.append(f"{timeframe} near support")
        if not np.isnan(resistance_distance) and resistance_distance < 0.05:
            signals.append(f"{timeframe} near resistance")

        indicator = IndicatorScore(
            name="Support/Resistance",
            value=close,
            score=score / weight if weight else 0.0,
            weight=weight,
            timeframe=timeframe,
            metadata={
                "recent_low": recent_low,
                "recent_high": recent_high,
                "support_distance": support_distance,
                "resistance_distance": resistance_distance,
            },
        )

        return score, indicator, signals

    def _score_support_resistance(self, support_dist: float, resistance_dist: float) -> float:
        if np.isnan(support_dist) or np.isnan(resistance_dist):
            return 0.0

        # Favor entries closer to support with adequate room to resistance.
        buffer = resistance_dist + support_dist
        if buffer == 0:
            return 50.0

        skew = resistance_dist - support_dist
        normalized = (skew / buffer + 1) / 2  # Map to 0-1
        return normalized * 100

    def _bias_from_score(self, score: float) -> str:
        if score >= 65:
            return "bullish"
        if score <= 35:
            return "bearish"
        return "neutral"
