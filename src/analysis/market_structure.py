"""Market structure scoring focused on liquidity and venue diversity."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from src.models import ModuleResult


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _score_ratio(value: float, target: float) -> float:
    if target <= 0:
        return 0.0

    ratio = min(value / target, 2.0) / 2.0
    return max(0.0, min(1.0, ratio)) * 100


class MarketStructureAnalyzer:
    """Evaluate liquidity depth and venue mix."""

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        self.config = config or {}

    def analyze(self, asset: Dict[str, Any]) -> ModuleResult:
        weights_and_scores: List[Tuple[float, float]] = []
        signals: List[str] = []
        metrics: Dict[str, Any] = {}

        liquidity_cfg = self.config.get("liquidity_metrics", {}) or {}
        if liquidity_cfg:
            weight = float(liquidity_cfg.get("weight", 0.0))
            if weight > 0:
                depth = _safe_float(asset.get("total_volume"))
                target = _safe_float(liquidity_cfg.get("min_depth_usd"), default=50_000)
                depth_score = _score_ratio(depth, max(1.0, target))
                weights_and_scores.append((weight, depth_score))
                metrics["depth_usd"] = depth
                if depth < target:
                    signals.append("liquidity depth below target")

                slippage_target = _safe_float(liquidity_cfg.get("max_slippage_5k"), default=0.05)
                if depth <= 0:
                    slippage_proxy = 1.0
                else:
                    slippage_proxy = min(1.0, 5_000 / (depth + 1))
                metrics["slippage_proxy"] = slippage_proxy
                if slippage_proxy > slippage_target:
                    signals.append("slippage risk elevated")

        venues_cfg = self.config.get("exchange_distribution", {}) or {}
        cex_weight = float(venues_cfg.get("cex_weight", 0.0))
        dex_weight = float(venues_cfg.get("dex_weight", 0.0))

        rank = _safe_float(asset.get("market_cap_rank"), default=1_000)
        if cex_weight > 0:
            # Higher ranked assets are more likely to be listed on CEX venues.
            rank_score = max(0.0, min(1.0, (400 - rank) / 400)) * 100
            weights_and_scores.append((cex_weight, rank_score))
            metrics["market_cap_rank"] = rank
            if rank > 300:
                signals.append("limited tier-1 exchange coverage")

        if dex_weight > 0:
            # Use trading volume as proxy for DEX availability.
            dex_volume = _safe_float(asset.get("total_volume"))
            dex_score = _score_ratio(dex_volume, target=100_000)
            weights_and_scores.append((dex_weight, dex_score))
            metrics["dex_volume_proxy"] = dex_volume
            if dex_volume < 50_000:
                signals.append("dex liquidity thin")

        if not weights_and_scores:
            return ModuleResult(
                name="market_structure",
                score=0.0,
                signals=["market structure data unavailable"],
                metrics=metrics,
            )

        total_weight = sum(weight for weight, _ in weights_and_scores)
        if total_weight <= 0:
            score = 0.0
        else:
            score = sum(weight * value for weight, value in weights_and_scores) / total_weight

        return ModuleResult(
            name="market_structure",
            score=score,
            signals=sorted(set(signals)),
            metrics=metrics,
        )

