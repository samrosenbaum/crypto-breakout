"""Heuristic on-chain analytics leveraging available market metadata."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import math

from src.models import ModuleResult


def _safe_ratio(numerator: Any, denominator: Any) -> float:
    try:
        num = float(numerator)
        den = float(denominator)
    except (TypeError, ValueError):
        return 0.0

    if den <= 0:
        return 0.0

    return max(0.0, num) / den


def _scale_ratio(value: float, target: float) -> float:
    if target <= 0:
        return 0.0

    # Clamp to twice the target so extreme outliers do not dominate scoring.
    scaled = min(value / target, 2.0) / 2.0
    return max(0.0, min(1.0, scaled)) * 100


def _log_scaling(value: float, pivot: float) -> float:
    if value is None or value <= 0 or pivot <= 0:
        return 0.0

    numerator = math.log10(value + 1)
    denominator = math.log10(pivot + 1)
    if denominator == 0:
        return 0.0

    return max(0.0, min(1.0, numerator / denominator)) * 100


class OnChainAnalyzer:
    """Score on-chain health using lightweight heuristics."""

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        self.config = config or {}

    def analyze(self, asset: Dict[str, Any]) -> ModuleResult:
        weights_and_scores: List[Tuple[float, float]] = []
        signals: List[str] = []
        metrics: Dict[str, Any] = {}

        market_cap = asset.get("market_cap")
        total_volume = asset.get("total_volume")
        watchlist_users = asset.get("watchlist_portfolio_users")
        community = asset.get("community", {}) or {}

        whale_cfg = self.config.get("whale_tracking", {}) or {}
        if whale_cfg.get("enabled"):
            weight = float(whale_cfg.get("accumulation_weight", 0.0))
            if weight > 0:
                ratio = _safe_ratio(total_volume, market_cap)
                threshold = float(whale_cfg.get("whale_threshold", 100_000))
                # Interpret threshold relative to market cap – higher volume implies accumulation.
                normalized = _scale_ratio(total_volume or 0.0, threshold)
                weights_and_scores.append((weight, normalized))
                metrics["whale_volume_ratio"] = ratio
                if ratio >= 0.15:
                    signals.append("whale accumulation detected")
                elif ratio <= 0.02 and total_volume:
                    signals.append("whale activity muted")

        holder_cfg = self.config.get("holder_analysis", {}) or {}
        if holder_cfg.get("enabled"):
            weight = float(holder_cfg.get("concentration_weight", 0.0))
            if weight > 0:
                holders_proxy = community.get("reddit_subscribers") or watchlist_users or 0
                normalized = _log_scaling(float(holders_proxy), pivot=50_000)
                weights_and_scores.append((weight, normalized))
                metrics["holder_reach_proxy"] = holders_proxy
                if holders_proxy and holders_proxy < 2_000:
                    signals.append("holder base concentrated")
                elif holders_proxy > 25_000:
                    signals.append("healthy holder dispersion")

        txn_cfg = self.config.get("transaction_analysis", {}) or {}
        if txn_cfg.get("enabled"):
            velocity_weight = float(txn_cfg.get("velocity_weight", 0.0))
            smart_money_weight = float(txn_cfg.get("smart_money_weight", 0.0))

            price_change_24h = asset.get("price_change_24h")
            if velocity_weight > 0:
                # Reward steady positive velocity while penalising strong drawdowns.
                change = 0.0 if price_change_24h is None else float(price_change_24h)
                normalized = max(0.0, min(1.0, (change + 20.0) / 40.0)) * 100
                weights_and_scores.append((velocity_weight, normalized))
                metrics["price_change_24h"] = change
                if change >= 15:
                    signals.append("on-chain velocity accelerating")
                elif change <= -10:
                    signals.append("on-chain outflows growing")

            if smart_money_weight > 0:
                commits = (asset.get("developer") or {}).get("commit_count_4_weeks")
                smart_score = _log_scaling(float(commits or 0), pivot=200)
                weights_and_scores.append((smart_money_weight, smart_score))
                metrics["dev_commit_4w"] = commits or 0
                if commits and commits > 100:
                    signals.append("developer activity attracting smart money")

        liquidity_cfg = self.config.get("liquidity", {}) or {}
        if liquidity_cfg.get("enabled"):
            pool_weight = float(liquidity_cfg.get("pool_depth_weight", 0.0))
            lock_weight = float(liquidity_cfg.get("lock_status_weight", 0.0))

            if pool_weight > 0:
                min_depth = float(liquidity_cfg.get("min_depth_usd", 100_000))
                depth_score = _scale_ratio(total_volume or 0.0, max(1.0, min_depth))
                weights_and_scores.append((pool_weight, depth_score))
                metrics["liquidity_depth_usd"] = total_volume or 0.0
                if total_volume and total_volume < min_depth:
                    signals.append("liquidity shallow across pools")

            if lock_weight > 0:
                # Without explicit lock data we approximate from watchlist growth.
                lock_proxy = community.get("telegram_users") or watchlist_users or 0
                lock_score = _log_scaling(float(lock_proxy), pivot=10_000)
                weights_and_scores.append((lock_weight, lock_score))
                metrics["liquidity_lock_proxy"] = lock_proxy
                if lock_proxy and lock_proxy < 1_000:
                    signals.append("liquidity lock unverified")

        if not weights_and_scores:
            return ModuleResult(name="onchain", score=0.0, signals=["insufficient on-chain data"], metrics=metrics)

        total_weight = sum(weight for weight, _ in weights_and_scores)
        if total_weight <= 0:
            score = 0.0
        else:
            score = sum(weight * value for weight, value in weights_and_scores) / total_weight

        return ModuleResult(name="onchain", score=score, signals=sorted(set(signals)), metrics=metrics)

