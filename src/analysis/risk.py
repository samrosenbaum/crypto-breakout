"""Risk assessment heuristics covering rug pull and volatility factors."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from src.models import ModuleResult


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _inverse_score(value: float, threshold: float) -> float:
    if threshold <= 0:
        return 0.0

    ratio = max(0.0, 1 - (value / threshold))
    return max(0.0, min(1.0, ratio)) * 100


class RiskAnalyzer:
    """Estimate downside risk using available metadata."""

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        self.config = config or {}

    def analyze(self, asset: Dict[str, Any]) -> ModuleResult:
        weights_and_scores: List[Tuple[float, float]] = []
        signals: List[str] = []
        metrics: Dict[str, Any] = {}

        contract_cfg = self.config.get("contract_security", {}) or {}
        if contract_cfg:
            weight = float(contract_cfg.get("audit_weight", 0.0))
            if weight > 0:
                # Use developer engagement as loose proxy for diligence.
                audits_proxy = (asset.get("developer") or {}).get("total_issues") or 0
                score = min(100.0, audits_proxy / 50 * 100)
                weights_and_scores.append((weight, score))
                metrics["audit_activity_proxy"] = audits_proxy
                if audits_proxy == 0 and contract_cfg.get("require_audit"):
                    signals.append("audit status unknown")

            if contract_cfg.get("honeypot_check"):
                signals.append("honeypot screening required")
            if contract_cfg.get("mint_function_check"):
                signals.append("verify mint controls")

        rug_cfg = self.config.get("rug_pull_indicators", {}) or {}
        if rug_cfg:
            liquidity_weight = float(rug_cfg.get("liquidity_lock_weight", 0.0))
            holder_weight = float(rug_cfg.get("holder_distribution_weight", 0.0))
            team_weight = float(rug_cfg.get("team_doxxed_weight", 0.0))
            ownership_weight = float(rug_cfg.get("contract_ownership_weight", 0.0))

            watchlist_users = asset.get("watchlist_portfolio_users") or 0
            metrics["watchlist_users"] = watchlist_users

            if liquidity_weight > 0:
                liquidity_score = min(100.0, watchlist_users / 5_000 * 100)
                weights_and_scores.append((liquidity_weight, liquidity_score))
                if watchlist_users < 500:
                    signals.append("liquidity lock risk")

            if holder_weight > 0:
                community = asset.get("community", {}) or {}
                holder_proxy = community.get("reddit_subscribers") or 0
                distribution_score = min(100.0, holder_proxy / 50_000 * 100)
                weights_and_scores.append((holder_weight, distribution_score))
                metrics["holder_distribution_proxy"] = holder_proxy
                if holder_proxy < 1_000:
                    signals.append("holder distribution concentrated")

            if team_weight > 0:
                github_stars = (asset.get("developer") or {}).get("github_stars") or 0
                doxx_score = min(100.0, github_stars / 1_000 * 100)
                weights_and_scores.append((team_weight, doxx_score))
                metrics["developer_reputation_proxy"] = github_stars

            if ownership_weight > 0:
                ownership_score = min(100.0, watchlist_users / 10_000 * 100)
                weights_and_scores.append((ownership_weight, ownership_score))

        vol_cfg = self.config.get("volatility_metrics", {}) or {}
        if vol_cfg:
            weight = float(vol_cfg.get("weight", 0.0))
            if weight > 0:
                daily_vol = abs(_safe_float(asset.get("price_change_24h"))) / 100
                max_daily = _safe_float(vol_cfg.get("max_daily_volatility"), default=0.5)
                vol_score = _inverse_score(daily_vol, max_daily)
                weights_and_scores.append((weight, vol_score))
                metrics["daily_volatility"] = daily_vol
                if daily_vol > max_daily:
                    signals.append("daily volatility exceeds tolerance")

                drawdown = abs(_safe_float(asset.get("price_change_30d"))) / 100
                tolerance = _safe_float(vol_cfg.get("drawdown_tolerance"), default=0.4)
                drawdown_score = _inverse_score(drawdown, tolerance)
                # Treat drawdown as an equally weighted component within volatility.
                weights_and_scores.append((weight, drawdown_score))
                metrics["drawdown_30d"] = drawdown
                if drawdown > tolerance:
                    signals.append("drawdown beyond profile tolerance")

        if not weights_and_scores:
            return ModuleResult(
                name="risk_assessment",
                score=0.0,
                signals=["risk data unavailable"],
                metrics=metrics,
            )

        total_weight = sum(weight for weight, _ in weights_and_scores)
        if total_weight <= 0:
            score = 0.0
        else:
            score = sum(weight * value for weight, value in weights_and_scores) / total_weight

        return ModuleResult(
            name="risk_assessment",
            score=score,
            signals=sorted(set(signals)),
            metrics=metrics,
        )

