"""Social and news sentiment heuristics."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from src.models import ModuleResult


def _log_score(value: Any, pivot: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0

    if numeric <= 0 or pivot <= 0:
        return 0.0

    import math

    numerator = math.log10(numeric + 1)
    denominator = math.log10(pivot + 1)
    if denominator == 0:
        return 0.0

    return max(0.0, min(1.0, numerator / denominator)) * 100


class SentimentAnalyzer:
    """Evaluate community engagement and news catalysts."""

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        self.config = config or {}

    def analyze(self, asset: Dict[str, Any]) -> ModuleResult:
        weights_and_scores: List[Tuple[float, float]] = []
        signals: List[str] = []
        metrics: Dict[str, Any] = {}

        community = asset.get("community", {}) or {}
        sentiment_votes = asset.get("sentiment_votes_up_percentage")

        twitter_cfg = self.config.get("twitter", {}) or {}
        if twitter_cfg.get("enabled"):
            weight = float(twitter_cfg.get("sentiment_weight", 0.0))
            engagement_weight = float(twitter_cfg.get("engagement_weight", 0.0))
            followers = community.get("twitter_followers") or 0
            mentions_target = float(twitter_cfg.get("min_mentions", 50))

            if weight > 0:
                sentiment_score = _log_score(followers, pivot=200_000)
                weights_and_scores.append((weight, sentiment_score))
                metrics["twitter_followers"] = followers
                if followers < mentions_target * 50:
                    signals.append("twitter traction light")
                elif followers > 250_000:
                    signals.append("twitter buzz elevated")

            if engagement_weight > 0:
                engagement_score = _log_score(followers, pivot=500_000)
                weights_and_scores.append((engagement_weight, engagement_score))

        reddit_cfg = self.config.get("reddit", {}) or {}
        if reddit_cfg.get("enabled"):
            weight = float(reddit_cfg.get("sentiment_weight", 0.0))
            subscribers = community.get("reddit_subscribers") or 0
            active = community.get("reddit_accounts_active_48h") or 0
            metrics["reddit_subscribers"] = subscribers
            metrics["reddit_active_48h"] = active

            if weight > 0:
                reddit_score = _log_score(subscribers + active * 10, pivot=150_000)
                weights_and_scores.append((weight, reddit_score))
                if active and active / max(subscribers, 1) > 0.05:
                    signals.append("reddit discussions trending")

        news_cfg = self.config.get("news", {}) or {}
        if news_cfg.get("enabled"):
            weight = float(news_cfg.get("catalyst_weight", 0.0))
            if weight > 0:
                up_votes = 0.0 if sentiment_votes is None else float(sentiment_votes)
                catalyst_score = max(0.0, min(1.0, (up_votes - 40) / 60.0)) * 100
                weights_and_scores.append((weight, catalyst_score))
                metrics["sentiment_votes_up_percentage"] = up_votes
                if up_votes >= 75:
                    signals.append("news catalysts positive")
                elif up_votes <= 45:
                    signals.append("news sentiment cautious")

        if not weights_and_scores:
            return ModuleResult(
                name="sentiment",
                score=0.0,
                signals=["sentiment data unavailable"],
                metrics=metrics,
            )

        total_weight = sum(weight for weight, _ in weights_and_scores)
        if total_weight <= 0:
            score = 0.0
        else:
            score = sum(weight * value for weight, value in weights_and_scores) / total_weight

        return ModuleResult(
            name="sentiment",
            score=score,
            signals=sorted(set(signals)),
            metrics=metrics,
        )

