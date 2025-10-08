# Crypto Signal Scanner - Implementation Guide

This guide walks you through implementing each module of the crypto signal scanner system.

## Phase 1: Foundation & Core Infrastructure (Week 1)

### Step 1.1: Set Up Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Install TA-Lib (required for technical analysis)
# macOS:
brew install ta-lib
# Linux:
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make
sudo make install

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys
```

### Step 1.2: Database Setup

```bash
# Install PostgreSQL 14+
# macOS: brew install postgresql@14
# Ubuntu: sudo apt install postgresql-14

# Create database
createdb crypto_signals

# Run migrations (after creating migration files)
alembic upgrade head
```

### Step 1.3: Redis Setup

```bash
# Install Redis
# macOS: brew install redis
# Ubuntu: sudo apt install redis

# Start Redis
redis-server
```

## Phase 2: Data Adapters (Week 1-2)

### Priority Order:
1. **CoinGecko adapter** (price data) - CRITICAL
2. **Dexscreener adapter** (DEX data) - CRITICAL
3. **Etherscan adapter** (on-chain data) - HIGH
4. **Twitter/sentiment adapters** - MEDIUM

### Implementation Pattern for Each Adapter:

```python
# src/adapters/base.py - Base adapter class
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

class BaseAdapter(ABC):
    def __init__(self, api_key: Optional[str] = None, rate_limit: int = 60):
        self.api_key = api_key
        self.rate_limit = rate_limit
        self.client = httpx.AsyncClient(timeout=30.0)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def _make_request(self, url: str, params: Dict = None) -> Dict:
        """Make HTTP request with retry logic."""
        response = await self.client.get(url, params=params)
        response.raise_for_status()
        return response.json()

    @abstractmethod
    async def get_asset_data(self, symbol: str) -> Dict[str, Any]:
        """Fetch data for a specific asset."""
        pass

    @abstractmethod
    async def get_market_data(self, symbols: List[str]) -> List[Dict[str, Any]]:
        """Fetch market data for multiple assets."""
        pass
```

### Example: CoinGecko Adapter

```python
# src/adapters/coingecko.py
from typing import Dict, List, Any
from .base import BaseAdapter

class CoinGeckoAdapter(BaseAdapter):
    BASE_URL = "https://api.coingecko.com/api/v3"

    async def get_asset_data(self, coin_id: str) -> Dict[str, Any]:
        """Get detailed data for a specific coin."""
        url = f"{self.BASE_URL}/coins/{coin_id}"
        params = {
            "localization": "false",
            "tickers": "true",
            "market_data": "true",
            "community_data": "true",
            "developer_data": "true"
        }

        if self.api_key:
            params["x_cg_pro_api_key"] = self.api_key

        data = await self._make_request(url, params)

        return {
            "symbol": data["symbol"].upper(),
            "name": data["name"],
            "price_usd": data["market_data"]["current_price"]["usd"],
            "market_cap": data["market_data"]["market_cap"]["usd"],
            "volume_24h": data["market_data"]["total_volume"]["usd"],
            "price_change_24h": data["market_data"]["price_change_percentage_24h"],
            "price_change_7d": data["market_data"]["price_change_percentage_7d"],
            "circulating_supply": data["market_data"]["circulating_supply"],
            "total_supply": data["market_data"]["total_supply"],
            "ath": data["market_data"]["ath"]["usd"],
            "ath_change_percentage": data["market_data"]["ath_change_percentage"]["usd"],
            "atl": data["market_data"]["atl"]["usd"],
            "atl_change_percentage": data["market_data"]["atl_change_percentage"]["usd"],
            "community": {
                "twitter_followers": data.get("community_data", {}).get("twitter_followers", 0),
                "telegram_users": data.get("community_data", {}).get("telegram_channel_user_count", 0),
                "reddit_subscribers": data.get("community_data", {}).get("reddit_subscribers", 0)
            },
            "developer": {
                "github_stars": data.get("developer_data", {}).get("stars", 0),
                "github_forks": data.get("developer_data", {}).get("forks", 0),
                "total_commits": data.get("developer_data", {}).get("commit_count_4_weeks", 0)
            }
        }

    async def get_trending(self) -> List[Dict[str, Any]]:
        """Get trending coins."""
        url = f"{self.BASE_URL}/search/trending"
        data = await self._make_request(url)
        return [coin["item"] for coin in data["coins"]]

    async def get_top_gainers(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get top gaining coins in last 24h."""
        url = f"{self.BASE_URL}/coins/markets"
        params = {
            "vs_currency": "usd",
            "order": "price_change_percentage_24h_desc",
            "per_page": limit,
            "sparkline": "false"
        }
        data = await self._make_request(url, params)
        return data
```

### Testing Your Adapter:

```python
# tests/adapters/test_coingecko.py
import pytest
from src.adapters.coingecko import CoinGeckoAdapter

@pytest.mark.asyncio
async def test_get_asset_data():
    adapter = CoinGeckoAdapter()
    data = await adapter.get_asset_data("bitcoin")

    assert data["symbol"] == "BTC"
    assert data["price_usd"] > 0
    assert data["market_cap"] > 0

@pytest.mark.asyncio
async def test_get_trending():
    adapter = CoinGeckoAdapter()
    trending = await adapter.get_trending()

    assert len(trending) > 0
    assert "symbol" in trending[0]
```

## Phase 3: Analyzers (Week 2-3)

### Each analyzer follows this structure:

```python
# src/analyzers/base.py
from abc import ABC, abstractmethod
from typing import Dict, Any
from dataclasses import dataclass

@dataclass
class AnalysisResult:
    score: float  # 0-100
    confidence: float  # 0-1
    signals: list[str]
    metadata: Dict[str, Any]

class BaseAnalyzer(ABC):
    def __init__(self, config: Dict):
        self.config = config
        self.weight = config.get("weight", 1.0)

    @abstractmethod
    async def analyze(self, asset_data: Dict[str, Any]) -> AnalysisResult:
        """Analyze asset and return score."""
        pass

    def normalize_score(self, value: float, min_val: float, max_val: float) -> float:
        """Normalize value to 0-100 scale."""
        if max_val == min_val:
            return 50.0
        normalized = ((value - min_val) / (max_val - min_val)) * 100
        return max(0, min(100, normalized))
```

### Example: Technical Analyzer

```python
# src/analyzers/technical.py
import pandas as pd
import talib
from .base import BaseAnalyzer, AnalysisResult

class TechnicalAnalyzer(BaseAnalyzer):
    async def analyze(self, asset_data: Dict[str, Any]) -> AnalysisResult:
        """Perform technical analysis on price data."""
        # Get OHLCV data
        df = asset_data["ohlcv"]  # DataFrame with open, high, low, close, volume

        signals = []
        scores = []

        # RSI Analysis
        rsi = talib.RSI(df["close"], timeperiod=14)
        current_rsi = rsi.iloc[-1]

        if current_rsi < 30:
            signals.append("RSI oversold (bullish)")
            scores.append(80)
        elif current_rsi > 70:
            signals.append("RSI overbought (bearish)")
            scores.append(20)
        else:
            scores.append(50)

        # MACD Analysis
        macd, signal, hist = talib.MACD(df["close"])
        macd_current = hist.iloc[-1]
        macd_prev = hist.iloc[-2]

        if macd_current > 0 and macd_prev <= 0:
            signals.append("MACD bullish crossover")
            scores.append(85)
        elif macd_current < 0 and macd_prev >= 0:
            signals.append("MACD bearish crossover")
            scores.append(15)
        else:
            scores.append(50)

        # Volume Analysis
        vol_sma = df["volume"].rolling(20).mean()
        current_vol = df["volume"].iloc[-1]
        avg_vol = vol_sma.iloc[-1]

        if current_vol > avg_vol * 2:
            signals.append(f"Volume spike: {current_vol/avg_vol:.1f}x average")
            scores.append(75)
        else:
            scores.append(50)

        # Bollinger Bands
        upper, middle, lower = talib.BBANDS(df["close"])
        price = df["close"].iloc[-1]

        if price <= lower.iloc[-1]:
            signals.append("Price at lower Bollinger Band (bullish)")
            scores.append(70)
        elif price >= upper.iloc[-1]:
            signals.append("Price at upper Bollinger Band (bearish)")
            scores.append(30)
        else:
            scores.append(50)

        # Calculate composite score
        composite_score = sum(scores) / len(scores)

        return AnalysisResult(
            score=composite_score,
            confidence=0.8,  # Adjust based on data quality
            signals=signals,
            metadata={
                "rsi": current_rsi,
                "macd_histogram": macd_current,
                "volume_ratio": current_vol / avg_vol if avg_vol > 0 else 0,
                "price_vs_bb_middle": (price - middle.iloc[-1]) / middle.iloc[-1]
            }
        )
```

## Phase 4: Scoring System (Week 3)

```python
# src/scoring/composer.py
from typing import Dict, List
from src.analyzers.base import AnalysisResult

class ScoreComposer:
    def __init__(self, analyzer_weights: Dict[str, float]):
        self.weights = analyzer_weights
        assert abs(sum(analyzer_weights.values()) - 1.0) < 0.01, "Weights must sum to 1.0"

    def compose_score(self, analysis_results: Dict[str, AnalysisResult]) -> Dict:
        """Combine analyzer results into composite score."""
        weighted_score = 0.0
        total_confidence = 0.0
        all_signals = []
        metadata = {}

        for analyzer_name, result in analysis_results.items():
            weight = self.weights.get(analyzer_name, 0)
            weighted_score += result.score * weight
            total_confidence += result.confidence * weight
            all_signals.extend(result.signals)
            metadata[analyzer_name] = {
                "score": result.score,
                "confidence": result.confidence,
                "signals": result.signals,
                "metadata": result.metadata
            }

        return {
            "composite_score": weighted_score,
            "confidence": total_confidence,
            "signals": all_signals,
            "analyzer_breakdown": metadata,
            "grade": self._score_to_grade(weighted_score)
        }

    def _score_to_grade(self, score: float) -> str:
        if score >= 85:
            return "EXCELLENT"
        elif score >= 75:
            return "GOOD"
        elif score >= 65:
            return "MODERATE"
        elif score >= 50:
            return "POOR"
        else:
            return "REJECT"
```

## Phase 5: Signal Generation (Week 4)

```python
# src/signals/generator.py
from typing import Dict, List, Optional
from datetime import datetime

class SignalGenerator:
    def __init__(self, config: Dict):
        self.config = config
        self.buy_threshold = config["signals"]["buy_signal"]["min_composite_score"]
        self.min_modules = config["signals"]["buy_signal"]["min_confirming_modules"]

    def generate_signal(self, asset: str, composite_result: Dict) -> Optional[Dict]:
        """Generate buy/sell signal based on composite score."""
        score = composite_result["composite_score"]
        signals = composite_result["signals"]

        # Count confirming modules
        confirming_count = sum(
            1 for analyzer, data in composite_result["analyzer_breakdown"].items()
            if data["score"] > 60  # Consider >60 as confirming
        )

        if score >= self.buy_threshold and confirming_count >= self.min_modules:
            return {
                "type": "BUY",
                "asset": asset,
                "score": score,
                "confidence": composite_result["confidence"],
                "timestamp": datetime.utcnow().isoformat(),
                "signals": signals,
                "position_size": self._calculate_position_size(score),
                "stop_loss": self._calculate_stop_loss(score),
                "profit_targets": self._calculate_profit_targets(score),
                "rationale": self._generate_rationale(composite_result)
            }

        return None

    def _calculate_position_size(self, score: float) -> float:
        """Calculate position size based on score."""
        tiers = self.config["signals"]["buy_signal"]["position_sizing"]["confidence_tiers"]

        if score >= 85:
            return tiers["high"]
        elif score >= 75:
            return tiers["medium"]
        else:
            return tiers["low"]

    def _calculate_stop_loss(self, score: float) -> float:
        """Calculate stop loss percentage."""
        return self.config["signals"]["sell_signal"]["stop_loss"]["fixed_percent"]

    def _calculate_profit_targets(self, score: float) -> Dict[str, float]:
        """Calculate profit target levels."""
        targets = self.config["signals"]["sell_signal"]["profit_targets"]
        return {
            "conservative": targets["conservative"],
            "moderate": targets["moderate"],
            "aggressive": targets["aggressive"]
        }

    def _generate_rationale(self, composite_result: Dict) -> str:
        """Generate human-readable rationale for signal."""
        signals = composite_result["signals"]
        top_signals = signals[:5]  # Top 5 signals
        return " | ".join(top_signals)
```

## Phase 6: API & Dashboard (Week 4-5)

```python
# src/api/app.py
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict
import asyncio

app = FastAPI(title="Crypto Signal Scanner API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory signal storage (use Redis in production)
active_signals: List[Dict] = []

@app.get("/api/signals")
async def get_signals(profile: str = "moderate", limit: int = 20):
    """Get current active signals."""
    return {
        "signals": active_signals[:limit],
        "count": len(active_signals),
        "profile": profile
    }

@app.get("/api/analyze/{symbol}")
async def analyze_asset(symbol: str, profile: str = "moderate"):
    """Analyze a specific asset on-demand."""
    # This would trigger full analysis pipeline
    # For now, return placeholder
    return {
        "symbol": symbol,
        "status": "analyzing",
        "message": "Analysis in progress..."
    }

@app.websocket("/ws/signals")
async def websocket_signals(websocket: WebSocket):
    """WebSocket endpoint for real-time signal updates."""
    await websocket.accept()

    try:
        while True:
            # Send updates every 5 seconds
            await asyncio.sleep(5)
            await websocket.send_json({
                "type": "signal_update",
                "signals": active_signals[:10],
                "timestamp": datetime.utcnow().isoformat()
            })
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        await websocket.close()

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

## Phase 7: Main Scanner Service

```python
# src/scanner/service.py
import asyncio
from typing import List, Dict
from src.adapters.coingecko import CoinGeckoAdapter
from src.analyzers.technical import TechnicalAnalyzer
from src.scoring.composer import ScoreComposer
from src.signals.generator import SignalGenerator
import yaml

class CryptoScanner:
    def __init__(self, config_path: str = "config/moderate.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        # Initialize components
        self.coingecko = CoinGeckoAdapter()
        self.analyzers = {
            "technical": TechnicalAnalyzer(self.config["technical_analysis"])
        }
        self.composer = ScoreComposer(self.config["analyzer_weights"])
        self.signal_generator = SignalGenerator(self.config)

    async def scan(self) -> List[Dict]:
        """Run full scan and return signals."""
        # Get top coins to analyze
        top_coins = await self.coingecko.get_top_gainers(100)

        signals = []
        for coin in top_coins:
            try:
                # Get detailed data
                asset_data = await self.coingecko.get_asset_data(coin["id"])

                # Run analyzers
                results = {}
                for name, analyzer in self.analyzers.items():
                    results[name] = await analyzer.analyze(asset_data)

                # Compose score
                composite = self.composer.compose_score(results)

                # Generate signal
                signal = self.signal_generator.generate_signal(coin["symbol"], composite)

                if signal:
                    signals.append(signal)

            except Exception as e:
                print(f"Error analyzing {coin['symbol']}: {e}")
                continue

        # Sort by score
        signals.sort(key=lambda x: x["score"], reverse=True)
        return signals

async def main():
    scanner = CryptoScanner()

    while True:
        print("Running scan...")
        signals = await scanner.scan()
        print(f"Found {len(signals)} signals")

        for signal in signals[:5]:  # Print top 5
            print(f"  {signal['asset']}: {signal['score']:.1f} - {signal['rationale']}")

        # Wait before next scan
        await asyncio.sleep(300)  # 5 minutes

if __name__ == "__main__":
    asyncio.run(main())
```

## Next Steps

1. **Week 1**: Set up environment, implement CoinGecko and Dexscreener adapters
2. **Week 2**: Build technical and on-chain analyzers
3. **Week 3**: Implement scoring system and signal generation
4. **Week 4**: Create API and basic dashboard
5. **Week 5**: Add alerting, backtesting, and refinement

## Testing Strategy

```bash
# Run unit tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Test specific module
pytest tests/adapters/test_coingecko.py -v
```

## Deployment

See `docker/README.md` for containerized deployment instructions.

## Getting Help

- Check `docs/` for detailed module documentation
- Review `tests/` for usage examples
- See `scripts/` for utility scripts
