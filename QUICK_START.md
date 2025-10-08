# Quick Start Guide

Get the crypto signal scanner running in under 10 minutes.

## Prerequisites

- Python 3.11+
- PostgreSQL 14+ (optional for MVP)
- Redis 7+ (optional for MVP)

## Minimal Setup (Testing Only)

```bash
# 1. Install dependencies
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Set up environment
cp .env.example .env
# Edit .env - at minimum, you need:
# - No API keys required for basic testing with CoinGecko free tier

# 3. Test the CoinGecko adapter
python -c "
import asyncio
from src.adapters.coingecko import CoinGeckoAdapter

async def test():
    async with CoinGeckoAdapter() as adapter:
        # Get Bitcoin data
        btc = await adapter.get_asset_data('bitcoin')
        print(f'Bitcoin: ${btc[\"price_usd\"]:,.2f}')
        print(f'Market Cap: ${btc[\"market_cap\"]:,.0f}')
        print(f'24h Change: {btc[\"price_change_24h\"]:.2f}%')

        # Get trending coins
        trending = await adapter.get_trending()
        print(f'\nTrending Coins:')
        for coin in trending[:5]:
            print(f'  {coin[\"symbol\"]}: {coin[\"name\"]}')

asyncio.run(test())
"
```

## Test Run

```bash
# Get top gaining coins
python -m scripts.test_coingecko

# This should output:
# ✅ CoinGecko Connection: OK
# 📈 Top 10 Gainers (24h):
#   1. COIN1: +45.2%
#   2. COIN2: +38.7%
#   ...
```

## Next Steps

1. **Add more adapters**: See `IMPLEMENTATION_GUIDE.md` Phase 2
2. **Build analyzers**: See `IMPLEMENTATION_GUIDE.md` Phase 3
3. **Create signals**: See `IMPLEMENTATION_GUIDE.md` Phase 5

## Project Structure

```
crypto-signal-scanner/
├── src/
│   ├── adapters/          # ✅ DONE: CoinGecko adapter
│   ├── analyzers/         # TODO: Technical, on-chain, sentiment
│   ├── scoring/           # TODO: Composite scoring
│   ├── signals/           # TODO: Signal generation
│   └── api/               # TODO: FastAPI dashboard
├── config/
│   ├── conservative.yaml  # ✅ DONE
│   ├── moderate.yaml      # ✅ DONE
│   └── aggressive.yaml    # ✅ DONE
├── IMPLEMENTATION_GUIDE.md  # ✅ Complete step-by-step guide
└── README.md              # ✅ Full documentation
```

## What's Included

### ✅ Completed
- Project structure and configuration
- Three risk profiles (conservative, moderate, aggressive)
- CoinGecko adapter with full functionality
- Base adapter class for easy extension
- Comprehensive implementation guide
- Environment configuration
- Requirements and dependencies

### 🚧 To Implement (see IMPLEMENTATION_GUIDE.md)
- Dexscreener adapter for DEX data
- On-chain analyzers (Etherscan, etc.)
- Technical analysis module
- Sentiment analysis
- Risk assessment
- Signal generation engine
- FastAPI dashboard
- Alerting system
- Backtesting framework

## Development Workflow

```bash
# 1. Create a new adapter
cp src/adapters/coingecko.py src/adapters/dexscreener.py
# Edit to match Dexscreener API

# 2. Test it
python -m pytest tests/adapters/test_dexscreener.py -v

# 3. Integrate into scanner
# See IMPLEMENTATION_GUIDE.md Phase 2
```

## Common Issues

### Import Errors
```bash
# Make sure you're in the project root and venv is activated
cd /Users/samrosenbaum/crypto-signal-scanner
source venv/bin/activate
export PYTHONPATH=.
```

### TA-Lib Installation Failed
```bash
# macOS
brew install ta-lib

# Ubuntu/Debian
sudo apt-get install ta-lib

# Then reinstall
pip install TA-Lib
```

### Rate Limiting
- Free tier: 50 requests/minute
- Add delays between requests
- Consider upgrading to CoinGecko Pro

## Example: Building Your First Analyzer

```python
# src/analyzers/simple_momentum.py
from .base import BaseAnalyzer, AnalysisResult

class SimpleMomentumAnalyzer(BaseAnalyzer):
    async def analyze(self, asset_data):
        score = 50  # Neutral starting point

        # Positive momentum indicators
        if asset_data["price_change_24h"] > 10:
            score += 20
        if asset_data["volume_24h"] > asset_data.get("avg_volume", 0) * 2:
            score += 15

        # Negative momentum indicators
        if asset_data["price_change_24h"] < -10:
            score -= 20

        return AnalysisResult(
            score=max(0, min(100, score)),
            confidence=0.7,
            signals=[f"24h change: {asset_data['price_change_24h']:.1f}%"],
            metadata=asset_data
        )
```

## Ready to Build?

See `IMPLEMENTATION_GUIDE.md` for the complete week-by-week implementation plan!
