# Crypto Signal Scanner

A quantitative crypto asset screening and signal generation system that identifies high-potential breakout opportunities across all crypto assets, from Bitcoin to micro-cap meme coins.

## Features

### Multi-Factor Analysis
- **Technical Analysis**: Price patterns, volume analysis, momentum indicators, volatility metrics
- **On-Chain Analysis**: Wallet concentration, whale tracking, smart money flow, liquidity monitoring
- **Market Structure**: Liquidity depth, market cap ratios, exchange distribution
- **Sentiment Analysis**: Social media momentum, news catalysts, community growth
- **Risk Assessment**: Rug pull detection, contract security, liquidity lock verification

### Signal Generation
- Composite scoring system (0-100) with configurable risk profiles
- Buy/sell signals based on multi-factor confirmation
- Real-time monitoring and alerts
- Position sizing recommendations based on conviction level

### Risk Profiles
- **Conservative**: Established coins, strong liquidity, low rug risk
- **Moderate**: Balance between opportunity and safety
- **Aggressive**: Early-stage breakout signals, higher risk tolerance

## Architecture

```
crypto-signal-scanner/
├── src/
│   ├── adapters/          # Data source integrations
│   │   ├── coingecko.py
│   │   ├── dexscreener.py
│   │   ├── onchain.py
│   │   └── sentiment.py
│   ├── analyzers/         # Analysis modules
│   │   ├── technical.py
│   │   ├── onchain.py
│   │   ├── sentiment.py
│   │   └── risk.py
│   ├── scoring/           # Signal scoring engine
│   │   ├── composer.py
│   │   └── profiles.py
│   ├── signals/           # Signal generation
│   │   ├── generator.py
│   │   └── validator.py
│   ├── backtesting/       # Strategy validation
│   │   └── engine.py
│   └── api/               # FastAPI application
│       ├── routes.py
│       └── websocket.py
├── config/
│   ├── conservative.yaml
│   ├── moderate.yaml
│   └── aggressive.yaml
├── data/                  # Local data storage
├── tests/                 # Unit and integration tests
├── scripts/               # Utility scripts
├── docker/                # Docker configurations
└── docs/                  # Documentation

```

## Quick Start

### Prerequisites
- Python 3.11+
- PostgreSQL 14+
- Redis 7+

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/crypto-signal-scanner.git
cd crypto-signal-scanner

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys

# Initialize database
python scripts/init_db.py

# Run the scanner
python -m src.scanner.service
```

### Configuration

Edit `config/moderate.yaml` to customize:
- Data source priorities
- Analysis weights
- Signal thresholds
- Risk tolerances

### API Keys Required

- **CoinGecko**: Free tier available (https://www.coingecko.com/api)
- **Dexscreener**: No API key required
- **Etherscan**: Free API key (https://etherscan.io/apis)
- **Twitter/X** (optional): For sentiment analysis
- **Telegram Bot** (optional): For alerts

## Usage

### Run the Scanner

```bash
# Start the scanner service
python -m src.scanner.service --profile moderate

# Start the web dashboard
python -m src.api.app
```

### Access Dashboard

Open http://localhost:8000 in your browser

### Get Signals via API

```bash
# Get current signals
curl http://localhost:8000/api/signals

# Get specific asset analysis
curl http://localhost:8000/api/analyze/SOL

# Subscribe to WebSocket for real-time updates
wscat -c ws://localhost:8000/ws/signals
```

### Backtesting

```bash
# Run backtest on historical data
python -m src.backtesting.engine --start 2024-01-01 --end 2024-10-01
```

## Signal Interpretation

### Buy Signal
- Composite score > 75 (conservative) / 70 (moderate) / 65 (aggressive)
- At least 3-4 analysis modules in agreement
- Risk metrics within acceptable ranges
- Optimal entry timing (not chasing pumps)

### Sell Signal
- Profit targets reached
- Momentum reversal detected
- Risk indicators deteriorate
- Whale distribution detected

## Alert Configuration

### Telegram Setup

1. Create a bot with @BotFather
2. Get your bot token
3. Add to `.env`: `TELEGRAM_BOT_TOKEN=your_token`
4. Start conversation with your bot
5. Get your chat ID: `/start`

### Discord Setup

1. Create a webhook in your Discord server
2. Add to `.env`: `DISCORD_WEBHOOK_URL=your_webhook_url`

## Development

### Project Structure

- **src/adapters**: Data source integrations with rate limiting and error handling
- **src/analyzers**: Independent analysis modules that score assets 0-100
- **src/scoring**: Composite scoring engine that combines analyzer outputs
- **src/signals**: Signal generation logic with buy/sell rules
- **src/api**: FastAPI application with REST and WebSocket endpoints

### Adding a New Data Source

1. Create adapter in `src/adapters/`
2. Implement `BaseAdapter` interface
3. Add configuration in `config/*.yaml`
4. Update `src/adapters/__init__.py`

### Adding a New Analyzer

1. Create analyzer in `src/analyzers/`
2. Implement `BaseAnalyzer` interface
3. Return score 0-100 with metadata
4. Add weight configuration in profiles

## Performance Metrics

The system tracks:
- Signal accuracy (win rate)
- Average return per signal
- Maximum drawdown
- Sharpe ratio
- Hit rate by asset type

View metrics at http://localhost:8000/metrics

## Security Considerations

- **API Keys**: Never commit `.env` file
- **Rate Limiting**: Respect API limits to avoid bans
- **Data Validation**: All data is validated before use
- **Paper Trading**: Test signals before real capital

## Troubleshooting

### Common Issues

**Rate limiting errors**
- Reduce scan frequency in config
- Use paid API tiers for higher limits

**Missing signals**
- Check analyzer weights in profile config
- Verify data sources are accessible
- Review logs in `data/logs/`

**Slow performance**
- Increase Redis cache TTL
- Reduce number of assets scanned
- Use database indexes

## Roadmap

- [ ] Machine learning signal enhancement
- [ ] Multi-chain support (Solana, Arbitrum, Base)
- [ ] Portfolio optimization module
- [ ] Copy trading integration
- [ ] Mobile app for alerts

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

MIT License - see LICENSE file

## Disclaimer

This software is for educational and research purposes only. Cryptocurrency trading carries significant risk. Never invest more than you can afford to lose. The developers are not responsible for any financial losses.

## Support

- Documentation: https://docs.crypto-signal-scanner.com
- Issues: https://github.com/yourusername/crypto-signal-scanner/issues
- Discord: https://discord.gg/crypto-signals
