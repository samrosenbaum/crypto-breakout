"""CoinGecko API adapter for cryptocurrency market data."""

from typing import Dict, List, Any, Optional
from .base import BaseAdapter
import structlog

logger = structlog.get_logger()


class CoinGeckoAdapter(BaseAdapter):
    """Adapter for CoinGecko API - primary source for price and market data."""

    BASE_URL = "https://api.coingecko.com/api/v3"
    PRO_URL = "https://pro-api.coingecko.com/api/v3"

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize CoinGecko adapter.

        Args:
            api_key: CoinGecko Pro API key (optional, uses free tier if not provided)
        """
        super().__init__(api_key=api_key, rate_limit=50 if not api_key else 500)
        self.base_url = self.PRO_URL if api_key else self.BASE_URL
        self.logger = logger.bind(adapter="CoinGecko", tier="pro" if api_key else "free")

    async def get_asset_data(self, coin_id: str) -> Dict[str, Any]:
        """
        Get comprehensive data for a specific cryptocurrency.

        Args:
            coin_id: CoinGecko coin ID (e.g., 'bitcoin', 'ethereum')

        Returns:
            Dictionary with price, market cap, volume, community, and developer data
        """
        url = f"{self.base_url}/coins/{coin_id}"
        params = {
            "localization": "false",
            "tickers": "true",
            "market_data": "true",
            "community_data": "true",
            "developer_data": "true",
            "sparkline": "false",
        }

        if self.api_key:
            params["x_cg_pro_api_key"] = self.api_key

        data = await self._make_request(url, params)

        # Transform to standardized format
        market_data = data.get("market_data", {})
        community_data = data.get("community_data", {})
        developer_data = data.get("developer_data", {})

        return {
            "id": data["id"],
            "symbol": data["symbol"].upper(),
            "name": data["name"],
            "categories": data.get("categories", []),
            "description": data.get("description", {}).get("en", "")[:500],  # First 500 chars
            # Price data
            "price_usd": market_data.get("current_price", {}).get("usd", 0),
            "price_btc": market_data.get("current_price", {}).get("btc", 0),
            "price_eth": market_data.get("current_price", {}).get("eth", 0),
            # Market metrics
            "market_cap": market_data.get("market_cap", {}).get("usd", 0),
            "market_cap_rank": data.get("market_cap_rank", 0),
            "fully_diluted_valuation": market_data.get("fully_diluted_valuation", {}).get("usd", 0),
            "total_volume": market_data.get("total_volume", {}).get("usd", 0),
            # Supply
            "circulating_supply": market_data.get("circulating_supply", 0),
            "total_supply": market_data.get("total_supply", 0),
            "max_supply": market_data.get("max_supply"),
            # Price changes
            "price_change_1h": market_data.get("price_change_percentage_1h_in_currency", {}).get("usd", 0),
            "price_change_24h": market_data.get("price_change_percentage_24h", 0),
            "price_change_7d": market_data.get("price_change_percentage_7d", 0),
            "price_change_30d": market_data.get("price_change_percentage_30d", 0),
            "price_change_1y": market_data.get("price_change_percentage_1y", 0),
            # ATH/ATL
            "ath": market_data.get("ath", {}).get("usd", 0),
            "ath_change_percentage": market_data.get("ath_change_percentage", {}).get("usd", 0),
            "ath_date": market_data.get("ath_date", {}).get("usd"),
            "atl": market_data.get("atl", {}).get("usd", 0),
            "atl_change_percentage": market_data.get("atl_change_percentage", {}).get("usd", 0),
            "atl_date": market_data.get("atl_date", {}).get("usd"),
            # Community metrics
            "community": {
                "twitter_followers": community_data.get("twitter_followers", 0),
                "telegram_users": community_data.get("telegram_channel_user_count", 0),
                "reddit_subscribers": community_data.get("reddit_subscribers", 0),
                "reddit_accounts_active_48h": community_data.get("reddit_accounts_active_48h", 0),
            },
            # Developer activity
            "developer": {
                "github_stars": developer_data.get("stars", 0),
                "github_forks": developer_data.get("forks", 0),
                "github_subscribers": developer_data.get("subscribers", 0),
                "total_issues": developer_data.get("total_issues", 0),
                "closed_issues": developer_data.get("closed_issues", 0),
                "pull_requests_merged": developer_data.get("pull_requests_merged", 0),
                "commit_count_4_weeks": developer_data.get("commit_count_4_weeks", 0),
            },
            # Misc
            "genesis_date": data.get("genesis_date"),
            "sentiment_votes_up_percentage": data.get("sentiment_votes_up_percentage", 50),
            "sentiment_votes_down_percentage": data.get("sentiment_votes_down_percentage", 50),
            "watchlist_portfolio_users": market_data.get("watchlist_portfolio_users", 0),
        }

    async def get_market_data(self, symbols: List[str]) -> List[Dict[str, Any]]:
        """
        Get market data for multiple assets at once.

        Note: CoinGecko API doesn't support filtering by symbols directly,
        so we fetch all and filter. For large lists, use get_asset_data() individually.

        Args:
            symbols: List of symbols (converted to lowercase IDs)

        Returns:
            List of market data dictionaries
        """
        # For now, fetch top coins and filter by symbols
        # In production, you'd want a symbol->id mapping
        url = f"{self.base_url}/coins/markets"
        params = {
            "vs_currency": "usd",
            "order": "market_cap_desc",
            "per_page": 250,
            "page": 1,
            "sparkline": "false",
        }

        if self.api_key:
            params["x_cg_pro_api_key"] = self.api_key

        data = await self._make_request(url, params)

        # Filter by requested symbols
        if symbols:
            symbols_lower = [s.lower() for s in symbols]
            data = [coin for coin in data if coin["symbol"].lower() in symbols_lower]

        return data

    async def get_trending(self) -> List[Dict[str, Any]]:
        """
        Get trending coins on CoinGecko.

        Returns:
            List of trending coins with basic data
        """
        url = f"{self.base_url}/search/trending"
        data = await self._make_request(url)

        trending = []
        for item in data.get("coins", []):
            coin = item.get("item", {})
            trending.append({
                "id": coin.get("id"),
                "symbol": coin.get("symbol", "").upper(),
                "name": coin.get("name"),
                "market_cap_rank": coin.get("market_cap_rank"),
                "thumb": coin.get("thumb"),
                "score": coin.get("score", 0),
            })

        return trending

    async def get_top_gainers(self, limit: int = 100, timeframe: str = "24h") -> List[Dict[str, Any]]:
        """
        Get top gaining cryptocurrencies.

        Args:
            limit: Number of results to return
            timeframe: '1h', '24h', '7d', '30d'

        Returns:
            List of top gainers sorted by price change percentage
        """
        url = f"{self.base_url}/coins/markets"

        order_map = {
            "1h": "price_change_percentage_1h_desc",
            "24h": "price_change_percentage_24h_desc",
            "7d": "price_change_percentage_7d_desc",
            "30d": "price_change_percentage_30d_desc",
        }

        params = {
            "vs_currency": "usd",
            "order": order_map.get(timeframe, "price_change_percentage_24h_desc"),
            "per_page": min(limit, 250),
            "page": 1,
            "sparkline": "false",
            "price_change_percentage": "1h,24h,7d,30d",
        }

        if self.api_key:
            params["x_cg_pro_api_key"] = self.api_key

        data = await self._make_request(url, params)
        return data

    async def get_new_listings(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get recently listed cryptocurrencies.

        Args:
            limit: Number of results to return

        Returns:
            List of new listings sorted by date added
        """
        url = f"{self.base_url}/coins/list"
        params = {"include_platform": "true"}

        if self.api_key:
            params["x_cg_pro_api_key"] = self.api_key

        all_coins = await self._make_request(url, params)

        # Get detailed data for most recent coins (approximation)
        # In production, you'd track new additions over time
        recent_coins = all_coins[-limit:]  # Get last N coins from list

        return recent_coins

    async def get_ohlcv(
        self,
        coin_id: str,
        days: int = 30,
        vs_currency: str = "usd",
    ) -> Dict[str, Any]:
        """
        Get OHLCV (candlestick) data for technical analysis.

        Args:
            coin_id: CoinGecko coin ID
            days: Number of days of historical data (1-365)
            vs_currency: Quote currency (default 'usd')

        Returns:
            Dictionary with OHLCV data
        """
        url = f"{self.base_url}/coins/{coin_id}/ohlc"
        params = {
            "vs_currency": vs_currency,
            "days": days,
        }

        if self.api_key:
            params["x_cg_pro_api_key"] = self.api_key

        data = await self._make_request(url, params)

        # Transform to structured format
        ohlcv = {
            "timestamps": [],
            "open": [],
            "high": [],
            "low": [],
            "close": [],
            "volume": [],
        }

        for candle in data:
            timestamp, o, h, l, c = candle
            ohlcv["timestamps"].append(timestamp)
            ohlcv["open"].append(o)
            ohlcv["high"].append(h)
            ohlcv["low"].append(l)
            ohlcv["close"].append(c)
            # The OHLC endpoint does not return volume, but downstream
            # analytics expect the key to exist. We fill with None values
            # so analytics modules can gracefully skip volume-based signals
            # when the data source does not provide it.
            ohlcv["volume"].append(None)

        return ohlcv

    async def search(self, query: str) -> List[Dict[str, Any]]:
        """
        Search for cryptocurrencies by name or symbol.

        Args:
            query: Search query

        Returns:
            List of matching coins
        """
        url = f"{self.base_url}/search"
        params = {"query": query}

        if self.api_key:
            params["x_cg_pro_api_key"] = self.api_key

        data = await self._make_request(url, params)
        return data.get("coins", [])
