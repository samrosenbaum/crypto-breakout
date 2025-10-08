"""Base adapter class for data sources."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential
import structlog

logger = structlog.get_logger()


class BaseAdapter(ABC):
    """Base class for all data source adapters."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        rate_limit: int = 60,
        timeout: int = 30,
    ):
        """
        Initialize adapter.

        Args:
            api_key: Optional API key for authenticated requests
            rate_limit: Requests per minute limit
            timeout: Request timeout in seconds
        """
        self.api_key = api_key
        self.rate_limit = rate_limit
        self.client = httpx.AsyncClient(timeout=timeout)
        self.logger = logger.bind(adapter=self.__class__.__name__)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
    )
    async def _make_request(
        self,
        url: str,
        params: Optional[Dict] = None,
        headers: Optional[Dict] = None,
    ) -> Dict:
        """
        Make HTTP request with retry logic.

        Args:
            url: Request URL
            params: Query parameters
            headers: Request headers

        Returns:
            JSON response as dictionary

        Raises:
            httpx.HTTPStatusError: On HTTP errors
        """
        self.logger.debug("Making request", url=url, params=params)

        try:
            response = await self.client.get(url, params=params, headers=headers)
            response.raise_for_status()
            data = response.json()

            self.logger.debug("Request successful", url=url, status=response.status_code)
            return data

        except httpx.HTTPStatusError as e:
            self.logger.error(
                "HTTP error",
                url=url,
                status=e.response.status_code,
                error=str(e),
            )
            raise
        except Exception as e:
            self.logger.error("Request failed", url=url, error=str(e))
            raise

    @abstractmethod
    async def get_asset_data(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch detailed data for a specific asset.

        Args:
            symbol: Asset symbol (e.g., BTC, ETH)

        Returns:
            Dictionary containing asset data
        """
        pass

    @abstractmethod
    async def get_market_data(self, symbols: List[str]) -> List[Dict[str, Any]]:
        """
        Fetch market data for multiple assets.

        Args:
            symbols: List of asset symbols

        Returns:
            List of dictionaries containing market data
        """
        pass

    async def close(self):
        """Close HTTP client connection."""
        await self.client.aclose()

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
