"""Data source adapters for crypto market data."""

from typing import Dict, Type

from .base import BaseAdapter
from .coingecko import CoinGeckoAdapter

ADAPTER_REGISTRY: Dict[str, Type[BaseAdapter]] = {
    "coingecko": CoinGeckoAdapter,
}


def get_adapter(name: str, **kwargs) -> BaseAdapter:
    """Return an adapter instance by name.

    Args:
        name: Name of the adapter registered in :data:`ADAPTER_REGISTRY`.
        **kwargs: Keyword arguments forwarded to the adapter constructor.

    Raises:
        ValueError: If the adapter name is not registered.

    Returns:
        Instantiated adapter ready for use.
    """

    try:
        adapter_cls = ADAPTER_REGISTRY[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unknown adapter: {name}") from exc

    return adapter_cls(**kwargs)


__all__ = ["BaseAdapter", "CoinGeckoAdapter", "get_adapter", "ADAPTER_REGISTRY"]
