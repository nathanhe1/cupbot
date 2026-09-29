"""The exchange interface every venue adapter implements (mock and live).

The engine only ever talks to this interface, so the exact code path that
trades in simulation is the one that trades live."""

from __future__ import annotations

import abc
from typing import Dict, List, Sequence

from ..types import Account, BookMap, Market, OrderRequest, OrderResult


class Exchange(abc.ABC):
    name = "abstract"

    @abc.abstractmethod
    async def get_markets(self) -> List[Market]: ...

    @abc.abstractmethod
    async def get_books(self, market_ids: Sequence[str]) -> BookMap: ...

    @abc.abstractmethod
    async def get_account(self) -> Account: ...

    @abc.abstractmethod
    async def place_order(self, req: OrderRequest) -> OrderResult: ...

    @abc.abstractmethod
    async def cancel_all(self, market_id: str = "") -> int:
        """Cancel resting orders (all markets if market_id is empty). Returns count."""

    async def open_orders(self) -> List[Dict]:
        return []

    async def close(self) -> None:
        return None
