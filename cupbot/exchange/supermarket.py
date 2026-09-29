"""Live client for the Predictions Cup platform (sig.thesuper.market).

STATUS: skeleton. Confirmed from public sources: base URL /api/v1, Bearer API
key with read + trade scopes, a bulk prices endpoint (/exchanges/prices, up to
100 ids), MARKET and LIMIT order kinds, Decimal-style numeric prices, and an
Idempotency-Key header on at least one order endpoint. Everything else (paths,
field names, price scale, rate and position limits) must be checked against
https://sig.thesuper.market/api/v1/docs while logged in, then set in
config/*.json -> venue.paths and the `_parse_*` helpers below.

Safety:
  * Orders are only sent if venue.live_trading_enabled is true in config AND the
    process was started with --live. Otherwise place_order() raises.
  * Every create carries an Idempotency-Key = client order id. A timeout returns
    status UNKNOWN; the engine never blindly resends, the reconciler resolves it.
  * Separate token buckets for reads and writes, below the published limits.
"""

from __future__ import annotations

import asyncio
import os
from decimal import Decimal
from typing import Any, Dict, List, Optional, Sequence

from ..config import VenueConfig
from ..ratelimit import TokenBucket, backoff_s
from ..types import (Account, BookMap, Level, Market, OrderBook, OrderKind, OrderRequest, OrderResult,
                     OrderStatus)
from .base import Exchange

try:  # optional dependency, only needed for live mode
    import aiohttp
except ImportError:  # pragma: no cover
    aiohttp = None


class LiveTradingDisabled(RuntimeError):
    pass


def _get(d: Dict[str, Any], *keys: str, default=None):
    """First present key (field names are unverified; accept common variants)."""
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


class SuperMarketExchange(Exchange):
    name = "supermarket"

    def __init__(self, cfg: VenueConfig, allow_orders: bool = False):
        if aiohttp is None:
            raise RuntimeError("live mode needs aiohttp: pip install -r requirements.txt")
        key = os.environ.get(cfg.api_key_env, "")
        if not key:
            raise RuntimeError(f"set {cfg.api_key_env} in your environment or .env")
        self.cfg = cfg
        self.allow_orders = allow_orders and cfg.live_trading_enabled
        self._headers = {"Authorization": f"Bearer {key}", "Accept": "application/json"}
        self.read_bucket = TokenBucket(cfg.read_rps)
        self.write_bucket = TokenBucket(cfg.write_rps)
        self._session: Optional["aiohttp.ClientSession"] = None
        self.market_cache: Dict[str, Market] = {}

    # ------------------------------------------------------------ transport
    async def _sess(self) -> "aiohttp.ClientSession":
        if self._session is None or self._session.closed:
            conn = aiohttp.TCPConnector(limit=8, keepalive_timeout=60, ttl_dns_cache=300)
            self._session = aiohttp.ClientSession(
                base_url=None, headers=self._headers, connector=conn,
                timeout=aiohttp.ClientTimeout(total=self.cfg.timeout_s))
        return self._session

    def _url(self, name: str, **kw: str) -> str:
        return self.cfg.base_url.rstrip("/") + self.cfg.paths[name].format(**kw)

    async def _request(self, method: str, name: str, *, write: bool = False, params=None, json=None,
                       headers=None, retries: int = 3, **path_kw) -> Any:
        bucket = self.write_bucket if write else self.read_bucket
        sess = await self._sess()
        url = self._url(name, **path_kw)
        if self.cfg.tournament_id:
            params = dict(params or {})
            params.setdefault("tournament_id", self.cfg.tournament_id)
        for attempt in range(retries + 1):
            await bucket.acquire()
            try:
                async with sess.request(method, url, params=params, json=json, headers=headers) as r:
                    if r.status == 429:
                        ra = float(r.headers.get("Retry-After", "1") or 1)
                        bucket.cooldown(ra)
                        continue
                    if r.status >= 500 and attempt < retries:
                        await asyncio.sleep(backoff_s(attempt))
                        continue
                    body = await r.json(content_type=None)
                    if r.status >= 400:
                        raise RuntimeError(f"{method} {url} -> {r.status}: {body}")
                    return body
            except asyncio.TimeoutError:
                if write:
                    raise  # never retry an order create blindly
                await asyncio.sleep(backoff_s(attempt))
        raise RuntimeError(f"{method} {url}: retries exhausted")

    # ----------------------------------------------------------- read side
    async def whoami(self) -> Any:
        return await self._request("GET", "whoami")

    async def get_markets(self) -> List[Market]:
        body = await self._request("GET", "markets")
        items = body if isinstance(body, list) else _get(body, "markets", "data", "items", default=[])
        out = []
        for it in items:
            mid = str(_get(it, "id", "market_id", "slug"))
            m = self.market_cache.get(mid) or Market(mid, str(_get(it, "title", "name", "question", default=mid)))
            m.closed = bool(_get(it, "closed", "resolved", default=False))
            self.market_cache[mid] = m
            out.append(m)
        return out

    async def get_books(self, market_ids: Sequence[str]) -> BookMap:
        """Bulk top-of-book via /exchanges/prices (<= 100 ids per call). Full depth
        per market costs one request each, so only fetch depth where needed."""
        out: BookMap = {}
        ids = list(market_ids)
        for i in range(0, len(ids), 100):
            chunk = ids[i:i + 100]
            body = await self._request("GET", "prices", params={"ids": ",".join(chunk)})
            items = body if isinstance(body, list) else _get(body, "prices", "data", default=[])
            for it in items:
                mid = str(_get(it, "id", "market_id", "exchange_id"))
                out[mid] = self._parse_bbo(mid, it)
        return out

    async def get_book_depth(self, market_id: str) -> OrderBook:
        body = await self._request("GET", "book", market_id=market_id)
        s = self.cfg.price_scale
        bids = [Level(float(Decimal(str(_get(l, "price", "p")))) / s, float(_get(l, "quantity", "qty", "size")))
                for l in _get(body, "bids", default=[])]
        asks = [Level(float(Decimal(str(_get(l, "price", "p")))) / s, float(_get(l, "quantity", "qty", "size")))
                for l in _get(body, "asks", default=[])]
        bids.sort(key=lambda l: -l.price)
        asks.sort(key=lambda l: l.price)
        return OrderBook(market_id, bids, asks)

    def _parse_bbo(self, mid: str, it: Dict[str, Any]) -> OrderBook:
        s = self.cfg.price_scale
        bb = _get(it, "best_bid", "bid", "yes_bid")
        ba = _get(it, "best_ask", "ask", "yes_ask")
        bq = _get(it, "best_bid_size", "bid_size", default=0)
        aq = _get(it, "best_ask_size", "ask_size", default=0)
        bids = [Level(float(Decimal(str(bb))) / s, float(bq or 0))] if bb is not None else []
        asks = [Level(float(Decimal(str(ba))) / s, float(aq or 0))] if ba is not None else []
        return OrderBook(mid, bids, asks)

    async def get_account(self) -> Account:
        bal = await self._request("GET", "balance")
        pos = await self._request("GET", "positions")
        cash = float(Decimal(str(_get(bal, "balance", "cash", "available", default=0))))
        items = pos if isinstance(pos, list) else _get(pos, "positions", "data", default=[])
        positions = {}
        for p in items:
            mid = str(_get(p, "market_id", "id", "exchange_id"))
            q = float(Decimal(str(_get(p, "quantity", "qty", "position", default=0))))
            side = str(_get(p, "side", "outcome", default="yes")).lower()
            positions[mid] = q if side in ("yes", "long", "buy") else -q
        return Account(cash, positions)

    # ---------------------------------------------------------- write side
    async def place_order(self, req: OrderRequest) -> OrderResult:
        if not self.allow_orders:
            raise LiveTradingDisabled("live orders are disabled (config venue.live_trading_enabled + --live)")
        s = self.cfg.price_scale
        payload = {
            "market_id": req.market_id,
            "side": req.side.value,
            "kind": "LIMIT" if req.kind == OrderKind.LIMIT else "MARKET",
            "quantity": str(int(req.qty)),
            "client_order_id": req.client_id,
        }
        if req.kind == OrderKind.LIMIT:
            payload["limit_price"] = str(Decimal(str(round(req.price * s, 4))))
            payload["time_in_force"] = req.tif.value.upper()
        try:
            body = await self._request("POST", "orders", write=True, json=payload,
                                       headers={"Idempotency-Key": req.client_id}, retries=0)
        except asyncio.TimeoutError:
            return OrderResult(req.client_id, OrderStatus.UNKNOWN, error="timeout: reconcile before resending")
        except RuntimeError as e:
            return OrderResult(req.client_id, OrderStatus.REJECTED, error=str(e))
        status = str(_get(body, "status", default="acked")).lower()
        filled = float(_get(body, "filled_quantity", "filled", default=0) or 0)
        avg = float(_get(body, "average_price", "avg_price", default=0) or 0) / s
        st = {"filled": OrderStatus.FILLED, "partially_filled": OrderStatus.PARTIAL, "open": OrderStatus.ACKED,
              "cancelled": OrderStatus.CANCELLED, "rejected": OrderStatus.REJECTED}.get(status, OrderStatus.ACKED)
        return OrderResult(req.client_id, st, filled, avg, venue_order_id=str(_get(body, "id", "order_id")))

    async def cancel_all(self, market_id: str = "") -> int:
        body = await self._request("GET", "orders", params={"status": "open"})
        items = body if isinstance(body, list) else _get(body, "orders", "data", default=[])
        n = 0
        for o in items:
            if market_id and str(_get(o, "market_id")) != market_id:
                continue
            # cancels can 503 under load and are safe to retry (platform changelog)
            await self._request("DELETE", "cancel", write=True, retries=4, order_id=str(_get(o, "id", "order_id")))
            n += 1
        return n

    async def close(self) -> None:
        if self._session is not None:
            await self._session.close()
