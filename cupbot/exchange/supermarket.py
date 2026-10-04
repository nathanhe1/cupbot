"""Live client for the Predictions Cup platform (sig.thesuper.market).

Endpoints, fields and limits follow the official spec at /api/v1/openapi.json
(a copy is saved in docs/openapi.json). Key facts:
  * The bot's market id is the Cup's exchangeId (the tradable contract). A binary
    market has one exchange; a multi-outcome market has one per option.
  * Prices are YES prices in 0-1; limit prices must sit on the 0.005 tick.
  * Every read and order is scoped to the tournament (venue.tournament_id / slug).
  * 100 reads and 30 writes per minute per account.

Safety:
  * Orders are only sent if venue.live_trading_enabled is true in config AND the
    process was started with --live. Otherwise place_order() raises.
  * Every create carries idempotencyKey = client order id. A timeout or a 5xx
    (including 502 ORDER_STATUS_UNKNOWN) returns status UNKNOWN; the engine never
    blindly resends, and lookup_order() resolves it from the order list.
  * Separate token buckets for reads and writes, below the published limits.
"""

from __future__ import annotations

import asyncio
import math
import os
import time
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional, Sequence

from ..config import VenueConfig
from ..ratelimit import TokenBucket, backoff_s
from ..types import (Account, BookMap, Level, Market, OrderBook, OrderKind, OrderRequest, OrderResult,
                     OrderStatus, Side, TimeInForce)
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
    bbo_sizes_unknown = True   # /exchanges/prices has no sizes; the engine reads depth before trading

    def __init__(self, cfg: VenueConfig, allow_orders: bool = False, api_key: Optional[str] = None):
        key = api_key if api_key is not None else os.environ.get(cfg.api_key_env, "")
        if not key:
            raise RuntimeError(f"set {cfg.api_key_env} in your environment or .env")
        self.cfg = cfg
        self.allow_orders = allow_orders and cfg.live_trading_enabled
        self._headers = {"Authorization": f"Bearer {key}", "Accept": "application/json"}
        self.read_bucket = TokenBucket(cfg.read_rps)
        self.write_bucket = TokenBucket(cfg.write_rps)
        # Two connection pools: orders never queue behind a slow read on the same
        # HTTP/1.1 connection (measured 217 ms vs 13 ms in tuning/bench_local.py).
        self._session: Optional["aiohttp.ClientSession"] = None        # reads
        self._order_session: Optional["aiohttp.ClientSession"] = None  # writes only
        self._warm_task = None
        self.market_cache: Dict[str, Market] = {}
        self.exchange_market: Dict[str, str] = {}           # exchangeId -> Cup marketId
        self._placed: Dict[str, Dict[str, Any]] = {}       # client id -> what we sent, for lookup_order
        self.last_rate_headers: Dict[str, str] = {}
        self.calls = 0

    # ------------------------------------------------------------ transport
    def _new_session(self, limit: int) -> "aiohttp.ClientSession":
        # aiohttp sets TCP_NODELAY on its sockets, so no Nagle / delayed-ACK stall
        conn = aiohttp.TCPConnector(limit=limit, keepalive_timeout=120, ttl_dns_cache=3600)
        return aiohttp.ClientSession(headers=self._headers, connector=conn,
                                     timeout=aiohttp.ClientTimeout(total=self.cfg.timeout_s))

    async def _sess(self, write: bool = False) -> "aiohttp.ClientSession":
        if aiohttp is None:
            raise RuntimeError("live mode needs aiohttp: pip3 install --user aiohttp")
        if write:
            if self._order_session is None or self._order_session.closed:
                self._order_session = self._new_session(limit=2)
            return self._order_session
        if self._session is None or self._session.closed:
            self._session = self._new_session(limit=6)
        return self._session

    async def keep_warm(self, every_s: float = 20.0) -> None:
        """Keep the order connection's TCP+TLS alive with a cheap authenticated GET, so an
        order never pays connection setup (measured 40 ms cold vs 13 ms warm)."""
        while True:
            try:
                await self._request("GET", "whoami", write=True, retries=0)
            except Exception:
                pass
            await asyncio.sleep(every_s)

    async def _request(self, method: str, name: str, *, write: bool = False, params=None, json=None,
                       headers=None, retries: int = 3, **path_kw) -> Any:
        bucket = self.write_bucket if write else self.read_bucket
        url = self._url(name, **path_kw)
        if self.cfg.tournament_id and method == "GET":
            params = dict(params or {})
            params.setdefault("tournamentId", self.cfg.tournament_id)
        for attempt in range(retries + 1):
            await bucket.acquire()
            try:
                status, rheaders, body = await self._send(method, url, params, json, headers, write)
                self.calls += 1
                rl = {k: v for k, v in rheaders.items() if "ratelimit" in k.lower() or k.lower() == "retry-after"}
                if rl:
                    self.last_rate_headers = rl
                if status == 429:
                    ra = float(rheaders.get("Retry-After", "1") or 1)
                    bucket.cooldown(ra)
                    continue
                if status >= 500 and attempt < retries:
                    await asyncio.sleep(backoff_s(attempt))
                    continue
                if status >= 400:
                    raise RuntimeError(f"{method} {url} -> {status}: {body}")
                return body
            except asyncio.TimeoutError:
                if write:
                    raise  # never retry an order create blindly
                await asyncio.sleep(backoff_s(attempt))
        raise RuntimeError(f"{method} {url}: retries exhausted")

    async def _send(self, method: str, url: str, params, json_body, headers, write: bool = False):
        """One HTTP exchange -> (status, headers, parsed body). Tests override this."""
        sess = await self._sess(write)
        async with sess.request(method, url, params=params, json=json_body, headers=headers) as r:
            try:
                body = await r.json(content_type=None)
            except ValueError:
                body = await r.text()
            return r.status, dict(r.headers), body

    def _url(self, name: str, **kw: str) -> str:
        kw.setdefault("slug", self.cfg.tournament_slug)
        return self.cfg.base_url.rstrip("/") + self.cfg.paths[name].format(**kw)

    # ----------------------------------------------------------- read side
    async def whoami(self) -> Any:
        return await self._request("GET", "whoami")

    async def get_markets(self) -> List[Market]:
        """Every exchange in the tournament as one bot market (paginated /markets)."""
        out: List[Market] = []
        cursor = None
        for _ in range(50):
            params = {"limit": 100}
            if cursor:
                params["cursor"] = cursor
            body = await self._request("GET", "markets", params=params)
            items = body if isinstance(body, list) else _get(body, "data", "markets", default=[])
            for it in items:
                title = str(_get(it, "title", default=it.get("id")))
                status = str(_get(it, "status", default="open")).lower()
                for ctx in it.get("contexts") or []:
                    t = ctx.get("tournament") or {}
                    if t.get("id") == self.cfg.tournament_id:
                        status = str(ctx.get("status") or status).lower()
                exs = it.get("exchanges") or [{"id": it.get("id"), "option": "YES"}]
                multi = bool(it.get("isMultiOutcome")) or len(exs) > 1
                for e in exs:
                    eid = str(e["id"])
                    t = f"{title} [{e.get('option')}]" if multi else title
                    m = self.market_cache.get(eid) or Market(eid, t)
                    m.closed = status not in ("open", "active")
                    self.market_cache[eid] = m
                    self.exchange_market[eid] = str(it.get("id"))
                    out.append(m)
            pag = body.get("pagination", {}) if isinstance(body, dict) else {}
            cursor = pag.get("nextCursor")
            if not pag.get("hasMore") or not cursor:
                break
        return out

    async def get_books(self, market_ids: Sequence[str]) -> BookMap:
        """Bulk top-of-book via /exchanges/prices (<= 100 exchange ids per call). The bulk
        snapshot has no sizes; full depth costs one read per exchange (get_book_depth)."""
        out: BookMap = {}
        ids = list(market_ids)
        for i in range(0, len(ids), 100):
            chunk = ids[i:i + 100]
            body = await self._request("GET", "prices", params={"ids": ",".join(chunk)})
            items = body if isinstance(body, list) else _get(body, "data", "prices", default=[])
            for it in items:
                mid = str(_get(it, "exchangeId", "id"))
                out[mid] = self._parse_bbo(mid, it)
        return out

    async def get_book_depth(self, market_id: str, depth: int = 20) -> OrderBook:
        body = await self._request("GET", "book", params={"depth": depth}, market_id=market_id)
        s = self.cfg.price_scale
        bids = [Level(float(Decimal(str(l["price"]))) / s, float(l["quantity"])) for l in body.get("bids") or []]
        asks = [Level(float(Decimal(str(l["price"]))) / s, float(l["quantity"])) for l in body.get("asks") or []]
        bids.sort(key=lambda l: -l.price)
        asks.sort(key=lambda l: l.price)
        return OrderBook(market_id, bids, asks)

    # Size shown for a bulk-snapshot level, which carries no size. The pre-open sponsor
    # quotes are 1000 deep; the strategy's own caps bound the order size anyway.
    BBO_SIZE_GUESS = 1000.0

    def _parse_bbo(self, mid: str, it: Dict[str, Any]) -> OrderBook:
        s = self.cfg.price_scale
        bb = _get(it, "bestBid", "best_bid")
        ba = _get(it, "bestAsk", "best_ask")
        bq = _get(it, "bestBidSize", "best_bid_size", default=self.BBO_SIZE_GUESS)
        aq = _get(it, "bestAskSize", "best_ask_size", default=self.BBO_SIZE_GUESS)
        bids = [Level(float(Decimal(str(bb))) / s, float(bq))] if bb is not None else []
        asks = [Level(float(Decimal(str(ba))) / s, float(aq))] if ba is not None else []
        return OrderBook(mid, bids, asks)

    async def get_account(self) -> Account:
        t = await self._request("GET", "balance")
        pos = await self._request("GET", "positions")
        cash = float(Decimal(str(_get(t, "myBalance", "balance", default=0))))
        items = pos if isinstance(pos, list) else _get(pos, "positions", "data", default=[])
        positions: Dict[str, float] = {}
        for p in items:
            if p.get("settled"):
                continue
            mid = str(_get(p, "exchangeId", "id"))
            lots = p.get("lots") or []
            if lots:   # YES lots count long, NO lots short (net YES exposure)
                q = sum(float(l["quantity"]) * (1 if str(l.get("side", "yes")).lower() == "yes" else -1)
                        for l in lots)
            else:
                q = float(_get(p, "quantity", default=0))
            if q:
                positions[mid] = positions.get(mid, 0.0) + q
        return Account(cash, positions, no_shares=True)

    # ---------------------------------------------------------- write side
    def _price_on_tick(self, price: float, side: Side) -> float:
        """Round a limit price onto the venue tick, never in our disfavour: buys down, sells up."""
        n = price * self.cfg.price_scale / self.cfg.tick
        n = math.floor(n + 1e-9) if side == Side.BUY else math.ceil(n - 1e-9)
        return min(0.995, max(0.005, round(n * self.cfg.tick, 3)))

    async def place_order(self, req: OrderRequest) -> OrderResult:
        if not self.allow_orders:
            raise LiveTradingDisabled("live orders are disabled (config venue.live_trading_enabled + --live)")
        payload: Dict[str, Any] = {
            "idempotencyKey": req.client_id,
            "exchangeId": req.market_id,
            "side": "yes",                      # always YES-denominated; selling flat buys NO
            "action": "buy" if req.side == Side.BUY else "sell",
            "quantity": int(req.qty),
        }
        if self.cfg.tournament_id:
            payload["tournamentId"] = self.cfg.tournament_id
        if req.kind == OrderKind.LIMIT:
            payload["price"] = self._price_on_tick(req.price, req.side)
            if req.tif == TimeInForce.IOC:
                # no IOC flag on this API: a short expiry stops a taking order from resting
                exp = datetime.now(timezone.utc) + timedelta(seconds=20)  # Mac clock measured ~5 s off the venue
                payload["expirationDate"] = exp.isoformat().replace("+00:00", "Z")
        self._placed[req.client_id] = {"exchangeId": req.market_id, "quantity": int(req.qty),
                                       "price": payload.get("price"), "ts": time.time()}
        try:
            body = await self._request("POST", "orders", write=True, json=payload, retries=0)
        except asyncio.TimeoutError:
            return OrderResult(req.client_id, OrderStatus.UNKNOWN, error="timeout: reconcile before resending")
        except RuntimeError as e:
            msg = str(e)
            if "-> 5" in msg or "ORDER_STATUS_UNKNOWN" in msg:
                return OrderResult(req.client_id, OrderStatus.UNKNOWN, error=msg)
            return OrderResult(req.client_id, OrderStatus.REJECTED, error=msg)
        filled = float(_get(body, "quantityTraded", default=0) or 0)
        total = float(_get(body, "totalCost", default=0) or 0)
        fp = _get(body, "fillPrice")
        avg = float(fp) if fp is not None else (total / filled if filled else 0.0)
        if filled >= int(req.qty) - 1e-9:
            st = OrderStatus.FILLED
        elif filled > 0:
            st = OrderStatus.PARTIAL
        elif body.get("open"):
            st = OrderStatus.ACKED
        else:
            st = OrderStatus.CANCELLED   # e.g. self-trade prevented, or a taking order with nothing to hit
        return OrderResult(req.client_id, st, filled, avg / self.cfg.price_scale,
                           venue_order_id=str(_get(body, "orderId", default="")))

    async def place_batch(self, reqs: Sequence[OrderRequest], expire_s: float) -> List[OrderResult]:
        """Resting limit orders, up to 50 per write (POST /orders/batch). Each carries a
        venue-side expiry so nothing outlives a dead bot by long. Per-item failures come
        back as REJECTED; a whole-request failure as UNKNOWN for every item (the ladder
        cancels the market before it requotes, so nothing is resent blindly)."""
        if not self.allow_orders:
            raise LiveTradingDisabled("live orders are disabled (config venue.live_trading_enabled + --live)")
        exp = (datetime.now(timezone.utc) + timedelta(seconds=expire_s)).isoformat().replace("+00:00", "Z")
        items = []
        for r in reqs:
            it: Dict[str, Any] = {"exchangeId": r.market_id, "side": "yes",
                                  "action": "buy" if r.side == Side.BUY else "sell",
                                  "quantity": int(r.qty), "price": self._price_on_tick(r.price, r.side),
                                  "expirationDate": exp}
            if self.cfg.tournament_id:
                it["tournamentId"] = self.cfg.tournament_id
            items.append(it)
        key = reqs[0].client_id + f"-b{len(reqs)}"
        try:
            body = await self._request("POST", "batch", write=True, json={"idempotencyKey": key, "orders": items},
                                       retries=0)
        except (asyncio.TimeoutError, RuntimeError) as e:
            return [OrderResult(r.client_id, OrderStatus.UNKNOWN, error=str(e)[:200]) for r in reqs]
        results = body.get("results", []) if isinstance(body, dict) else []
        out = [OrderResult(r.client_id, OrderStatus.UNKNOWN) for r in reqs]
        for res in results:
            i = res.get("index")
            if not isinstance(i, int) or not 0 <= i < len(reqs):
                continue
            if not res.get("ok"):
                out[i] = OrderResult(reqs[i].client_id, OrderStatus.REJECTED, error=str(res.get("data"))[:200])
                continue
            d = res.get("data") or {}
            filled = float(_get(d, "quantityTraded", default=0) or 0)
            fp = _get(d, "fillPrice")
            avg = float(fp) if fp is not None else (float(_get(d, "totalCost", default=0) or 0) / filled
                                                    if filled else 0.0)
            st = OrderStatus.ACKED if d.get("open") else (OrderStatus.FILLED if filled else OrderStatus.CANCELLED)
            out[i] = OrderResult(reqs[i].client_id, st, filled, avg, venue_order_id=str(_get(d, "orderId", default="")))
        return out

    async def get_vwap(self, market_id: str, hours: int = 5) -> Optional[float]:
        """Volume-weighted price of the last `hours` hourly candles (the Cup's valuation
        price is a 5 h VWAP), or None without trades."""
        body = await self._request("GET", "history", params={"resolution": "1h", "limit": hours},
                                   market_id=market_id)
        cs = (body.get("candles") or []) if isinstance(body, dict) else []
        v = sum(float(c.get("volume") or 0) for c in cs)
        if v <= 0:
            return None
        return sum(float(c["vwap"]) * float(c.get("volume") or 0) for c in cs if c.get("vwap") is not None) / v

    async def open_orders(self) -> List[Dict]:
        body = await self._request("GET", "orders", params={"status": "open", "limit": 200})
        return body if isinstance(body, list) else _get(body, "data", "orders", default=[])

    async def lookup_order(self, client_id: str) -> Optional[Dict]:
        """Resolve an UNKNOWN create: the order list has no client id, so match what we
        sent (exchange, quantity, limit price) among orders created since we sent it."""
        sent = self._placed.get(client_id)
        if not sent:
            return None
        body = await self._request("GET", "orders", params={"status": "all", "exchangeId": sent["exchangeId"],
                                                            "limit": 50})
        items = body if isinstance(body, list) else _get(body, "data", default=[])
        for o in items:
            try:
                created = datetime.fromisoformat(str(o["createdAt"]).replace("Z", "+00:00")).timestamp()
            except (KeyError, ValueError):
                continue
            if created < sent["ts"] - 10 or int(float(o.get("quantity", 0))) != sent["quantity"]:
                continue
            if sent["price"] is not None and o.get("priceLimit") is not None \
                    and abs(float(o["priceLimit"]) - sent["price"]) > 1e-6 \
                    and abs(1 - float(o["priceLimit"]) - sent["price"]) > 1e-6:
                continue
            return o
        return None

    async def cancel_all(self, market_id: str = "") -> int:
        """POST /orders/cancel-all for the tournament (optionally one exchange). A 503 means
        it paused partway; retry until 200, which confirms nothing targeted is still resting."""
        body: Dict[str, Any] = {}
        if self.cfg.tournament_id:
            body["tournamentId"] = self.cfg.tournament_id
        if market_id:
            body["exchangeId"] = market_id
        res = await self._request("POST", "cancel_all", write=True, json=body, retries=6)
        return int(_get(res, "cancelled", default=0) or 0) if isinstance(res, dict) else 0

    async def close(self) -> None:
        for sess in (self._session, self._order_session):
            if sess is not None:
                await sess.close()
