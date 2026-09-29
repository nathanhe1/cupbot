"""External price anchors: Kalshi and Polymarket.

Both venues' market data is public (no key needed for REST reads, nor for
Polymarket's market websocket). Kalshi's websocket needs an API key; until
KALSHI_API_KEY is set we poll Kalshi REST instead.

Mapping lives in config/market_map.json:
  {"<cup market id>": {"race_id": "SEN-NC", "kind": "race", "d_sign": 1,
                       "kalshi": {"ticker": "...", "invert": false},
                       "polymarket": {"token_id": "...", "invert": false}}}
`invert: true` when the venue's YES is the opposite of the Cup market's YES.

Anchor = mid of best bid/ask when the spread is tight (<= max_spread), else
last trade / midpoint fallback; several venues are averaged in log-odds.
Every update calls `on_change(market_id)` so the engine can reprice at once."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Callable, Dict, Optional

from .mathutil import logit_blend

try:
    import aiohttp
except ImportError:  # pragma: no cover
    aiohttp = None

KALSHI_REST = "https://api.elections.kalshi.com/trade-api/v2"
POLY_CLOB = "https://clob.polymarket.com"
POLY_WS = "wss://ws-subscriptions-clob.polymarket.com/ws/market"


class AnchorBook:
    """Latest price per (cup market, venue), with timestamps."""

    def __init__(self, max_age_s: float = 120.0, max_spread: float = 0.06):
        self.max_age_s = max_age_s
        self.max_spread = max_spread
        self.quotes: Dict[str, Dict[str, tuple]] = {}   # mid -> venue -> (p, ts)
        self.on_change: Optional[Callable[[str], None]] = None

    def update(self, market_id: str, venue: str, bid: Optional[float], ask: Optional[float],
               last: Optional[float] = None, invert: bool = False) -> None:
        p = None
        if bid is not None and ask is not None and 0 < bid <= ask < 1 and ask - bid <= self.max_spread:
            p = (bid + ask) / 2
        elif last is not None and 0 < last < 1:
            p = last
        if p is None:
            return
        if invert:
            p = 1 - p
        prev = self.quotes.get(market_id, {}).get(venue)
        self.quotes.setdefault(market_id, {})[venue] = (p, time.time())
        if self.on_change and (prev is None or abs(prev[0] - p) >= 0.005):
            self.on_change(market_id)

    def latest(self) -> Dict[str, float]:
        now = time.time()
        out = {}
        for mid, venues in self.quotes.items():
            ps = [p for p, ts in venues.values() if now - ts <= self.max_age_s]
            if ps:
                out[mid] = logit_blend(ps, [1.0] * len(ps))
        return out


class AnchorFeed:
    def __init__(self, mapping: Dict[str, dict], book: AnchorBook, poll_s: float = 5.0):
        if aiohttp is None:
            raise RuntimeError("anchors need aiohttp: pip install -r requirements.txt")
        self.mapping = mapping
        self.book = book
        self.poll_s = poll_s
        self._session: Optional["aiohttp.ClientSession"] = None

    async def _sess(self):
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5))
        return self._session

    # ------------------------------------------------------------ Kalshi REST
    async def poll_kalshi_once(self) -> None:
        sess = await self._sess()
        for mid, m in self.mapping.items():
            k = m.get("kalshi")
            if not k or not k.get("ticker"):
                continue
            try:
                async with sess.get(f"{KALSHI_REST}/markets/{k['ticker']}") as r:
                    if r.status != 200:
                        continue
                    body = (await r.json()).get("market", {})
            except Exception:
                continue
            bid = _cents_or_dollars(body, "yes_bid")
            ask = _cents_or_dollars(body, "yes_ask")
            last = _cents_or_dollars(body, "last_price")
            self.book.update(mid, "kalshi", bid, ask, last, invert=k.get("invert", False))

    async def run_kalshi_polling(self) -> None:
        while True:
            await self.poll_kalshi_once()
            await asyncio.sleep(self.poll_s)

    # ------------------------------------------------------ Polymarket stream
    async def run_polymarket_ws(self) -> None:
        tokens = {m["polymarket"]["token_id"]: (mid, m["polymarket"].get("invert", False))
                  for mid, m in self.mapping.items() if m.get("polymarket", {}).get("token_id")}
        if not tokens:
            return
        best: Dict[str, Dict[str, float]] = {}
        while True:
            try:
                sess = await self._sess()
                async with sess.ws_connect(POLY_WS, heartbeat=20) as ws:
                    await ws.send_str(json.dumps({"type": "market", "assets_ids": list(tokens)}))
                    async for msg in ws:
                        if msg.type != aiohttp.WSMsgType.TEXT:
                            continue
                        data = json.loads(msg.data)
                        for ev in data if isinstance(data, list) else [data]:
                            self._poly_event(ev, tokens, best)
            except Exception:
                await asyncio.sleep(2)  # reconnect; REST polling below covers the gap

    def _poly_event(self, ev: dict, tokens, best) -> None:
        tid = ev.get("asset_id")
        if tid not in tokens:
            return
        mid, inv = tokens[tid]
        b = best.setdefault(tid, {})
        if ev.get("event_type") == "book":
            bids = [float(x["price"]) for x in ev.get("bids", [])]
            asks = [float(x["price"]) for x in ev.get("asks", [])]
            b["bid"] = max(bids) if bids else None
            b["ask"] = min(asks) if asks else None
        elif ev.get("event_type") == "price_change":
            for ch in ev.get("changes", ev.get("price_changes", [])):
                if "best_bid" in ch:
                    b["bid"] = float(ch["best_bid"])
                if "best_ask" in ch:
                    b["ask"] = float(ch["best_ask"])
        elif ev.get("event_type") == "last_trade_price":
            b["last"] = float(ev.get("price", 0))
        self.book.update(mid, "polymarket", b.get("bid"), b.get("ask"), b.get("last"), invert=inv)

    async def poll_polymarket_once(self) -> None:
        sess = await self._sess()
        for mid, m in self.mapping.items():
            pm = m.get("polymarket")
            if not pm or not pm.get("token_id"):
                continue
            try:
                async with sess.get(f"{POLY_CLOB}/book", params={"token_id": pm["token_id"]}) as r:
                    if r.status != 200:
                        continue
                    body = await r.json()
            except Exception:
                continue
            bids = [float(x["price"]) for x in body.get("bids", [])]
            asks = [float(x["price"]) for x in body.get("asks", [])]
            self.book.update(mid, "polymarket", max(bids) if bids else None, min(asks) if asks else None,
                             invert=pm.get("invert", False))

    async def run(self) -> None:
        async def poly_poll():
            while True:
                await self.poll_polymarket_once()
                await asyncio.sleep(max(self.poll_s, 15))
        await asyncio.gather(self.run_kalshi_polling(), self.run_polymarket_ws(), poly_poll())

    async def close(self) -> None:
        if self._session is not None:
            await self._session.close()


def _cents_or_dollars(body: dict, key: str) -> Optional[float]:
    """Kalshi has used integer cents (yes_bid) and dollar strings (yes_bid_dollars)."""
    v = body.get(f"{key}_dollars")
    if v is not None:
        try:
            return float(v)
        except (TypeError, ValueError):
            pass
    v = body.get(key)
    if v is None:
        return None
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v / 100 if v > 1 else v
