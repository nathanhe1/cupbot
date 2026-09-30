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
import collections
import json
import time
from typing import Callable, Dict, Optional

from .mathutil import logit, logit_blend

try:
    import aiohttp
except ImportError:  # pragma: no cover
    aiohttp = None

KALSHI_REST = "https://api.elections.kalshi.com/trade-api/v2"
POLY_CLOB = "https://clob.polymarket.com"
POLY_WS = "wss://ws-subscriptions-clob.polymarket.com/ws/market"


class AnchorBook:
    """Latest price per (cup market, venue), with timestamps."""

    def __init__(self, max_age_s: float = 60.0, max_spread: float = 0.06):
        self.max_age_s = max_age_s
        self.max_spread = max_spread
        self.quotes: Dict[str, Dict[str, tuple]] = {}   # mid -> venue -> (p, ts)
        self.on_change: Optional[Callable[[str], None]] = None
        self.move_window_s = 120.0
        self.history: Dict[str, "collections.deque"] = {}

    def _blend(self, mid: str, now: float) -> Optional[float]:
        ps = [p for p, ts in self.quotes.get(mid, {}).values() if now - ts <= self.max_age_s]
        return logit_blend(ps, [1.0] * len(ps)) if ps else None

    def recent_moves(self) -> Dict[str, float]:
        """market -> change of the blended anchor mid over the last move_window_s (prob)."""
        out = {}
        for mid, h in self.history.items():
            vals = [v for _, v in h if v is not None]
            if len(vals) >= 2:
                out[mid] = vals[-1] - vals[0]
        return out

    def update(self, market_id: str, venue: str, bid: Optional[float], ask: Optional[float],
               last: Optional[float] = None, invert: bool = False) -> None:
        # quote mid only: trade prints overshoot by 1-2 cents and half-retrace within
        # minutes (tuning/RECOMMENDATIONS.md, finding 5), so `last` is ignored
        if not (bid is not None and ask is not None and 0 < bid <= ask < 1 and ask - bid <= self.max_spread):
            return
        p = (bid + ask) / 2
        if invert:
            p = 1 - p
        prev = self.quotes.get(market_id, {}).get(venue)
        now = time.time()
        self.quotes.setdefault(market_id, {})[venue] = (p, now)
        h = self.history.setdefault(market_id, collections.deque())
        h.append((now, self._blend(market_id, now)))
        while h and h[0][0] < now - self.move_window_s:
            h.popleft()
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

    def disagreement(self) -> Dict[str, float]:
        """Probability gap between venues per market (missing with one venue). Kalshi prices
        the Democratic side 1-3 points below Polymarket almost everywhere and the gap
        persists for hours, so it is noise to trade around, not signal."""
        now = time.time()
        out = {}
        for mid, venues in self.quotes.items():
            ps = [p for p, ts in venues.values() if now - ts <= self.max_age_s]
            if len(ps) >= 2:
                out[mid] = max(ps) - min(ps)
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
        """All mapped Kalshi markets in one batched request (GET /markets?tickers=...)."""
        sess = await self._sess()
        by_ticker = {m["kalshi"]["ticker"]: (mid, m["kalshi"].get("invert", False))
                     for mid, m in self.mapping.items() if (m.get("kalshi") or {}).get("ticker")}
        tickers = list(by_ticker)
        for i in range(0, len(tickers), 50):
            try:
                async with sess.get(f"{KALSHI_REST}/markets", params={"tickers": ",".join(tickers[i:i + 50])}) as r:
                    if r.status != 200:
                        continue
                    body = await r.json()
            except Exception:
                continue
            for mk in body.get("markets", []):
                if mk.get("ticker") not in by_ticker:
                    continue
                mid, inv = by_ticker[mk["ticker"]]
                self.book.update(mid, "kalshi", _cents_or_dollars(mk, "yes_bid"), _cents_or_dollars(mk, "yes_ask"),
                                 _cents_or_dollars(mk, "last_price"), invert=inv)

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
        """Handle one market-channel message. `book` carries asset_id at the top level;
        `price_change` carries a list of per-asset changes, each with best_bid / best_ask
        (format checked against recorded messages in tuning/data/live)."""
        kind = ev.get("event_type")
        if kind == "book":
            self._poly_set(ev.get("asset_id"), tokens, best,
                           bid=max((float(x["price"]) for x in ev.get("bids", [])), default=None),
                           ask=min((float(x["price"]) for x in ev.get("asks", [])), default=None), reset=True)
        elif kind == "price_change":
            for ch in ev.get("price_changes", ev.get("changes", [])):
                self._poly_set(ch.get("asset_id", ev.get("asset_id")), tokens, best,
                               bid=_f(ch.get("best_bid")), ask=_f(ch.get("best_ask")))
        elif kind == "last_trade_price":
            self._poly_set(ev.get("asset_id"), tokens, best, last=_f(ev.get("price")))

    def _poly_set(self, tid, tokens, best, bid=None, ask=None, last=None, reset=False) -> None:
        if tid not in tokens:
            return
        mid, inv = tokens[tid]
        b = best.setdefault(tid, {})
        if reset or bid is not None:
            b["bid"] = bid
        if reset or ask is not None:
            b["ask"] = ask
        if last is not None:
            b["last"] = last
        self.book.update(mid, "polymarket", b.get("bid"), b.get("ask"), b.get("last"), invert=inv)

    async def poll_polymarket_once(self) -> None:
        """All mapped Polymarket books in one batched request (POST /books)."""
        by_token = {m["polymarket"]["token_id"]: (mid, m["polymarket"].get("invert", False))
                    for mid, m in self.mapping.items() if (m.get("polymarket") or {}).get("token_id")}
        if not by_token:
            return
        sess = await self._sess()
        try:
            async with sess.post(f"{POLY_CLOB}/books", json=[{"token_id": t} for t in by_token]) as r:
                if r.status != 200:
                    return
                books = await r.json()
        except Exception:
            return
        for bk in books:
            if bk.get("asset_id") not in by_token:
                continue
            mid, inv = by_token[bk["asset_id"]]
            bids = [float(x["price"]) for x in bk.get("bids", [])]
            asks = [float(x["price"]) for x in bk.get("asks", [])]
            self.book.update(mid, "polymarket", max(bids) if bids else None, min(asks) if asks else None, invert=inv)

    async def run(self) -> None:
        async def poly_poll():
            while True:
                await self.poll_polymarket_once()
                await asyncio.sleep(max(self.poll_s, 15))
        # Kalshi: batched REST at poll_s until a Kalshi websocket key is added; Polymarket:
        # public websocket, with a slow batched REST sweep as a safety net
        await asyncio.gather(self.run_kalshi_polling(), self.run_polymarket_ws(), poly_poll())

    async def close(self) -> None:
        if self._session is not None:
            await self._session.close()


def _f(x) -> Optional[float]:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


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
