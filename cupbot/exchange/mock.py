"""In-memory exchange with a real price-time-priority limit order book.

Used by the simulator and by paper trading. Other participants (the sponsor's
market maker, student traders) submit orders through the same book via the
synchronous `submit()` method, so the bot's fills, queue position and price
impact come out of actual matching rather than an assumed fill model.

Prices are integer ticks internally (1..99 cents)."""

from __future__ import annotations

import itertools
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Sequence, Tuple

from ..types import (Account, BookMap, Fill, Level, Market, OrderBook, OrderKind, OrderRequest,
                     OrderResult, OrderStatus, Side, TimeInForce)
from .base import Exchange

BOT = "bot"
TICKS = 100


@dataclass
class Resting:
    oid: int
    owner: str
    side: Side
    tick: int
    qty: float
    client_id: str = ""
    born: int = 0  # sim tick when placed (for stale-order stats)


class Book:
    def __init__(self) -> None:
        self.bids: Dict[int, Deque[Resting]] = {}
        self.asks: Dict[int, Deque[Resting]] = {}

    def best_bid(self) -> Optional[int]:
        return max(self.bids) if self.bids else None

    def best_ask(self) -> Optional[int]:
        return min(self.asks) if self.asks else None

    def add(self, o: Resting) -> None:
        side = self.bids if o.side == Side.BUY else self.asks
        side.setdefault(o.tick, deque()).append(o)

    def remove_where(self, pred) -> int:
        n = 0
        for side in (self.bids, self.asks):
            for t in list(side):
                q = side[t]
                keep = deque(o for o in q if not pred(o))
                n += len(q) - len(keep)
                if keep:
                    side[t] = keep
                else:
                    del side[t]
        return n

    def snapshot(self, market_id: str, ts: float, depth: int = 10) -> OrderBook:
        bids = [Level(t / TICKS, sum(o.qty for o in self.bids[t]))
                for t in sorted(self.bids, reverse=True)[:depth]]
        asks = [Level(t / TICKS, sum(o.qty for o in self.asks[t]))
                for t in sorted(self.asks)[:depth]]
        return OrderBook(market_id, bids, asks, ts)


class MockExchange(Exchange):
    name = "mock"

    def __init__(self, markets: Sequence[Market], starting_cash: float = 100_000.0,
                 position_limit: float = 1e12, allow_short: bool = True):
        self.markets: Dict[str, Market] = {m.market_id: m for m in markets}
        self.books: Dict[str, Book] = {m.market_id: Book() for m in markets}
        self.account = Account(cash=starting_cash)
        self.position_limit = position_limit
        self.allow_short = allow_short
        self.now = 0.0
        self.tick_index = 0
        self._oid = itertools.count(1)
        self.fills: List[Fill] = []
        self.counterparty_volume: Dict[str, float] = {}
        self.order_count = 0
        self.rejects = 0

    # ------------------------------------------------------ bot interface
    async def get_markets(self) -> List[Market]:
        return list(self.markets.values())

    async def get_books(self, market_ids: Sequence[str]) -> BookMap:
        return {m: self.books[m].snapshot(m, self.now) for m in market_ids if m in self.books}

    async def get_account(self) -> Account:
        return Account(self.account.cash, dict(self.account.positions))

    async def place_order(self, req: OrderRequest) -> OrderResult:
        self.order_count += 1
        return self.submit(BOT, req)

    async def cancel_all(self, market_id: str = "") -> int:
        n = 0
        for mid, book in self.books.items():
            if market_id and mid != market_id:
                continue
            n += book.remove_where(lambda o: o.owner == BOT)
        return n

    async def open_orders(self) -> List[Dict]:
        out = []
        for mid, book in self.books.items():
            for side in (book.bids, book.asks):
                for q in side.values():
                    for o in q:
                        if o.owner == BOT:
                            out.append({"market_id": mid, "side": o.side.value, "price": o.tick / TICKS,
                                        "qty": o.qty, "client_id": o.client_id})
        return out

    # ------------------------------------------------ participant interface
    def cancel_owner(self, owner: str, market_id: str = "", older_than: Optional[int] = None) -> int:
        n = 0
        for mid, book in self.books.items():
            if market_id and mid != market_id:
                continue
            if older_than is None:
                n += book.remove_where(lambda o: o.owner == owner)
            else:
                n += book.remove_where(lambda o: o.owner == owner and o.born <= older_than)
        return n

    def submit(self, owner: str, req: OrderRequest) -> OrderResult:
        m = self.markets.get(req.market_id)
        if m is None or m.closed:
            return self._reject(req, "market closed or unknown")
        if req.qty <= 0:
            return self._reject(req, "qty must be positive")
        if req.kind == OrderKind.MARKET:
            tick = TICKS - 1 if req.side == Side.BUY else 1
        else:
            if req.price is None:
                return self._reject(req, "limit order needs a price")
            tick = int(round(req.price * TICKS))
            if tick < 1 or tick > TICKS - 1:
                return self._reject(req, "price out of range")
        qty = req.qty
        if owner == BOT:
            qty = self._bot_capacity(req.market_id, req.side, qty, tick)
            if qty <= 0:
                return self._reject(req, "insufficient balance, position limit, or shorting disabled")
        book = self.books[req.market_id]
        # self-trade prevention: cancel this owner's resting orders that the new order would cross
        if req.side == Side.BUY:
            book.remove_where(lambda o: o.owner == owner and o.side == Side.SELL and o.tick <= tick)
        else:
            book.remove_where(lambda o: o.owner == owner and o.side == Side.BUY and o.tick >= tick)
        fills: List[Fill] = []
        remaining = qty
        if req.side == Side.BUY:
            while remaining > 1e-9 and book.asks:
                best = min(book.asks)
                if best > tick:
                    break
                remaining = self._match_level(book.asks, best, owner, req, remaining, fills)
        else:
            while remaining > 1e-9 and book.bids:
                best = max(book.bids)
                if best < tick:
                    break
                remaining = self._match_level(book.bids, best, owner, req, remaining, fills)
        rests = req.tif == TimeInForce.GTC and req.kind == OrderKind.LIMIT and remaining > 1e-9
        if rests:
            book.add(Resting(next(self._oid), owner, req.side, tick, remaining, req.client_id, self.tick_index))
        filled = qty - remaining
        avg = sum(f.qty * f.price for f in fills) / filled if filled > 0 else 0.0
        if filled >= qty - 1e-9:
            status = OrderStatus.FILLED
        elif rests:
            status = OrderStatus.PARTIAL if filled > 0 else OrderStatus.ACKED
        else:
            status = OrderStatus.PARTIAL if filled > 0 else OrderStatus.CANCELLED
        return OrderResult(req.client_id, status, filled, avg, fills=fills)

    # -------------------------------------------------------------- internals
    def _reject(self, req: OrderRequest, why: str) -> OrderResult:
        self.rejects += 1
        return OrderResult(req.client_id, OrderStatus.REJECTED, error=why)

    def short_liability(self, acct: Optional[Account] = None) -> float:
        acct = acct or self.account
        return sum(-q for q in acct.positions.values() if q < 0)

    def _bot_capacity(self, market_id: str, side: Side, qty: float, tick: int) -> float:
        """Largest qty the bot can trade given cash, collateral, limits (checked at limit price,
        which is the worst case)."""
        pos = self.account.pos(market_id)
        price = tick / TICKS
        free = self.account.cash - self.short_liability()
        if side == Side.BUY:
            cover = min(qty, max(0.0, -pos))            # covering a short releases 1.0 collateral each
            per_unit_new = price
            # covering: pay price, release 1 of liability -> frees (1 - price)
            new = qty - cover
            cap_new = max(0.0, (free + cover * (1 - price)) / per_unit_new) if per_unit_new > 0 else new
            new = min(new, cap_new, max(0.0, self.position_limit - max(pos, 0.0)))
            return cover + new
        else:
            close = min(qty, max(0.0, pos))
            new = qty - close
            if not self.allow_short:
                new = 0.0
            per_unit_new = 1 - price                    # receive price, owe 1.0 collateral
            cap_new = max(0.0, (free + close * price) / per_unit_new) if per_unit_new > 0 else new
            new = min(new, cap_new, max(0.0, self.position_limit - max(-pos, 0.0)))
            return close + new

    def _match_level(self, side_book: Dict[int, Deque[Resting]], tick: int, owner: str,
                     req: OrderRequest, remaining: float, fills: List[Fill]) -> float:
        q = side_book[tick]
        while q and remaining > 1e-9:
            maker = q[0]
            take = min(remaining, maker.qty)
            price = tick / TICKS
            self._settle(owner, req.side, req.market_id, take, price, req.client_id)
            self._settle(maker.owner, Side.SELL if req.side == Side.BUY else Side.BUY,
                         req.market_id, take, price, maker.client_id)
            if owner == BOT:
                fills.append(Fill(req.market_id, req.side, take, price, req.client_id, self.now))
            maker.qty -= take
            remaining -= take
            if maker.qty <= 1e-9:
                q.popleft()
        if not q:
            del side_book[tick]
        return remaining

    def _settle(self, owner: str, side: Side, market_id: str, qty: float, price: float, cid: str) -> None:
        if owner == BOT:
            sgn = 1 if side == Side.BUY else -1
            self.account.cash -= sgn * qty * price
            newpos = self.account.pos(market_id) + sgn * qty
            if abs(newpos) < 1e-9:
                self.account.positions.pop(market_id, None)
            else:
                self.account.positions[market_id] = newpos
            self.fills.append(Fill(market_id, side, qty, price, cid, self.now))
        else:
            key = owner
            self.counterparty_volume[key] = self.counterparty_volume.get(key, 0.0) + qty

    def resolve(self, market_id: str, yes: bool) -> None:
        """Pay out and close a market."""
        m = self.markets[market_id]
        m.closed = True
        self.books[market_id] = Book()
        q = self.account.positions.pop(market_id, 0.0)
        if yes:
            self.account.cash += q  # longs receive 1, shorts pay 1
