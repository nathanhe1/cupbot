"""Shared data types. Prices are always floats in [0, 1] for one YES contract
paying 1.0 if the market resolves YES. Exchange adapters convert to and from the
venue's own format (cents, Decimal strings, ...)."""

from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


class Side(str, enum.Enum):
    BUY = "buy"    # buy YES
    SELL = "sell"  # sell YES (closes a long, or opens a short / buys NO if the venue allows)


class OrderKind(str, enum.Enum):
    LIMIT = "limit"
    MARKET = "market"


class TimeInForce(str, enum.Enum):
    GTC = "gtc"  # rests on the book
    IOC = "ioc"  # fill what you can immediately, cancel the rest


class OrderStatus(str, enum.Enum):
    NEW = "new"
    PENDING = "pending"
    ACKED = "acked"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    UNKNOWN = "unknown"  # timed out: never blindly resend, reconcile instead


@dataclass
class Market:
    market_id: str
    title: str
    kind: str = "race"             # race | chamber | combo | other
    race_id: Optional[str] = None  # links a market to a race in the model
    d_sign: int = 0                # +1 if YES = Democratic outcome, -1 if Republican, 0 unknown
    group: Optional[str] = None    # mutually exclusive set id (combos), if any
    anchor_keys: Dict[str, str] = field(default_factory=dict)  # {"kalshi": ticker, "polymarket": slug}
    closed: bool = False


@dataclass
class Level:
    price: float
    qty: float


@dataclass
class OrderBook:
    market_id: str
    bids: List[Level] = field(default_factory=list)  # best first (highest)
    asks: List[Level] = field(default_factory=list)  # best first (lowest)
    ts: float = field(default_factory=time.time)

    @property
    def best_bid(self) -> Optional[float]:
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> Optional[float]:
        return self.asks[0].price if self.asks else None

    @property
    def mid(self) -> Optional[float]:
        if self.bids and self.asks:
            return (self.bids[0].price + self.asks[0].price) / 2
        return self.best_bid if self.bids else self.best_ask

    def depth_to(self, side: Side, limit_price: float) -> float:
        """Quantity available to a taker on `side` at prices no worse than limit_price."""
        total = 0.0
        if side == Side.BUY:
            for lv in self.asks:
                if lv.price <= limit_price + 1e-12:
                    total += lv.qty
        else:
            for lv in self.bids:
                if lv.price >= limit_price - 1e-12:
                    total += lv.qty
        return total


@dataclass
class OrderRequest:
    market_id: str
    side: Side
    qty: float
    price: Optional[float]  # limit price; None only for MARKET
    kind: OrderKind = OrderKind.LIMIT
    tif: TimeInForce = TimeInForce.IOC
    client_id: str = ""
    reason: str = ""        # human-readable why, for the journal


@dataclass
class Fill:
    market_id: str
    side: Side
    qty: float
    price: float
    client_id: str = ""
    ts: float = field(default_factory=time.time)


@dataclass
class OrderResult:
    client_id: str
    status: OrderStatus
    filled_qty: float = 0.0
    avg_price: float = 0.0
    venue_order_id: Optional[str] = None
    error: Optional[str] = None
    fills: List[Fill] = field(default_factory=list)


@dataclass
class Account:
    cash: float
    positions: Dict[str, float] = field(default_factory=dict)  # market_id -> signed YES qty

    def pos(self, market_id: str) -> float:
        return self.positions.get(market_id, 0.0)

    def mark_to(self, marks: Dict[str, float]) -> float:
        """Equity with positions marked at the given prices (missing marks -> 0.5)."""
        eq = self.cash
        for mid, q in self.positions.items():
            eq += q * marks.get(mid, 0.5)
        return eq


@dataclass
class Signal:
    """A capped probability shift proposed by the news analyst (never an order)."""
    market_id: str
    delta: float            # shift in probability points, e.g. -0.03
    confidence: str         # low | medium | high
    reason: str
    sources: List[str]
    ts: float = field(default_factory=time.time)
    ttl_s: float = 6 * 3600  # decays linearly to zero over this window


PriceMap = Dict[str, float]
BookMap = Dict[str, OrderBook]
Edge = Tuple[str, float]
