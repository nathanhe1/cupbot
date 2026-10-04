"""Resting-order ladder: standing bids below and asks above the Cup's consensus price.

Why (Oct 3-4 analysis of the Cup tapes): the field trades with market orders into
books that are deep only near the touch, so prints land 3-10 points off the 1 h VWAP
and revert within ~30 minutes (+3.0 pts after a 3-pt print, +6.2 after a 10-pt one).
The top of the leaderboard got there by being the resting side of that flow, at
volumes of 40-240x their balance. A resting order catches it without any latency race.

Per market (ranked list in data/ladder_markets.json):
  consensus  = EMA of the Cup mid (tight books only), seeded from the 5 h VWAP, which
               is also what the scoreboard marks positions at
  bid levels = consensus - offsets, ask levels = consensus + offsets (0.005 tick, never
               crossing the book)
  fair guard = where Kalshi/Polymarket anchor the fair value, a level may only OPEN a
               position on the right side of fair (longs below fair - guard, shorts
               above fair + guard); levels that only REDUCE the position ignore fair, so
               inventory turns over at the consensus instead of sitting to November
  sizing     = a share of `total_frac` x equity per side, by rank; no level can take the
               ladder past `max_pos_frac` x equity in that market
Each requote is one cancel-all for that exchange plus a shared batch create, under its
own write budget. Every order goes through RiskManager.check and expires venue-side.
"""

from __future__ import annotations

import json
import math
import os
import time
from typing import Dict, List, Optional, Tuple

from .config import LadderConfig
from .fairvalue import FairQuote
from .types import Account, BookMap, Market, OrderRequest, Side, TimeInForce

TICK = 0.005
Quote = Tuple[str, float, int]   # (side, price, qty)


def bid_tick(p: float) -> float:
    return round(max(TICK, math.floor(p / TICK + 1e-9) * TICK), 3)


def ask_tick(p: float) -> float:
    return round(min(1 - TICK, math.ceil(p / TICK - 1e-9) * TICK), 3)


def load_ranked(path: str) -> List[str]:
    if not path or not os.path.exists(path):
        return []
    with open(path) as f:
        data = json.load(f)
    return [str(m["id"]) for m in data.get("markets", [])]


class Ladder:
    def __init__(self, cfg: LadderConfig, markets: Dict[str, Market], journal, clock=time.time):
        self.cfg = cfg
        self.markets = markets
        self.journal = journal
        self.clock = clock
        ranked = [m for m in load_ranked(cfg.markets_file) if m in markets] or sorted(markets)
        self.ids = ranked[:cfg.max_markets]
        # rank weights: sqrt so the busiest markets get more without starving the rest
        w = {m: 1.0 / math.sqrt(i + 1) for i, m in enumerate(self.ids)}
        tot = sum(w.values()) or 1.0
        self.share = {m: v / tot for m, v in w.items()}
        self.ref: Dict[str, float] = {}
        self._ref_t: Dict[str, float] = {}
        self.live: Dict[str, dict] = {}        # market -> {"quotes", "ref", "fair", "pos", "t"}
        self.last_pos: Dict[str, float] = {}
        self._tokens = cfg.writes_per_min
        self._tok_t = clock()

    # ------------------------------------------------------------ consensus
    def seed(self, vwaps: Dict[str, Optional[float]]) -> None:
        now = self.clock()
        for m, v in vwaps.items():
            if v is not None and 0 < v < 1:
                self.ref[m] = v
                self._ref_t[m] = now

    def update_refs(self, books: BookMap) -> None:
        now = self.clock()
        for m in self.ids:
            b = books.get(m)
            if b is None or b.best_bid is None or b.best_ask is None:
                continue
            if b.best_ask - b.best_bid > self.cfg.max_spread_for_mid:
                continue
            mid = (b.best_bid + b.best_ask) / 2
            if m not in self.ref:
                self.ref[m], self._ref_t[m] = mid, now
                continue
            dt = max(0.0, now - self._ref_t[m])
            a = 1 - 0.5 ** (dt / self.cfg.ema_half_life_s)
            self.ref[m] += a * (mid - self.ref[m])
            self._ref_t[m] = now

    # ------------------------------------------------------------ quotes
    def desired(self, m: str, fq: Optional[FairQuote], pos: float, equity: float, book) -> List[Quote]:
        c = self.cfg
        ref = self.ref.get(m)
        if ref is None or book is None or equity <= 0:
            return []
        if book.mid is not None and abs(book.mid - ref) > c.pause_dev:
            return []   # the market has moved away from its consensus: don't stand in front of it
        anchored = fq is not None and "anchor" in fq.parts
        offs = c.offsets if anchored else c.offsets_unanchored
        side_budget = c.total_frac * equity * self.share.get(m, 0.0)
        cap = c.max_pos_frac * equity
        out: List[Quote] = []
        # bids: buying YES. Room = how far the ladder may still go long (reducing a short first).
        long_val = max(pos, 0.0) * ref
        room = cap - long_val
        short_qty = max(-pos, 0.0)
        for off, w in zip(offs, c.weights):
            px = bid_tick(ref - off)
            if px < TICK or (book.best_ask is not None and px >= book.best_ask):
                continue
            qty = int(side_budget * w / px)
            reduce_q = min(qty, int(short_qty))
            open_q = qty - reduce_q
            if open_q and anchored and px > fq.p - c.fair_guard:
                open_q = 0            # would open a long above fair
            open_q = max(0, min(open_q, int(room / px)))
            q = reduce_q + open_q
            short_qty -= reduce_q
            room -= open_q * px
            if q >= c.min_qty:
                out.append(("buy", px, q))
        # asks: selling YES (buying NO at 1 - price)
        short_val = max(-pos, 0.0) * (1 - ref)
        room = cap - short_val
        long_qty = max(pos, 0.0)
        for off, w in zip(offs, c.weights):
            px = ask_tick(ref + off)
            if px > 1 - TICK or (book.best_bid is not None and px <= book.best_bid):
                continue
            qty = int(side_budget * w / (1 - px))
            reduce_q = min(qty, int(long_qty))
            open_q = qty - reduce_q
            if open_q and anchored and px < fq.p + c.fair_guard:
                open_q = 0            # would open a short below fair
            open_q = max(0, min(open_q, int(room / (1 - px))))
            q = reduce_q + open_q
            long_qty -= reduce_q
            room -= open_q * (1 - px)
            if q >= c.min_qty:
                out.append(("sell", px, q))
        return out

    def _needs_requote(self, m: str, want: List[Quote], fq: Optional[FairQuote], pos: float, now: float) -> float:
        """0 = leave it; otherwise a priority (higher first)."""
        cur = self.live.get(m)
        if cur is None:
            return 1.0 if want else 0.0
        c = self.cfg
        if not want:
            return 3.0 if cur["quotes"] else 0.0
        pri = 0.0
        if fq is not None and cur["fair"] is not None and abs(fq.p - cur["fair"]) >= c.requote_move:
            pri = max(pri, 3.0 + abs(fq.p - cur["fair"]))     # fair moved: get out of the way first
        if abs(self.ref[m] - cur["ref"]) >= c.requote_move:
            pri = max(pri, 2.0)
        if abs(pos - cur["pos"]) >= max(c.min_qty, 0.2 * sum(q for _, _, q in cur["quotes"]) / max(1, len(cur["quotes"]))):
            pri = max(pri, 2.5)                                  # filled: rebalance the sides
        if now - cur["t"] >= c.requote_age_s:
            pri = max(pri, 1.0)
        return pri

    def _refill(self, now: float) -> None:
        self._tokens = min(self.cfg.writes_per_min, self._tokens + (now - self._tok_t) * self.cfg.writes_per_min / 60)
        self._tok_t = now

    def plan(self, fairs: Dict[str, FairQuote], books: BookMap, acct: Account, equity: float
             ) -> Tuple[List[str], Dict[str, List[Quote]]]:
        """-> (markets to cancel and requote this step, their new quotes). Spends write tokens:
        one cancel per market plus one batch per 50 orders."""
        now = self.clock()
        self._refill(now)
        for m in self.ids:   # log fills of resting orders (seen as position changes)
            p = acct.pos(m)
            if m in self.last_pos and abs(p - self.last_pos[m]) >= 1:
                self.journal.log("ladder_fill", {"market": m, "delta": round(p - self.last_pos[m]),
                                                 "pos": round(p), "ref": round(self.ref.get(m, 0), 3)})
            self.last_pos[m] = p
        cands = []
        wants: Dict[str, List[Quote]] = {}
        for m in self.ids:
            mk = self.markets.get(m)
            if mk is None or mk.closed:
                continue
            want = self.desired(m, fairs.get(m), acct.pos(m), equity, books.get(m))
            pri = self._needs_requote(m, want, fairs.get(m), acct.pos(m), now)
            if pri > 0:
                cands.append((pri, m))
                wants[m] = want
        cands.sort(reverse=True)
        chosen: List[str] = []
        n_orders = 0
        for _, m in cands:
            extra = len(wants[m])
            batches = math.ceil((n_orders + extra) / 50) if n_orders + extra else 0
            cost = len(chosen) + 1 + batches
            if cost > self._tokens:
                break
            chosen.append(m)
            n_orders += extra
        if chosen:
            self._tokens -= len(chosen) + math.ceil(n_orders / 50)
        return chosen, {m: wants[m] for m in chosen}

    def orders_for(self, quotes: Dict[str, List[Quote]], now: float, cid) -> List[OrderRequest]:
        out = []
        for m, qs in quotes.items():
            for side, px, q in qs:
                out.append(OrderRequest(m, Side.BUY if side == "buy" else Side.SELL, q, px, tif=TimeInForce.GTC,
                                        client_id=f"lad-{int(now)}-{next(cid)}",
                                        reason=f"ladder: ref {self.ref.get(m, 0):.3f}"))
        return out

    def mark_quoted(self, m: str, quotes: List[Quote], fq: Optional[FairQuote], pos: float, now: float) -> None:
        self.live[m] = {"quotes": quotes, "ref": self.ref.get(m, 0.0), "fair": fq.p if fq else None,
                        "pos": pos, "t": now}
