"""Rule-based strategy: fair values + books + account -> desired orders.

Deterministic, no LLM, no I/O. Rules, in order:
  1. Take mispricings: buy when ask <= fair - threshold, sell when bid >= fair + threshold,
     and only when the expected return on the capital tied up until settlement is at
     least min_roc (stops 97c buys of 99% favourites that lock up cash for 1-2 points).
     Threshold = min_edge + uncertainty_mult * fair-value uncertainty. The limit
     price is the worst price that still clears the threshold, so the order sweeps
     stale student orders and stops before the edge is gone. Size = fractional
     Kelly, capped per market and per order. After a fair-value jump the target is
     halved until the move has held for settle_s ("take first moves small").
  2. Exit when a position's edge has reversed by more than exit_edge.
  3. Election night capital recycling: sell near-certain winners (bid >= recycle_price)
     when there is a better use for the cash elsewhere.
  4. Mutually exclusive sets (balance-of-power combos) whose prices don't sum to 1.
  5. Optional passive quotes around fair value.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

from .config import StrategyConfig
from .fairvalue import FairQuote
from .mathutil import kelly_no, kelly_yes
from .types import Account, BookMap, Market, OrderKind, OrderRequest, Side, TimeInForce

TICK = 0.005   # the Cup tick (was 0.01, which skipped every half-cent price)
MIN_QTY = 1.0


def floor_tick(p: float) -> float:
    return max(TICK, math.floor(p / TICK + 1e-9) * TICK)


def ceil_tick(p: float) -> float:
    return min(1 - TICK, math.ceil(p / TICK - 1e-9) * TICK)


class Strategy:
    def __init__(self, cfg: StrategyConfig, markets: Dict[str, Market]):
        self.cfg = cfg
        self.markets = markets

    def phase_params(self, phase: str):
        c = self.cfg
        if phase == "night":
            return c.night_kelly_mult, c.night_max_market_frac, c.night_min_edge
        if phase == "final_week":
            return c.final_week_kelly_mult, c.final_week_max_market_frac, c.min_edge
        return c.kelly_mult, c.max_market_frac, c.min_edge

    def threshold(self, mid: str, fq: FairQuote, min_edge: float, venue_gap=None, anchor_moves=None) -> float:
        """Edge needed to trade. Real-data rules (tuning/RECOMMENDATIONS.md):
        lag trade (the anchor just moved >= 2 ticks): lag_min_edge;
        otherwise max(min_edge, Kalshi-Polymarket gap + gap_buffer). Plus a share of the
        fair value's own uncertainty."""
        c = self.cfg
        moved = (anchor_moves or {}).get(mid, 0.0)
        if abs(moved) >= c.lag_move and ("anchor" in fq.parts or not fq.parts):
            base = min(min_edge, c.lag_min_edge)
        else:
            base = max(min_edge, (venue_gap or {}).get(mid, 0.0) + c.gap_buffer) \
                if (venue_gap or {}).get(mid) is not None else min_edge
        return base + c.uncertainty_mult * fq.unc

    def decide(self, fairs: Dict[str, FairQuote], books: BookMap, acct: Account, equity: float,
               phase: str, prev_fairs: Optional[Dict[str, FairQuote]] = None,
               kelly_scale: float = 1.0, venue_gap: Optional[Dict[str, float]] = None,
               anchor_moves: Optional[Dict[str, float]] = None,
               damp: Optional[Dict[str, float]] = None) -> List[OrderRequest]:
        c = self.cfg
        if phase == "closed" or equity <= 0:
            return []
        kmult, mfrac, min_edge = self.phase_params(phase)
        kmult *= kelly_scale
        prev_fairs = prev_fairs or {}
        orders: List[OrderRequest] = []
        edges: Dict[str, float] = {}

        for mid, fq in fairs.items():
            book = books.get(mid)
            m = self.markets.get(mid)
            if book is None or m is None or m.closed:
                continue
            ask, bid = book.best_ask, book.best_bid
            e = max((fq.p - ask) if ask is not None else -1, (bid - fq.p) if bid is not None else -1)
            edges[mid] = e

        best_alt = max(edges.values()) if edges else 0.0

        for mid, fq in fairs.items():
            book = books.get(mid)
            m = self.markets.get(mid)
            if book is None or m is None or m.closed:
                continue
            pos = acct.pos(mid)
            thr = self.threshold(mid, fq, min_edge, venue_gap, anchor_moves)
            if damp is not None:
                size_mult = damp.get(mid, 1.0)
            else:
                prev = prev_fairs.get(mid)
                size_mult = c.jump_damp if prev is not None and abs(fq.p - prev.p) > c.jump_threshold else 1.0
            ask, bid = book.best_ask, book.best_bid
            max_order = c.max_order_frac * equity

            # 1a. buy YES
            if ask is not None and fq.p - ask >= thr - 1e-9:
                # also demand min_roc return on the capital tied up: limit <= fair / (1 + min_roc)
                limit = floor_tick(min(fq.p - thr, fq.p / (1 + c.min_roc)))
                if limit >= ask - 1e-9:
                    f = min(kmult * kelly_yes(fq.p, limit), mfrac)
                    target_qty = f * equity / limit * size_mult
                    want = target_qty - max(pos, 0.0) + max(-pos, 0.0)
                    qty = min(want, book.depth_to(Side.BUY, limit), max_order / limit)
                    if qty >= MIN_QTY:
                        orders.append(OrderRequest(mid, Side.BUY, math.floor(qty), limit,
                                                   reason=f"take: fair {fq.p:.3f} ask {ask:.2f} thr {thr:.3f}"))
                        continue
            # 1b. sell YES (buy NO)
            if bid is not None and bid - fq.p >= thr - 1e-9:
                # NO costs 1 - limit and returns limit - fair: limit >= (fair + min_roc) / (1 + min_roc)
                limit = ceil_tick(max(fq.p + thr, (fq.p + c.min_roc) / (1 + c.min_roc)))
                if limit <= bid + 1e-9:
                    f = min(kmult * kelly_no(fq.p, limit), mfrac)
                    target_qty = f * equity / (1 - limit) * size_mult
                    want = target_qty - max(-pos, 0.0) + max(pos, 0.0)
                    qty = min(want, book.depth_to(Side.SELL, limit), max_order / (1 - limit))
                    if qty >= MIN_QTY:
                        orders.append(OrderRequest(mid, Side.SELL, math.floor(qty), limit,
                                                   reason=f"take: fair {fq.p:.3f} bid {bid:.2f} thr {thr:.3f}"))
                        continue
            # 2. exits when the edge has reversed
            exit_edge = 0.5 * min_edge
            if pos > 0 and bid is not None and bid - fq.p >= exit_edge:
                limit = ceil_tick(fq.p + exit_edge)
                q = min(pos, book.depth_to(Side.SELL, limit))
                if q >= MIN_QTY and limit <= bid + 1e-9:
                    orders.append(OrderRequest(mid, Side.SELL, math.floor(q), limit,
                                               reason=f"exit long: fair {fq.p:.3f} bid {bid:.2f}"))
                    continue
            if pos < 0 and ask is not None and fq.p - ask >= exit_edge:
                limit = floor_tick(fq.p - exit_edge)
                q = min(-pos, book.depth_to(Side.BUY, limit))
                if q >= MIN_QTY and limit >= ask - 1e-9:
                    orders.append(OrderRequest(mid, Side.BUY, math.floor(q), limit,
                                               reason=f"exit short: fair {fq.p:.3f} ask {ask:.2f}"))
                    continue
            # 3. election-night recycling of near-certain positions
            if phase == "night" and best_alt >= c.recycle_min_alt_edge:
                if pos > 0 and bid is not None and bid >= c.recycle_price and fq.p - bid < best_alt:
                    q = min(pos, book.depth_to(Side.SELL, bid))
                    if q >= MIN_QTY:
                        orders.append(OrderRequest(mid, Side.SELL, math.floor(q), bid,
                                                   reason=f"recycle: bid {bid:.2f}, better edge {best_alt:.3f} elsewhere"))
                        continue
                if pos < 0 and ask is not None and ask <= 1 - c.recycle_price and ask - fq.p < best_alt:
                    q = min(-pos, book.depth_to(Side.BUY, ask))
                    if q >= MIN_QTY:
                        orders.append(OrderRequest(mid, Side.BUY, math.floor(q), ask,
                                                   reason=f"recycle short: ask {ask:.2f}"))
                        continue
            # 5. passive quotes
            if c.passive_quotes and phase in ("pre", "final_week"):
                size_notional = c.passive_frac * equity
                bpx = floor_tick(fq.p - c.passive_edge - fq.unc)
                apx = ceil_tick(fq.p + c.passive_edge + fq.unc)
                if ask is None or bpx < ask:
                    orders.append(OrderRequest(mid, Side.BUY, math.floor(size_notional / bpx), bpx,
                                               tif=TimeInForce.GTC, reason="passive bid"))
                if bid is None or apx > bid:
                    orders.append(OrderRequest(mid, Side.SELL, math.floor(size_notional / (1 - apx)), apx,
                                               tif=TimeInForce.GTC, reason="passive ask"))

        orders.extend(self._combo_arbs(books, equity))
        return [o for o in orders if o.qty >= MIN_QTY]

    def _combo_arbs(self, books: BookMap, equity: float) -> List[OrderRequest]:
        c = self.cfg
        groups: Dict[str, List[str]] = {}
        for mid, m in self.markets.items():
            if m.group and not m.closed:
                groups.setdefault(m.group, []).append(mid)
        out: List[OrderRequest] = []
        for g, mids in groups.items():
            bks = [books.get(x) for x in mids]
            if any(b is None for b in bks):
                continue
            asks = [b.best_ask for b in bks]
            bids = [b.best_bid for b in bks]
            if all(a is not None for a in asks) and sum(asks) <= 1 - c.combo_arb_margin:
                q = min(min(b.asks[0].qty for b in bks), c.max_order_frac * equity / sum(asks))
                if q >= MIN_QTY:
                    out += [OrderRequest(x, Side.BUY, math.floor(q), a, reason=f"combo arb {g}: asks sum {sum(asks):.2f}")
                            for x, a in zip(mids, asks)]
            elif all(b is not None for b in bids) and sum(bids) >= 1 + c.combo_arb_margin:
                q = min(min(b.bids[0].qty for b in bks), c.max_order_frac * equity / max(1e-9, len(bids) - sum(bids)))
                if q >= MIN_QTY:
                    out += [OrderRequest(x, Side.SELL, math.floor(q), b, reason=f"combo arb {g}: bids sum {sum(bids):.2f}")
                            for x, b in zip(mids, bids)]
        return out
