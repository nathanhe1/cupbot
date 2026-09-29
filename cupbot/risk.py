"""Hard pre-trade risk checks. Every order passes through `check()`; nothing
(including the news analyst) can bypass it. Orders that only reduce an existing
position are always allowed unless the kill switch is on."""

from __future__ import annotations

import collections
import os
import time
from dataclasses import dataclass
from typing import Deque, Dict, Optional, Tuple

from .config import RiskConfig
from .types import Account, Market, OrderBook, OrderRequest, Side


@dataclass
class RiskState:
    equity_high_24h: float = 0.0
    halted: bool = False
    halt_reason: str = ""


class RiskManager:
    def __init__(self, cfg: RiskConfig, markets: Dict[str, Market], run_dir: str = "."):
        self.cfg = cfg
        self.markets = markets
        self.run_dir = run_dir
        self.state = RiskState()
        self._order_times: Deque[float] = collections.deque()
        self._equity_marks: Deque[Tuple[float, float]] = collections.deque()
        self.blocked: Dict[str, int] = collections.Counter()
        self.clock = time.time
        self.relax_price_band = False  # set on election night: prices legitimately jump

    # ----------------------------------------------------------- kill switch
    def killed(self) -> bool:
        return os.path.exists(os.path.join(self.run_dir, self.cfg.kill_file))

    # ------------------------------------------------------------- equity
    def update_equity(self, equity: float, now: Optional[float] = None) -> None:
        now = self.clock() if now is None else now
        self._equity_marks.append((now, equity))
        while self._equity_marks and self._equity_marks[0][0] < now - 86400:
            self._equity_marks.popleft()
        hi = max(e for _, e in self._equity_marks)
        if hi > 0 and (hi - equity) / hi >= self.cfg.daily_loss_halt_frac:
            if not self.state.halted:
                self.state.halted = True
                self.state.halt_reason = f"drawdown {(hi - equity) / hi:.0%} in 24h"
        elif self.state.halted and self.state.halt_reason.startswith("drawdown"):
            # stays halted until a human clears it
            pass

    def clear_halt(self) -> None:
        self.state = RiskState()

    # -------------------------------------------------------------- checks
    def check(self, req: OrderRequest, acct: Account, book: Optional[OrderBook],
              marks: Dict[str, float], equity: float, now: Optional[float] = None) -> Tuple[bool, str]:
        now = self.clock() if now is None else now
        ok, why = self._check(req, acct, book, marks, equity, now)
        if not ok:
            self.blocked[why.split(":")[0]] += 1
        else:
            self._order_times.append(now)
        return ok, why

    def _check(self, req, acct, book, marks, equity, now):
        c = self.cfg
        if self.killed():
            return False, "kill switch"
        m = self.markets.get(req.market_id)
        if m is None or m.closed:
            return False, "unknown or closed market"
        if req.price is None or not (0.0 < req.price < 1.0):
            return False, "price: must be a limit strictly inside (0, 1)"
        if req.qty <= 0:
            return False, "qty: non-positive"
        pos = acct.pos(req.market_id)
        reducing = (req.side == Side.SELL and pos > 0 and req.qty <= pos + 1e-9) or \
                   (req.side == Side.BUY and pos < 0 and req.qty <= -pos + 1e-9)
        if reducing:
            return True, "reduce-only"
        if self.state.halted:
            return False, f"halted: {self.state.halt_reason}"
        while self._order_times and self._order_times[0] < now - 60:
            self._order_times.popleft()
        if len(self._order_times) >= c.max_orders_per_minute:
            return False, "rate: orders per minute"
        if book is not None and c.max_book_age_s > 0 and now - book.ts > c.max_book_age_s:
            return False, "stale: book too old"
        unit = req.price if req.side == Side.BUY else 1 - req.price
        notional = req.qty * unit
        if notional > c.max_order_notional:
            return False, "size: order notional"
        if equity <= 0:
            return False, "equity: non-positive"
        mid = book.mid if book is not None else None
        if mid is not None and not self.relax_price_band and abs(req.price - mid) > c.max_price_deviation:
            return False, "price: outside sanity band"
        new_pos = pos + (req.qty if req.side == Side.BUY else -req.qty)
        px = marks.get(req.market_id, req.price)
        mkt_notional = abs(new_pos) * (px if new_pos > 0 else 1 - px)
        if mkt_notional > c.max_market_notional_frac * equity:
            return False, "size: market notional"
        after = dict(acct.positions)
        after[req.market_id] = new_pos
        gross_after, net_after = self._exposure(after, marks)
        _, net_before = self._exposure(acct.positions, marks)
        if gross_after > c.max_gross_frac * equity:
            return False, "size: gross exposure"
        if abs(net_after) > c.max_net_d_frac * equity and abs(net_after) >= abs(net_before):
            return False, "size: net directional exposure"
        return True, "ok"

    def _exposure(self, positions: Dict[str, float], marks: Dict[str, float]) -> Tuple[float, float]:
        """(gross notional at risk, net Democratic-direction notional)."""
        gross = net_d = 0.0
        for mid, q in positions.items():
            if not q:
                continue
            p = marks.get(mid, 0.5)
            val = abs(q) * (p if q > 0 else 1 - p)
            gross += val
            mk = self.markets.get(mid)
            if mk is not None and mk.d_sign:
                net_d += mk.d_sign * (1 if q > 0 else -1) * val
        return gross, net_d
