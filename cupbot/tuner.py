"""Unattended self-tuning: nudges min_edge and Kelly size from how recent fills are doing.

Only ever tightens relative to the configured baseline: edge goes up (never below the
config value) and size goes down (never above it). Caps, the drawdown halt and order
expiry are untouched.

Every `interval_s` it scores each fill from the last `window_s` by markout, the
current fair value minus the fill price (signed so positive = we bought cheap / sold
dear). If the average markout per contract is negative the bot is being picked off:
add edge and cut size. If it is comfortably positive, step back toward the baseline.
A drawdown of `dd_frac` from the session's starting equity also halves size."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class TunerConfig:
    interval_s: float = 900.0        # re-tune every 15 min
    window_s: float = 3 * 3600.0     # score fills from the last 3 h
    min_fills: int = 4               # need this many fills before acting
    bad_markout: float = 0.0         # avg markout below this: tighten
    good_markout: float = 0.01       # avg markout above this: relax toward baseline
    edge_step: float = 0.01
    max_edge_add: float = 0.03       # never more than +3 pts over the configured min_edge
    size_step: float = 0.8
    min_size: float = 0.5            # never below half the configured Kelly
    dd_frac: float = 0.03            # session drawdown that forces size to min_size


class Tuner:
    def __init__(self, strategy_cfg, journal, cfg: Optional[TunerConfig] = None):
        self.s = strategy_cfg
        self.journal = journal
        self.cfg = cfg or TunerConfig()
        self.base_min_edge = strategy_cfg.min_edge
        self.base_kelly = strategy_cfg.kelly_mult
        self.edge_add = 0.0
        self.size = 1.0
        self.fills: List[Tuple[float, str, int, float, float]] = []   # (t, market, sign, qty, price)
        self.start_equity: Optional[float] = None
        self._last = time.time()

    def on_fill(self, t: float, market: str, sign: int, qty: float, price: float) -> None:
        self.fills.append((t, market, sign, qty, price))

    def markout(self, fairs: Dict[str, float], now: float) -> Tuple[int, float]:
        cut = now - self.cfg.window_s
        self.fills = [f for f in self.fills if f[0] >= cut]
        n, tot, q = 0, 0.0, 0.0
        for _, mid, sgn, qty, px in self.fills:
            p = fairs.get(mid)
            if p is None:
                continue
            n += 1
            tot += sgn * (p - px) * qty
            q += qty
        return n, (tot / q if q else 0.0)

    def maybe_tune(self, now: float, fairs: Dict[str, float], equity: float) -> None:
        if equity > 0 and self.start_equity is None:
            self.start_equity = equity
        if now - self._last < self.cfg.interval_s:
            return
        self._last = now
        c = self.cfg
        n, mk = self.markout(fairs, now)
        edge_add, size, why = self.edge_add, self.size, "hold"
        if n >= c.min_fills and mk < c.bad_markout:
            edge_add = min(c.max_edge_add, edge_add + c.edge_step)
            size = max(c.min_size, size * c.size_step)
            why = "fills losing to fair: tighten"
        elif n >= c.min_fills and mk > c.good_markout:
            edge_add = max(0.0, edge_add - c.edge_step / 2)
            size = min(1.0, size / c.size_step)
            why = "fills beating fair: relax toward baseline"
        if self.start_equity and equity < self.start_equity * (1 - c.dd_frac):
            size = c.min_size
            why += "; session drawdown: half size"
        self.edge_add, self.size = edge_add, size
        self.s.min_edge = self.base_min_edge + edge_add
        self.s.kelly_mult = self.base_kelly * size
        self.journal.log("tune", {"fills": n, "markout": round(mk, 4), "min_edge": round(self.s.min_edge, 4),
                                  "kelly_mult": round(self.s.kelly_mult, 4), "equity": round(equity),
                                  "why": why})
