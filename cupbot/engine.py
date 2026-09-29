"""The trading loop: data -> fair value -> strategy -> risk -> orders -> journal.

`step()` runs one full cycle. The simulator calls it once per simulated tick;
live, `run()` calls it on a timer and immediately whenever a feed signals a
change (an anchor move, a new result), so reaction time is bounded by the
network round trip rather than the loop interval."""

from __future__ import annotations

import asyncio
import itertools
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .config import BotConfig
from .exchange.base import Exchange
from .election_night import LiveResultsModel, NightParams
from .fairvalue import FairQuote, FairValueEngine, to_d
from .mathutil import norm_ppf
from .feeds import Feeds
from .journal import NullJournal
from .racemodel import RaceModel
from .risk import RiskManager
from .strategy import Strategy
from .tournament import (best_scale, candidate_baskets, chamber_outcomes, plan_basket,
                         scenario_payoffs)
from .types import OrderRequest
from .types import Account, Market, OrderStatus, Side


@dataclass
class EngineStats:
    steps: int = 0
    orders_sent: int = 0
    orders_blocked: int = 0
    fills: int = 0
    volume: float = 0.0
    reasons: Dict[str, int] = field(default_factory=dict)


class Engine:
    def __init__(self, cfg: BotConfig, exchange: Exchange, feeds: Feeds, model: RaceModel,
                 markets: List[Market], journal=None, run_dir: str = ".", dry_run: bool = False):
        self.cfg = cfg
        self.ex = exchange
        self.feeds = feeds
        self.model = model
        self.markets: Dict[str, Market] = {m.market_id: m for m in markets}
        self.fair = FairValueEngine(cfg.fair, self.markets, model)
        self.strategy = Strategy(cfg.strategy, self.markets)
        self.risk = RiskManager(cfg.risk, self.markets, run_dir)
        self.journal = journal or NullJournal()
        self.dry_run = dry_run
        self.prev_fairs: Dict[str, FairQuote] = {}
        self.last_fairs: Dict[str, FairQuote] = {}
        self.stats = EngineStats()
        self._cid = itertools.count(1)
        self._model_key = None
        self._model_pd: Dict[str, float] = {}
        self.kelly_scale = 1.0          # set by the tournament sizer in the final week
        self.tournament_p = 0.0
        self.basket: Optional[Dict] = None      # {"name", "frac", "p", "targets": {mid: signed qty}}
        self.wake = asyncio.Event() if _has_loop() else None
        self.last_marks: Dict[str, float] = {}
        self.night_model: Optional[LiveResultsModel] = None

    # ------------------------------------------------------------------
    def elapsed_frac(self, now: float) -> float:
        span = self.cfg.election_ts - self.cfg.open_ts
        return min(1.0, max(0.0, (now - self.cfg.open_ts) / span)) if span > 0 else 0.0

    def _model_probs(self, now: float) -> Dict[str, float]:
        means = self.feeds.poll_means()
        scale = (1 - self.cfg.fair.pre_info_frac * self.elapsed_frac(now)) ** 0.5
        self.model.sd_scale = scale
        self.fair.consistency.sd_scale = scale
        key = (round(scale, 3),) + tuple(sorted((k, round(v, 3)) for k, v in means.items()))
        if key != self._model_key:
            for r in self.model.races:
                if r.race_id in means:
                    r.mean = means[r.race_id]
            if self.model.house and self.model.house.race_id in means:
                self.model.house.mean = means[self.model.house.race_id]
            self._model_pd = self.model.summary()
            self._model_key = key
        return self._model_pd

    def _live(self, phase: str):
        """Election night: run our live results model. Its prior for each race is
        the market-implied margin at poll close (our best pre-result estimate),
        not our own poll average."""
        if phase != "night":
            return None
        reports = self.feeds.live_reports()
        if reports is None:
            return None
        if self.night_model is None:
            params = self.feeds.night_params() or NightParams()
            sd_close = params.sd_close()
            prior: Dict[str, float] = {}
            pd_model = self._model_pd
            for r in self.model.races + ([self.model.house] if self.model.house else []):
                p_d = pd_model.get(r.race_id, 0.5)
                for mid, fq in self.prev_fairs.items():
                    mk = self.markets[mid]
                    if mk.kind == "race" and mk.race_id == r.race_id:
                        p_d = to_d(fq.p, mk.d_sign)
                if r.race_id == "HOUSE" and "HOUSE_D" in pd_model:
                    for mid, fq in self.prev_fairs.items():
                        if self.markets[mid].race_id == "HOUSE_D":
                            p_d = to_d(fq.p, self.markets[mid].d_sign)
                prior[r.race_id] = sd_close * norm_ppf(p_d)
            states = {r.race_id: r.state for r in self.model.races}
            self.night_model = LiveResultsModel(prior, states, params)
        return self.night_model.update(reports)

    def _tournament_scale(self, phase: str, acct: Account, marks: Dict[str, float], equity: float) -> None:
        t = self.cfg.tournament
        if not t.enabled or phase not in t.phases or not acct.positions:
            self.kelly_scale = 1.0
            return
        cm = self.fair.consistency   # means already set from current race fair values
        outcomes = chamber_outcomes(cm.outcome_matrix(), cm.params.d_seats_needed - cm.params.d_holdover)
        n = cm.params.n_draws
        pnl = scenario_payoffs(acct.positions, marks, self.markets, outcomes, n)
        grid = [g for g in (0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0) if t.min_scale <= g <= t.max_scale]
        lam, p = best_scale(equity, pnl, t.target_multiple * self.cfg.starting_cash, grid)
        self.kelly_scale = lam
        self.tournament_p = p

    def _basket_orders(self, now, phase, fairs, books, acct, marks, equity, orders):
        """Final days: place (once) and then hold the tournament scenario basket."""
        t = self.cfg.tournament
        if not (t.enabled and t.basket_enabled) or phase not in ("final_week", "night"):
            return orders
        if self.basket is None and phase == "final_week" and \
                now >= self.cfg.election_ts - t.basket_days_before * 86400:
            cm = self.fair.consistency
            outcomes = chamber_outcomes(cm.outcome_matrix(), cm.params.d_seats_needed - cm.params.d_holdover)
            n = cm.params.n_draws
            base = scenario_payoffs(acct.positions, marks, self.markets, outcomes, n)
            asks = {m: b.best_ask for m, b in books.items() if b.best_ask is not None}
            bids = {m: b.best_bid for m, b in books.items() if b.best_bid is not None}
            cands = candidate_baskets(self.markets, {m: f.p for m, f in fairs.items()}, asks, bids,
                                      t.basket_lo, t.basket_hi)
            fr = [f for f in (0.1, 0.2, 0.35, 0.5, 0.75) if f <= t.basket_max_frac]
            name, frac, p, qty = plan_basket(equity, base, t.target_multiple * self.cfg.starting_cash,
                                             cands, self.markets, outcomes, n, fr)
            targets = {m: acct.pos(m) + q for m, q in qty.items()}
            self.basket = {"name": name, "frac": frac, "p": p, "targets": targets}
            self.risk.cfg.max_net_d_frac = max(self.risk.cfg.max_net_d_frac, t.final_max_net_d_frac)
            self.journal.log("basket", {"name": name, "frac": frac, "p_target": p, "legs": len(targets)})
        if not self.basket or not self.basket["targets"]:
            return orders
        targets = self.basket["targets"]
        # never let the edge strategy unwind basket legs before resolution
        keep = []
        for o in orders:
            tq = targets.get(o.market_id)
            if tq is not None and ((tq > 0 and o.side == Side.SELL) or (tq < 0 and o.side == Side.BUY)):
                continue
            keep.append(o)
        if phase != "final_week":
            return keep
        for mid, tq in targets.items():
            b, fq = books.get(mid), fairs.get(mid)
            if b is None or fq is None:
                continue
            gap = tq - acct.pos(mid)
            if gap > 1 and b.best_ask is not None and b.best_ask <= fq.p + t.basket_max_premium:
                q = min(gap, b.depth_to(Side.BUY, b.best_ask))
                if q >= 1:
                    keep.append(OrderRequest(mid, Side.BUY, int(q), b.best_ask,
                                             reason=f"basket: {self.basket['name']}"))
            elif gap < -1 and b.best_bid is not None and b.best_bid >= fq.p - t.basket_max_premium:
                q = min(-gap, b.depth_to(Side.SELL, b.best_bid))
                if q >= 1:
                    keep.append(OrderRequest(mid, Side.SELL, int(q), b.best_bid,
                                             reason=f"basket: {self.basket['name']}"))
        return keep

    async def step(self) -> None:
        now = self.feeds.now()
        phase = self.feeds.phase()
        self.risk.clock = self.feeds.now
        self.risk.relax_price_band = phase == "night"
        self.stats.steps += 1
        if phase == "closed":
            return
        if self.risk.killed():
            n = await self.ex.cancel_all()
            self.journal.log("kill", {"cancelled": n})
            return

        live_ids = [m for m, mk in self.markets.items() if not mk.closed]
        books = await self.ex.get_books(live_ids)
        acct = await self.ex.get_account()
        marks = dict(self.last_marks)
        mids: Dict[str, float] = {}
        for mid, b in books.items():
            if b.mid is not None:
                marks[mid] = b.mid
                mids[mid] = b.mid
        self.last_marks = marks
        equity = acct.mark_to(marks)
        self.risk.update_equity(equity, now)

        model_pd = self._model_probs(now)
        for s in self.feeds.signals():
            if self.fair.add_signal(s):
                self.journal.log("signal", {"market": s.market_id, "delta": s.delta, "reason": s.reason,
                                            "sources": s.sources})
        live = self._live(phase)
        fairs = self.fair.compute(now, model_pd, self.feeds.anchors(), mids, live)

        self._tournament_scale(phase, acct, marks, equity)
        orders = self.strategy.decide(fairs, books, acct, equity, phase, self.prev_fairs, self.kelly_scale)
        orders = self._basket_orders(now, phase, fairs, books, acct, marks, equity, orders)
        if self.cfg.strategy.passive_quotes:
            await self.ex.cancel_all()

        for o in orders:
            o.client_id = f"cb-{int(now)}-{next(self._cid)}"
            ok, why = self.risk.check(o, acct, books.get(o.market_id), marks, equity, now)
            if not ok:
                self.stats.orders_blocked += 1
                self.journal.log("blocked", {"market": o.market_id, "side": o.side.value, "qty": o.qty,
                                             "price": o.price, "why": why, "reason": o.reason})
                continue
            key = o.reason.split(":")[0]
            self.stats.reasons[key] = self.stats.reasons.get(key, 0) + 1
            if self.dry_run:
                self.journal.log("dry_order", {"market": o.market_id, "side": o.side.value, "qty": o.qty,
                                               "price": o.price, "reason": o.reason})
                continue
            res = await self.ex.place_order(o)
            self.stats.orders_sent += 1
            if res.status == OrderStatus.UNKNOWN:
                # never resend blindly: the reconciler fixes state from the venue's view
                self.journal.log("unknown", {"client_id": o.client_id})
            for f in res.fills:
                sgn = 1 if f.side == Side.BUY else -1
                acct.cash -= sgn * f.qty * f.price
                acct.positions[f.market_id] = acct.pos(f.market_id) + sgn * f.qty
                self.stats.fills += 1
                self.stats.volume += f.qty * f.price
            self.journal.log("order", {"client_id": o.client_id, "market": o.market_id, "side": o.side.value,
                                       "qty": o.qty, "price": o.price, "status": res.status.value,
                                       "filled": res.filled_qty, "avg": res.avg_price, "reason": o.reason,
                                       "fair": fairs.get(o.market_id).p if o.market_id in fairs else None})
        self.prev_fairs = fairs
        self.last_fairs = fairs

    async def run(self, stop_after_s: Optional[float] = None) -> None:
        """Live loop: step on a timer, or immediately when a feed calls `poke()`."""
        start = time.time()
        if self.wake is None:
            self.wake = asyncio.Event()
        while True:
            t0 = time.time()
            try:
                await self.step()
            except Exception as e:  # keep running; the journal records the failure
                self.journal.log("error", {"error": repr(e)})
            if stop_after_s is not None and time.time() - start > stop_after_s:
                return
            timeout = max(0.0, self.cfg.loop_interval_s - (time.time() - t0))
            try:
                await asyncio.wait_for(self.wake.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                pass
            self.wake.clear()

    def poke(self) -> None:
        if self.wake is not None:
            self.wake.set()


def _has_loop() -> bool:
    try:
        asyncio.get_running_loop()
        return True
    except RuntimeError:
        return False
