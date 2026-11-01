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
from .tuner import Tuner
from .ladder import Ladder
from .tournament import (bar_per_scenario, best_scale, candidate_baskets, chamber_outcomes,
                         plan_basket, scenario_payoffs)
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
        self._field_prices: Optional[Dict[str, float]] = None
        self._logged_calls = set()
        self._moves: Dict[str, tuple] = {}   # market -> (t0, fair before, fair after) of last jump
        self._acct: Optional[Account] = None
        self._acct_t = 0.0
        self.last_equity = 0.0
        self.last_error = ""
        self.last_acct: Optional[Account] = None
        self.unknown: Dict[str, str] = {}       # client_id -> market_id of timed-out creates
        self._unknown_tries: Dict[str, int] = {}
        self.basket: Optional[Dict] = None      # {"name", "frac", "p", "targets": {mid: signed qty}}
        self.wake = asyncio.Event() if _has_loop() else None
        self.last_marks: Dict[str, float] = {}
        self.night_model: Optional[LiveResultsModel] = None
        # live only: nudges min_edge / Kelly from fill markouts (the sim keeps fixed settings)
        self.tuner: Optional[Tuner] = Tuner(cfg.strategy, self.journal) if cfg.self_tune and not dry_run else None
        # live venue only (the simulator's exchange has no batch orders)
        self.ladder: Optional[Ladder] = Ladder(cfg.ladder, self.markets, self.journal) \
            if cfg.ladder.enabled and hasattr(exchange, "place_batch") else None

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
        book = self.feeds.call_book()
        if reports is None:
            return self._apply_calls(None, book)
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
        return self._apply_calls(self.night_model.update(reports), book)

    def _apply_calls(self, live, book):
        """Confirmed race calls pin a race at 0.995 / 0.005 (never certainty)."""
        if book is None:
            return live
        eff = book.effective(live)
        if not eff:
            return live
        live = dict(live or {})
        for rid, party in eff.items():
            live[rid] = (0.995 if party == "D" else 0.005, 0.003)
            if rid not in self._logged_calls:
                self._logged_calls.add(rid)
                self.journal.log("race_call", {"race": rid, "party": party})
        return live

    def _bar(self, outcomes, marks, n):
        """Top-3 bar: per scenario from the field model, or the fixed target."""
        t = self.cfg.tournament
        if not t.field_model:
            return t.target_multiple * self.cfg.starting_cash
        key = (round(self.feeds.now() / 21600),)   # recompute at most every 6 hours
        if getattr(self, "_bar_key", None) != key:
            if self._field_prices is None:
                self._field_prices = dict(marks)   # the crowd's entry prices: first look in the late phase
            self._bar_cache = bar_per_scenario(self._field_prices, self.markets, outcomes, n, t.field_size,
                                               t.field_concentrated_frac, self.cfg.starting_cash)
            self._bar_key = key
        return self._bar_cache

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
        lam, p = best_scale(equity, pnl, self._bar(outcomes, marks, n), grid)
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
            name, frac, p, qty = plan_basket(equity, base, self._bar(outcomes, marks, n),
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

    def _first_move_damp(self, now: float, fairs: Dict[str, FairQuote]) -> Dict[str, float]:
        """Half size on a fresh fair-value move; full size once it has held at least half
        of the move for settle_s; stays at half if it gave back more (moves expire after 1 h)."""
        c = self.cfg.strategy
        out: Dict[str, float] = {}
        for mid, fq in fairs.items():
            prev = self.prev_fairs.get(mid)
            if prev is not None and abs(fq.p - prev.p) >= c.jump_threshold:
                self._moves[mid] = (now, prev.p, fq.p)
            mv = self._moves.get(mid)
            if mv is None:
                continue
            t0, before, after = mv
            if now - t0 > 3600:
                del self._moves[mid]
                continue
            if now - t0 < c.settle_s:
                out[mid] = c.jump_damp
            else:
                held = (fq.p - before) / (after - before) if after != before else 1.0
                out[mid] = 1.0 if held >= 0.5 else c.jump_damp
        return out

    async def _resolve_unknown(self) -> None:
        """Orders whose create timed out: ask the venue what happened, by client id."""
        if not self.unknown or not hasattr(self.ex, "lookup_order"):
            return
        for cid in list(self.unknown):
            tries = self._unknown_tries.get(cid, 0) + 1
            self._unknown_tries[cid] = tries
            try:
                found = await self.ex.lookup_order(cid)
            except Exception as e:
                self.journal.log("error", {"lookup_order": cid, "error": repr(e)})
                continue
            if found is not None or tries >= 5:
                # found: positions come from get_account on the next step; not found after
                # several looks: treat as never placed
                self.journal.log("unknown_resolved", {"client_id": cid, "found": found is not None})
                del self.unknown[cid]

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

        await self._resolve_unknown()
        live_ids = [m for m, mk in self.markets.items() if not mk.closed]
        refresh = self.cfg.account_refresh_s
        if self._acct is None or refresh <= 0 or time.time() - self._acct_t >= refresh:
            books, acct = await asyncio.gather(self.ex.get_books(live_ids), self.ex.get_account())
            self._acct, self._acct_t = acct, time.time()
        else:
            books = await self.ex.get_books(live_ids)
            acct = self._acct   # tracked locally from fills until the next refresh
        marks = dict(self.last_marks)
        mids: Dict[str, float] = {}
        for mid, b in books.items():
            if b.mid is not None:
                marks[mid] = b.mid
                mids[mid] = b.mid
        self.last_marks = marks
        equity = acct.mark_to(marks)
        was_halted = self.risk.state.halted
        self.risk.update_equity(equity, now)
        if self.risk.state.halted and not was_halted:
            self.journal.log("halt", {"reason": self.risk.state.halt_reason, "equity": round(equity)})
        self.last_equity = equity
        self.last_acct = acct
        model_pd = self._model_probs(now)
        for s in self.feeds.signals():
            if self.fair.add_signal(s):
                self.journal.log("signal", {"market": s.market_id, "delta": s.delta, "reason": s.reason,
                                            "sources": s.sources})
        live = self._live(phase)
        self.fair.anchor_gap = self.feeds.anchor_disagreement()
        fairs = self.fair.compute(now, model_pd, self.feeds.anchors(), mids, live)

        if self.tuner is not None:
            self.tuner.maybe_tune(now, {m: f.p for m, f in fairs.items()}, equity)
        self._tournament_scale(phase, acct, marks, equity)
        gaps = self.feeds.anchor_disagreement()
        moves = self.feeds.anchor_moves()
        damp = self._first_move_damp(now, fairs)
        orders = self.strategy.decide(fairs, books, acct, equity, phase, self.prev_fairs, self.kelly_scale,
                                      venue_gap=gaps, anchor_moves=moves, damp=damp)
        if orders and getattr(self.ex, "bbo_sizes_unknown", False):
            # The bulk snapshot has no sizes: read the real book of each market we want to
            # trade (in parallel), then size against it. Markets not read this step wait.
            want = list(dict.fromkeys(o.market_id for o in orders))[:self.cfg.strategy.depth_reads_per_step]
            got = await asyncio.gather(*(self.ex.get_book_depth(m) for m in want), return_exceptions=True)
            fresh = {m: b for m, b in zip(want, got) if not isinstance(b, Exception)}
            books.update(fresh)
            orders = [o for o in self.strategy.decide(fairs, books, acct, equity, phase, self.prev_fairs,
                                                      self.kelly_scale, venue_gap=gaps, anchor_moves=moves,
                                                      damp=damp) if o.market_id in fresh]
        orders = self._basket_orders(now, phase, fairs, books, acct, marks, equity, orders)
        if self.cfg.strategy.passive_quotes:
            await self.ex.cancel_all()

        for o in orders:
            o.client_id = f"cb-{int(now)}-{next(self._cid)}"
            if o.market_id in self.unknown.values():
                self.stats.orders_blocked += 1
                continue   # an earlier order here is unresolved: don't stack a second one
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
            if getattr(acct, "no_shares", False) and _opening_cost(o, acct) > acct.cash + 1e-6:
                self.stats.orders_blocked += 1
                continue   # the venue would reject it for balance; don't spend a write finding out
            res = await self.ex.place_order(o)
            self.stats.orders_sent += 1
            if res.status == OrderStatus.UNKNOWN:
                # never resend blindly: look it up by client id on later steps
                self.unknown[o.client_id] = o.market_id
                self.journal.log("unknown", {"client_id": o.client_id})
            fills = [(f.side, f.qty, f.price) for f in res.fills]
            if not fills and res.filled_qty > 0:     # venues that report only totals
                fills = [(o.side, res.filled_qty, res.avg_price or o.price)]
            for side, qty, px in fills:
                acct.apply_fill(o.market_id, side, qty, px)
                if self.tuner is not None:
                    self.tuner.on_fill(now, o.market_id, 1 if side == Side.BUY else -1, qty, px)
                self.stats.fills += 1
                self.stats.volume += qty * px
            self.journal.log("order", {"client_id": o.client_id, "market": o.market_id, "side": o.side.value,
                                       "qty": o.qty, "price": o.price, "status": res.status.value,
                                       "filled": res.filled_qty, "avg": res.avg_price, "reason": o.reason,
                                       "fair": fairs.get(o.market_id).p if o.market_id in fairs else None})
        if self.ladder is not None:
            await self._ladder_step(now, fairs, books, acct, marks, equity)
        self.prev_fairs = fairs
        self.last_fairs = fairs

    async def _ladder_step(self, now, fairs, books, acct, marks, equity) -> None:
        """Requote the resting ladder (cupbot/ladder.py) on the markets that need it most,
        within the ladder's write budget and what's left of the orders-per-minute cap."""
        lad = self.ladder
        lad.update_refs(books)
        if self.risk.state.halted:
            return
        chosen, quotes = lad.plan(fairs, books, acct, equity)
        if not chosen:
            return
        recent = sum(1 for t in self.risk._order_times if t > now - 60)
        allowance = self.cfg.risk.max_orders_per_minute - recent - self.cfg.ladder.orders_reserve
        sent: Dict[str, list] = {}
        reqs: List[OrderRequest] = []
        cash_left = acct.cash   # resting orders may reserve balance: never queue more than we have
        for m in chosen:
            if not self.dry_run:
                try:
                    await self.ex.cancel_all(m)
                except Exception as e:
                    self.journal.log("error", {"ladder_cancel": m, "error": repr(e)[:300]})
                    continue
            sent[m] = []
            for o in lad.orders_for({m: quotes[m]}, now, self._cid):
                if allowance <= 0:
                    break
                ok, why = self.risk.check(o, acct, books.get(m), marks, equity, now)
                cost = _opening_cost(o, acct) if getattr(acct, "no_shares", False) else 0.0
                if not ok or cost > cash_left + 1e-6:
                    self.stats.orders_blocked += 1
                    continue
                cash_left -= cost
                allowance -= 1
                reqs.append(o)
                sent[m].append(("buy" if o.side == Side.BUY else "sell", o.price, int(o.qty)))
        for m, qs in sent.items():
            lad.mark_quoted(m, qs, fairs.get(m), acct.pos(m), now)
        summary = {m: [f"{s[0][0]}{s[2]}@{s[1]:.3f}" for s in qs] for m, qs in sent.items()}
        if self.dry_run or not reqs:
            self.journal.log("ladder", {"dry": self.dry_run, "quotes": summary})
            return
        filled = 0
        for i in range(0, len(reqs), 50):
            res = await self.ex.place_batch(reqs[i:i + 50], self.cfg.ladder.expire_s)
            for o, r in zip(reqs[i:i + 50], res):
                self.stats.orders_sent += 1
                if r.filled_qty > 0:   # crossed on arrival (book moved since the snapshot)
                    acct.apply_fill(o.market_id, o.side, r.filled_qty, r.avg_price or o.price)
                    filled += 1
        bad = sum(1 for r in res if r.status.value in ("rejected", "unknown"))
        self.journal.log("ladder", {"quotes": summary, "orders": len(reqs), "filled_on_entry": filled,
                                    "failed_last_batch": bad})

    async def run(self, stop_after_s: Optional[float] = None) -> None:
        """Live loop: step on a timer, or immediately when a feed calls `poke()`."""
        start = time.time()
        if self.wake is None:
            self.wake = asyncio.Event()
        self.stopping = False
        while not self.stopping:
            t0 = time.time()
            try:
                await self.step()
            except Exception as e:  # keep running; the journal records the failure
                self.journal.log("error", {"error": repr(e)[:500]})
                self.last_error = repr(e)[:200]
            if stop_after_s is not None and time.time() - start > stop_after_s:
                return
            timeout = max(0.0, self.cfg.loop_interval_s - (time.time() - t0))
            try:
                await asyncio.wait_for(self.wake.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                pass
            self.wake.clear()

    def stop(self) -> None:
        """Ask run() to finish after the current step (SIGINT / SIGTERM)."""
        self.stopping = True
        self.poke()

    def poke(self) -> None:
        if self.wake is not None:
            self.wake.set()


def _opening_cost(o: OrderRequest, acct: Account) -> float:
    """Cash the venue needs for the part of `o` that opens or grows a position."""
    pos = acct.pos(o.market_id)
    if o.side == Side.BUY:
        return max(0.0, o.qty - max(-pos, 0.0)) * o.price
    return max(0.0, o.qty - max(pos, 0.0)) * (1 - o.price)


def _has_loop() -> bool:
    try:
        asyncio.get_running_loop()
        return True
    except RuntimeError:
        return False
