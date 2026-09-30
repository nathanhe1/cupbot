"""A simulated Predictions Cup, from Oct 1 open to resolution.

What is simulated (every number here is an ASSUMPTION, set in SimParams):

  Truth      Each race's final margin = prior mean + pre-election drift (polls,
             news jumps) + election-night reveal (national, regional, race parts)
             + a small residual nobody knows until the count ends. The 'true'
             probability at any moment is the probability given everything
             revealed so far.
  Sponsor MM Quotes every market around its own estimate: the truth, seen with a
             lag, plus a slowly varying bias, a big seeded error at open that
             decays, and an inventory skew (so trading against it moves its price).
             It prices chamber/combos from its own noisy view, not from races.
  Students   Random market orders with a longshot tilt, plus resting limit orders
             at noisy prices that go stale when the truth moves.
  Anchors    'Kalshi/Polymarket' prices: truth plus a smaller persistent bias,
             no lag, only for a fraction of markets.
  Bot feeds  Poll averages (truth-ish means with noise), news signals for some
             jumps (with noise), and election-night results with each state's
             counting-order 'mirage'.

The simulator can only reward edges we built into it. Its job is to check
mechanics, risk limits and sizing, and to show how results depend on the
assumptions (see backtest --sweep), not to prove the bot makes money.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ..election_night import NightParams, RaceReport
from ..exchange.mock import MockExchange
from ..feeds import Feeds
from ..mathutil import clip_p, logit, norm_cdf, sigmoid, student_t
from ..types import Market, OrderKind, OrderRequest, Side, Signal, TimeInForce

TICK_S = 6 * 3600          # pre-election tick = 6 hours
NIGHT_TICK_S = 1800        # election-night tick = 30 minutes

# early-count lean by state (D minus R points in early counts vs final); + means early counts
# lean D (early/mail votes counted first), - means lean R (mail counted late). Illustrative.
DEFAULT_MIRAGE = {"PA": -5, "WI": -5, "MI": -4, "GA": 3, "NC": 3, "FL": 2, "OH": 1, "AZ": -2, "NV": -2,
                  "TX": 1, "VA": 2, "NH": 0, "ME": -1, "MN": -2, "IA": 1}

CLOSE_TICK = {"northeast": 0, "south": 1, "midwest": 2, "west": 4, "national": 0}


@dataclass
class SimParams:
    seed: int = 1
    days: int = 34
    ticks_per_day: int = 4
    night_ticks: int = 36
    # variance budget of the national error (points); race sd comes from races file
    national_sd: float = 3.0
    region_sd: float = 1.5
    frac_pre: float = 0.5        # share of each error revealed before election night
    frac_resid: float = 0.07     # share never revealed before trading closes
    jump_prob: float = 0.015     # per race per tick: a news jump
    jump_sd: float = 2.5
    final_week_days: int = 7
    # sponsor market maker
    mm_lag_pre: int = 1          # ticks
    mm_lag_night: int = 1
    mm_bias_sd: float = 0.20     # logit, stationary sd of slowly varying bias
    mm_bias_phi: float = 0.97
    mm_open_err_sd: float = 0.45  # logit error at open ...
    mm_open_halflife_days: float = 2.0  # ... decaying with this half-life
    mm_half_spread: float = 0.02
    mm_size: float = 1500.0      # contracts per level
    mm_levels: int = 3
    mm_inv_k: float = 4e-5       # logit shift per contract of MM inventory
    mm_inv_decay: float = 0.9    # MM lays off inventory over time (per tick)
    mm_chamber_extra_bias: float = 0.25
    # students
    stud_rate: float = 1.5       # orders per market per tick
    stud_limit_frac: float = 0.5
    stud_size: float = 150.0
    stud_belief_sd: float = 0.45  # logit noise of student limit prices
    stud_order_life: int = 24    # ticks before a resting student order is pulled
    stud_longshot_tilt: float = 0.15
    # anchors
    anchor_coverage: float = 0.75
    anchor_bias_sd: float = 0.10
    anchor_bias_phi: float = 0.97
    # bot feeds
    poll_noise_sd: float = 1.5
    news_detect_prob: float = 0.7
    news_noise: float = 0.3
    mirage_scale: float = 1.0
    bot_mirage_err_sd: float = 1.5
    truth_samples: int = 300
    # election-night call desks
    call_desks: int = 2
    call_p: float = 0.995        # a desk calls a race once the truth passes this
    call_delay_max: int = 2      # ticks
    wrong_call_prob: float = 0.01  # per desk per race: calls the wrong side (tests the confirmation rule)


@dataclass
class RaceTruth:
    race_id: str
    state: str
    region: str
    mean0: float
    sd: float
    pre_path: List[float] = field(default_factory=list)   # cumulative pre-election drift by tick
    jumps: Dict[int, float] = field(default_factory=dict)
    night_race: float = 0.0
    resid: float = 0.0
    close_tick: int = 0
    report_ticks: int = 12


class SimWorld(Feeds):
    def __init__(self, races_json: str, params: Optional[SimParams] = None):
        self.p = params or SimParams()
        self.rng = random.Random(self.p.seed)
        with open(races_json) as f:
            data = json.load(f)
        self.d_holdover = data["senate"]["d_holdover"]
        self.d_needed = data["senate"]["d_seats_needed"]
        self.pre_ticks = self.p.days * self.p.ticks_per_day
        self.total_ticks = self.pre_ticks + self.p.night_ticks
        self.t = 0
        self._build_truth(data)
        self.markets = self._build_markets()
        self.exchange = MockExchange(self.markets, starting_cash=100_000.0)
        self._init_agents()
        self._pending_signals: List[Signal] = []
        self.resolved: Dict[str, bool] = {}
        self._truth_cache: Dict[str, float] = {}
        self._refresh_truth()

    # ======================================================== truth
    def _build_truth(self, data) -> None:
        p, rng = self.p, self.rng
        fp, fr = p.frac_pre, p.frac_resid
        fn = 1 - fp - fr
        self.races: Dict[str, RaceTruth] = {}
        entries = list(data["races"])
        if "house" in data:
            h = data["house"]
            entries.append({"race_id": h.get("race_id", "HOUSE"), "state": "US", "region": "national",
                            "mean": h["mean"], "sd": h["sd"]})
        regions = sorted({e["region"] for e in entries} - {"national"})
        n_pre = self.pre_ticks
        # national and regional parts: pre drift paths + night reveals
        self.nat_pre = self._path(p.national_sd * math.sqrt(fp), n_pre)
        self.nat_night = p.national_sd * math.sqrt(fn) * student_t(rng, 6)
        self.nat_resid = p.national_sd * math.sqrt(fr) * rng.gauss(0, 1)
        self.reg_pre = {r: self._path(p.region_sd * math.sqrt(fp), n_pre) for r in regions}
        self.reg_night = {r: p.region_sd * math.sqrt(fn) * rng.gauss(0, 1) for r in regions}
        self.reg_night["national"] = 0.0
        for e in entries:
            rt = RaceTruth(e["race_id"], e["state"], e["region"], float(e["mean"]), float(e["sd"]))
            # pre drift: diffusion plus rare news jumps (jumps use part of the budget)
            rt.pre_path = self._path(rt.sd * math.sqrt(fp * 0.7), n_pre)
            cum = 0.0
            jumps_cum = []
            for t in range(n_pre):
                if rng.random() < p.jump_prob:
                    j = rng.gauss(0, p.jump_sd)
                    rt.jumps[t] = j
                    cum += j
                jumps_cum.append(cum)
            rt.pre_path = [a + b for a, b in zip(rt.pre_path, jumps_cum)]
            rt.night_race = rt.sd * math.sqrt(fn) * student_t(rng, 6)
            rt.resid = rt.sd * math.sqrt(fr) * rng.gauss(0, 1)
            rt.close_tick = CLOSE_TICK.get(rt.region, 1) + rng.randint(0, 1)
            rt.report_ticks = rng.randint(6, 20) if rt.state not in ("AK", "ME") else 60
            self.races[rt.race_id] = rt
        self.mirage = {s: v * p.mirage_scale for s, v in DEFAULT_MIRAGE.items()}
        # pre-drawn normal shocks for the truth chamber probability (common random numbers)
        n = p.truth_samples
        self._z_nat = [rng.gauss(0, 1) for _ in range(n)]
        self._z_reg = {r: [rng.gauss(0, 1) for _ in range(n)] for r in regions}
        self._z_race = {rid: [rng.gauss(0, 1) for _ in range(n)] for rid in self.races}
        self.regions = regions

    def _path(self, total_sd: float, n: int) -> List[float]:
        step = total_sd / math.sqrt(max(n, 1))
        x, out = 0.0, []
        for _ in range(n):
            x += self.rng.gauss(0, step)
            out.append(x)
        return out

    def final_margin(self, rid: str) -> float:
        rt = self.races[rid]
        reg = rt.region
        nat = self.nat_pre[-1] + self.nat_night + self.nat_resid
        rg = 0.0 if reg == "national" else self.reg_pre[reg][-1] + self.reg_night[reg]
        return rt.mean0 + nat + rg + rt.pre_path[-1] + rt.night_race + rt.resid

    def _night_t(self) -> int:
        return self.t - self.pre_ticks  # < 0 before election night

    def rho(self, rid: str) -> float:
        nt = self._night_t()
        rt = self.races[rid]
        if rt.region == "national":
            vals = [self.rho(r) for r in self.races if self.races[r].region != "national"]
            return sum(vals) / len(vals)
        if nt < rt.close_tick:
            return 0.0
        return min(1.0, (nt - rt.close_tick + 1) / rt.report_ticks)

    def _known_and_var(self, rid: str, t: Optional[int] = None) -> Tuple[float, float]:
        """(known expected margin, remaining variance) at tick t (default now)."""
        p = self.p
        saved = self.t
        if t is not None:
            self.t = t
        try:
            rt = self.races[rid]
            ti = min(self.t, self.pre_ticks - 1)
            fp, fr = p.frac_pre, p.frac_resid
            fn = 1 - fp - fr
            frac_left_pre = 1 - (ti + 1) / self.pre_ticks
            known = rt.mean0 + self.nat_pre[ti] + rt.pre_path[ti]
            var = (p.national_sd ** 2 + rt.sd ** 2) * fp * frac_left_pre
            if rt.region != "national":
                known += self.reg_pre[rt.region][ti]
                var += p.region_sd ** 2 * fp * frac_left_pre
            if self._night_t() >= 0:
                rbar = sum(self.rho(r) for r in self.races if self.races[r].region != "national") / \
                    max(1, len([r for r in self.races if self.races[r].region != "national"]))
                a_n = min(1.0, 3 * rbar)
                my = self.rho(rid)
                known += a_n * self.nat_night + my * rt.night_race
                var += (1 - a_n) * p.national_sd ** 2 * fn + (1 - my) * rt.sd ** 2 * fn
                if rt.region != "national":
                    regs = [self.rho(r) for r in self.races if self.races[r].region == rt.region]
                    a_r = min(1.0, 2 * sum(regs) / len(regs))
                    known += a_r * self.reg_night[rt.region]
                    var += (1 - a_r) * p.region_sd ** 2 * fn
            else:
                var += (p.national_sd ** 2 + rt.sd ** 2) * fn
                if rt.region != "national":
                    var += p.region_sd ** 2 * fn
            var += (p.national_sd ** 2 + rt.sd ** 2 + (0 if rt.region == "national" else p.region_sd ** 2)) * fr
            return known, var
        finally:
            self.t = saved

    def _refresh_truth(self) -> None:
        """True P(YES) for every market at the current tick."""
        pd: Dict[str, float] = {}
        kv = {rid: self._known_and_var(rid) for rid in self.races}
        for rid, (k, v) in kv.items():
            pd[rid] = norm_cdf(k / math.sqrt(v))
        # chamber truth via conditional simulation of the remaining uncertainty
        # (national / regional / race parts approximated as independent normals)
        n = self.p.truth_samples
        senate = [r for r in self.races if r != "HOUSE"]
        need = self.d_needed - self.d_holdover
        p = self.p
        nat_share = p.national_sd ** 2
        pre = []
        for rid in senate:
            k, v = kv[rid]
            rt = self.races[rid]
            tot = nat_share + p.region_sd ** 2 + rt.sd ** 2
            wn = math.sqrt(nat_share / tot)
            wr = math.sqrt(p.region_sd ** 2 / tot)
            wi = math.sqrt(max(0.0, 1 - wn ** 2 - wr ** 2))
            sd = math.sqrt(v)
            pre.append((k, sd * wn, sd * wr, sd * wi, self._z_reg[rt.region], self._z_race[rid]))
        hk, hv = kv["HOUSE"]
        hs = math.sqrt(hv)
        hw = math.sqrt(nat_share / (nat_share + self.races["HOUSE"].sd ** 2))
        hw2 = math.sqrt(1 - hw ** 2)
        zh = self._z_race["HOUSE"]
        sen_d = house_d = dd = dr = rd = rr = 0
        for s in range(n):
            zn = self._z_nat[s]
            seats = 0
            for k, a_n, a_r, a_i, zr, zi in pre:
                if k + a_n * zn + a_r * zr[s] + a_i * zi[s] > 0:
                    seats += 1
            sen = seats >= need
            house = hk + hs * (hw * zn + hw2 * zh[s]) > 0
            sen_d += sen
            house_d += house
            dd += house and sen
            dr += house and not sen
            rd += (not house) and sen
            rr += (not house) and (not sen)
        pd["SENATE_D"] = sen_d / n
        pd["HOUSE_D"] = house_d / n
        pd["COMBO_DD"], pd["COMBO_DR"], pd["COMBO_RD"], pd["COMBO_RR"] = dd / n, dr / n, rd / n, rr / n
        out = {}
        for m in self.markets:
            key = m.race_id
            out[m.market_id] = pd[key] if m.d_sign >= 0 else 1 - pd[key]
        self._truth_cache = out
        self._truth_hist.append(dict(out)) if hasattr(self, "_truth_hist") else setattr(self, "_truth_hist", [dict(out)])

    def truth(self, lag: int = 0) -> Dict[str, float]:
        h = self._truth_hist
        return h[max(0, len(h) - 1 - lag)]

    # ======================================================== markets
    def _build_markets(self) -> List[Market]:
        ms: List[Market] = []
        for rid, rt in self.races.items():
            if rid == "HOUSE":
                continue
            # R-held seats are listed as 'Will the Republican win?' to exercise d_sign handling
            holder_r = rt.mean0 < 0
            ms.append(Market(f"M-{rid}", f"{'Republican' if holder_r else 'Democrat'} wins {rid}", "race",
                             rid, -1 if holder_r else +1))
        ms.append(Market("M-SENATE", "Democrats control the Senate", "chamber", "SENATE_D", +1))
        ms.append(Market("M-HOUSE", "Democrats control the House", "chamber", "HOUSE_D", +1))
        for k, t in (("DD", "D House, D Senate"), ("DR", "D House, R Senate"),
                     ("RD", "R House, D Senate"), ("RR", "R House, R Senate")):
            ms.append(Market(f"M-BOP-{k}", t, "combo", f"COMBO_{k}", +1, group="BOP"))
        return ms

    # ======================================================== agents
    def _init_agents(self) -> None:
        rng = self.rng
        p = self.p
        self.mm_bias = {m.market_id: rng.gauss(0, p.mm_bias_sd) for m in self.markets}
        self.mm_open_err = {m.market_id: rng.gauss(0, p.mm_open_err_sd) *
                            (1 + (p.mm_chamber_extra_bias if m.kind != "race" else 0)) for m in self.markets}
        self.mm_inv = {m.market_id: 0.0 for m in self.markets}
        self.has_anchor = {m.market_id: (rng.random() < p.anchor_coverage) for m in self.markets}
        for m in self.markets:
            if m.kind == "combo" or m.kind == "chamber":
                self.has_anchor[m.market_id] = True
        self.anchor_bias = {m.market_id: rng.gauss(0, p.anchor_bias_sd) for m in self.markets}
        self.poll_err = {rid: rng.gauss(0, p.poll_noise_sd) for rid in self.races}
        self.bot_mirage = {s: v + rng.gauss(0, p.bot_mirage_err_sd) for s, v in self.mirage.items()}
        self._stud_owner = "students"

    def _mm_estimate(self, mid: str) -> float:
        p = self.p
        lag = p.mm_lag_night if self._night_t() >= 0 else p.mm_lag_pre
        base = self.truth(lag)[mid]
        days = self.t / p.ticks_per_day
        open_err = self.mm_open_err[mid] * 0.5 ** (days / p.mm_open_halflife_days)
        x = logit(base) + self.mm_bias[mid] + open_err - p.mm_inv_k * self.mm_inv[mid]
        return sigmoid(x)

    def _mm_quote(self) -> None:
        p = self.p
        ex = self.exchange
        ex.cancel_owner("mm")
        for m in self.markets:
            if m.closed:
                continue
            mid = m.market_id
            est = self._mm_estimate(mid)
            # tighter spread and smaller size in the tails
            for lvl in range(p.mm_levels):
                off = p.mm_half_spread + 0.01 * lvl
                b = math.floor((est - off) * 100) / 100
                a = math.ceil((est + off) * 100) / 100
                if 0.01 <= b <= 0.99:
                    ex.submit("mm", OrderRequest(mid, Side.BUY, p.mm_size, b, tif=TimeInForce.GTC, client_id="mm"))
                if 0.01 <= a <= 0.99:
                    ex.submit("mm", OrderRequest(mid, Side.SELL, p.mm_size, a, tif=TimeInForce.GTC, client_id="mm"))

    def _students(self) -> None:
        p, rng, ex = self.p, self.rng, self.exchange
        ex.cancel_owner(self._stud_owner, older_than=self.t - p.stud_order_life)
        for m in self.markets:
            if m.closed:
                continue
            mid = m.market_id
            book = ex.books[mid]
            bb, ba = book.best_bid(), book.best_ask()
            if bb is None or ba is None:
                continue
            mkt_mid = (bb + ba) / 200
            k = self._poisson(p.stud_rate)
            for _ in range(k):
                size = max(10.0, rng.expovariate(1 / p.stud_size))
                if rng.random() < p.stud_limit_frac:
                    price = sigmoid(logit(mkt_mid) + rng.gauss(0, p.stud_belief_sd))
                    side = Side.BUY if rng.random() < 0.5 else Side.SELL
                    px = round(price * 100) / 100
                    if 0.01 <= px <= 0.99:
                        ex.submit(self._stud_owner, OrderRequest(mid, side, size, px, tif=TimeInForce.GTC))
                else:
                    tilt = p.stud_longshot_tilt if mkt_mid < 0.2 else (-p.stud_longshot_tilt if mkt_mid > 0.8 else 0)
                    side = Side.BUY if rng.random() < 0.5 + tilt else Side.SELL
                    ex.submit(self._stud_owner, OrderRequest(mid, side, size, None, kind=OrderKind.MARKET,
                                                             tif=TimeInForce.IOC))

    def _poisson(self, lam: float) -> int:
        l, k, prod = math.exp(-lam), 0, self.rng.random()
        while prod > l:
            k += 1
            prod *= self.rng.random()
        return k

    def _update_mm_state(self) -> None:
        """Track MM inventory from counterparty volume changes (approximation: the bot's
        net flow in each market is the MM's inventory change, since students net out)."""
        p = self.p
        for mid in self.mm_inv:
            self.mm_inv[mid] *= p.mm_inv_decay
            self.mm_bias[mid] = p.mm_bias_phi * self.mm_bias[mid] + \
                math.sqrt(1 - p.mm_bias_phi ** 2) * self.rng.gauss(0, p.mm_bias_sd)
            self.anchor_bias[mid] = p.anchor_bias_phi * self.anchor_bias[mid] + \
                math.sqrt(1 - p.anchor_bias_phi ** 2) * self.rng.gauss(0, p.anchor_bias_sd)

    # ======================================================== feeds (what the bot sees)
    def now(self) -> float:
        if self.t <= self.pre_ticks:
            return self.t * TICK_S
        return self.pre_ticks * TICK_S + (self.t - self.pre_ticks) * NIGHT_TICK_S

    def phase(self) -> str:
        if self.t >= self.total_ticks:
            return "closed"
        if self.t >= self.pre_ticks:
            return "night"
        if self.t >= self.pre_ticks - self.p.final_week_days * self.p.ticks_per_day:
            return "final_week"
        return "pre"

    def anchors(self) -> Dict[str, float]:
        tr = self.truth(0)
        out = {}
        for m in self.markets:
            if m.closed or not self.has_anchor[m.market_id]:
                continue
            out[m.market_id] = sigmoid(logit(tr[m.market_id]) + self.anchor_bias[m.market_id])
        return out

    def poll_means(self) -> Dict[str, float]:
        ti = min(self.t, self.pre_ticks - 1)
        out = {}
        for rid, rt in self.races.items():
            known = rt.mean0 + self.nat_pre[ti] + rt.pre_path[ti]
            if rt.region != "national":
                known += self.reg_pre[rt.region][ti]
            out[rid] = known + self.poll_err[rid]
        return out

    def signals(self) -> List[Signal]:
        out, self._pending_signals = self._pending_signals, []
        return out

    def _emit_news(self) -> None:
        """News jumps that happened this tick -> noisy capped-later signals for the bot."""
        if self.t >= self.pre_ticks or self.t == 0:
            return
        before = self.truth(1)
        after = self.truth(0)
        for rid, rt in self.races.items():
            if self.t in rt.jumps and self.rng.random() < self.p.news_detect_prob:
                for m in self.markets:
                    if m.race_id == rid:
                        true_dp = after[m.market_id] - before[m.market_id]
                        est = true_dp * (1 + self.rng.gauss(0, self.p.news_noise))
                        self._pending_signals.append(Signal(m.market_id, est, "medium", "sim news",
                                                            ["sim://news"], ts=self.now(), ttl_s=2 * TICK_S))

    def live_reports(self):
        if self._night_t() < 0:
            return None
        reports = {}
        for rid, rt in self.races.items():
            r = self.rho(rid)
            if r <= 0:
                continue
            k, _ = self._known_and_var(rid)
            counted = k + self.mirage.get(rt.state, 0.0) * (1 - r) + self.rng.gauss(0, 1.0) * (1 - r)
            reports[rid] = RaceReport(rt.state, r, counted)
        return reports

    def call_book(self):
        if self._night_t() < 0 or not self.p.call_desks:
            return None
        from ..calls import Call, CallBook
        if not hasattr(self, "_calls"):
            self._calls = CallBook()
            self._call_plan = {}
        tr = self.truth(0)
        for rid in self.races:
            if rid == "HOUSE":
                continue
            m = next(x for x in self.markets if x.race_id == rid)
            p_d = tr[m.market_id] if m.d_sign >= 0 else 1 - tr[m.market_id]
            for desk in range(self.p.call_desks):
                key = (rid, desk)
                if key in self._call_plan:
                    continue
                wrong = self.rng.random() < self.p.wrong_call_prob / max(1, self.total_ticks - self.pre_ticks)
                if p_d >= self.p.call_p or p_d <= 1 - self.p.call_p or wrong:
                    party = "D" if p_d >= 0.5 else "R"
                    if wrong:
                        party = "R" if party == "D" else "D"
                    self._call_plan[key] = (self.t + self.rng.randint(0, self.p.call_delay_max), party)
            for desk in range(self.p.call_desks):
                plan = self._call_plan.get((rid, desk))
                if plan and plan[0] <= self.t:
                    self._calls.add(Call(rid, plan[1], f"desk{desk}", self.now(), "sim"))
        return self._calls

    def night_params(self):
        fn = 1 - self.p.frac_pre - self.p.frac_resid
        return NightParams(national_night_sd=self.p.national_sd * math.sqrt(fn),
                           region_night_sd=self.p.region_sd * math.sqrt(fn),
                           race_night_sd=4.0 * math.sqrt(fn),
                           resid_sd=4.0 * math.sqrt(self.p.frac_resid),
                           mirage=dict(self.bot_mirage), mirage_err_sd=self.p.bot_mirage_err_sd)

    # ======================================================== clock
    def advance(self) -> None:
        """Move the world forward one tick: truth, anchors, MM, students, news."""
        self.t += 1
        self.exchange.tick_index = self.t
        self.exchange.now = self.now()
        if self.t >= self.total_ticks:
            return
        self._refresh_truth()
        self._update_mm_state()
        self._emit_news()
        self._mm_quote()
        self._students()

    def start(self) -> None:
        self.exchange.now = self.now()
        self._mm_quote()
        self._students()

    def note_bot_flow(self, before: Dict[str, float], after: Dict[str, float]) -> None:
        for mid in set(before) | set(after):
            d = after.get(mid, 0.0) - before.get(mid, 0.0)
            if d:
                self.mm_inv[mid] -= d  # MM took the other side

    def resolve_all(self) -> Dict[str, bool]:
        margins = {rid: self.final_margin(rid) for rid in self.races}
        seats = sum(1 for r in self.races if r != "HOUSE" and margins[r] > 0)
        sen = seats >= self.d_needed - self.d_holdover
        house = margins["HOUSE"] > 0
        outcome_d = dict((rid, margins[rid] > 0) for rid in self.races)
        outcome_d.update({"SENATE_D": sen, "HOUSE_D": house, "COMBO_DD": house and sen,
                          "COMBO_DR": house and not sen, "COMBO_RD": (not house) and sen,
                          "COMBO_RR": (not house) and (not sen)})
        for m in self.markets:
            d_yes = outcome_d[m.race_id]
            yes = d_yes if m.d_sign >= 0 else not d_yes
            self.resolved[m.market_id] = yes
            self.exchange.resolve(m.market_id, yes)
        self.seats = self.d_holdover + seats
        return self.resolved
