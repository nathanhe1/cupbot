"""Run the real engine against simulated contests and measure results.

    python -m cupbot sim --runs 40                 # headline scenario
    python -m cupbot sim --runs 40 --sweep         # how results depend on the assumptions
"""

from __future__ import annotations

import asyncio
import copy
import math
import random
import statistics
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

from ..config import BotConfig
from ..engine import Engine
from ..racemodel import ModelParams, RaceModel
from .world import TICK_S, SimParams, SimWorld

RACES = "data/races_2026.json"


@dataclass
class RunResult:
    seed: int
    final: float
    peak: float
    max_drawdown: float
    pre_night_equity: float
    min_equity: float
    orders: int
    blocked: int
    fills: int
    volume: float
    reasons: Dict[str, int] = field(default_factory=dict)
    blocked_why: Dict[str, int] = field(default_factory=dict)
    pnl_by_phase: Dict[str, float] = field(default_factory=dict)
    risk_breaches: int = 0
    field_threshold: float = 0.0      # 3rd-best final balance among simulated competitors
    top3: bool = False


def field_threshold(prices: Dict[str, float], resolved: Dict[str, bool], rng: random.Random,
                    n: int = 2000, start: float = 100_000.0) -> float:
    """Third-best final balance among n simulated competitors (an ASSUMED field):
    50% casual (small random drift), 35% all-in-ish on one market bought at final-week
    prices with a longshot tilt, 15% split across 2-3 longshots. Outcomes are this
    world's, so crowd longshots hit together in the scenarios that favour them."""
    mids = [m for m, p in prices.items() if 0.02 <= p <= 0.98 and m in resolved]
    if not mids:
        return start
    # longshot tilt: weight ~ 1/p for buying YES; the NO side of a favourite is also a longshot
    choices = []
    for m in mids:
        p = prices[m]
        choices.append((m, True, p, 1 / p))
        choices.append((m, False, 1 - p, 1 / (1 - p)))
    tot = sum(c[3] for c in choices)
    cum, acc = [], 0.0
    for c in choices:
        acc += c[3] / tot
        cum.append(acc)

    def pick():
        u = rng.random()
        for c, x in zip(choices, cum):
            if u <= x:
                return c
        return choices[-1]

    finals = []
    for _ in range(n):
        u = rng.random()
        if u < 0.5:
            finals.append(start * math.exp(rng.gauss(0.02, 0.15)))
            continue
        legs = [pick()] if u < 0.85 else [pick() for _ in range(rng.randint(2, 3))]
        f = rng.uniform(0.3, 1.0)
        bal = start * (1 - f)
        per = start * f / len(legs)
        for m, yes_side, price, _ in legs:
            px = min(0.99, price + 0.01)          # they pay the spread
            won = resolved[m] if yes_side else not resolved[m]
            bal += per / px if won else 0.0
        finals.append(bal)
    finals.sort(reverse=True)
    return finals[2]


def run_one(seed: int, sim: Optional[SimParams] = None, cfg: Optional[BotConfig] = None,
            races: str = RACES, strategy_off: bool = False, field_size: int = 2000) -> RunResult:
    sim = copy.deepcopy(sim) if sim else SimParams()
    sim.seed = seed
    cfg = copy.deepcopy(cfg) if cfg else BotConfig()
    world = SimWorld(races, sim)
    cfg.open_ts, cfg.election_ts = 0.0, world.pre_ticks * TICK_S
    model = RaceModel.from_json(races, ModelParams(n_draws=1000, seed=seed + 1000))
    eng = Engine(cfg, world.exchange, world, model, world.markets)
    eng.fair.consistency = RaceModel.from_json(races, ModelParams(n_draws=800, seed=seed + 2000))
    world.start()
    equity_path: List[float] = []
    phase_start: Dict[str, float] = {}
    breaches = 0
    fw_prices: Dict[str, float] = {}

    async def go():
        nonlocal breaches
        while world.phase() != "closed":
            ph = world.phase()
            before = dict(world.exchange.account.positions)
            if not strategy_off:
                await eng.step()
            after = dict(world.exchange.account.positions)
            world.note_bot_flow(before, after)
            truth = world.truth(0)
            eq = world.exchange.account.mark_to(truth)
            equity_path.append(eq)
            if ph not in phase_start and ph == "final_week":
                fw_prices.update({m: b.mid for m, b in (await world.exchange.get_books(list(truth))).items()
                                  if b.mid is not None})
            phase_start.setdefault(ph, eq)
            breaches += _check_invariants(world, cfg)
            world.advance()

    asyncio.run(go())
    pre_night = phase_start.get("night", equity_path[-1] if equity_path else 100_000.0)
    resolved = world.resolve_all()
    final = world.exchange.account.cash
    thr = field_threshold(fw_prices, resolved, random.Random(seed * 7919), n=field_size)
    peak, mdd, run_peak = 0.0, 0.0, 0.0
    for e in equity_path + [final]:
        run_peak = max(run_peak, e)
        mdd = max(mdd, (run_peak - e) / run_peak if run_peak > 0 else 0)
        peak = max(peak, e)
    starts = [(k, v) for k, v in phase_start.items()]
    pnl = {}
    order = ["pre", "final_week", "night"]
    for i, ph in enumerate(order):
        if ph not in phase_start:
            continue
        nxt = next((phase_start[o] for o in order[i + 1:] if o in phase_start), final)
        pnl[ph] = nxt - phase_start[ph]
    return RunResult(seed, final, peak, mdd, pre_night, min(equity_path + [final]),
                     eng.stats.orders_sent, eng.stats.orders_blocked, eng.stats.fills, eng.stats.volume,
                     dict(eng.stats.reasons), dict(eng.risk.blocked), pnl, breaches, thr, final >= thr)


def _check_invariants(world: SimWorld, cfg: BotConfig) -> int:
    """Count violations of hard limits that the risk manager is supposed to enforce."""
    acct = world.exchange.account
    eq = acct.mark_to(world.truth(0))
    bad = 0
    if acct.cash < -1e-6:
        bad += 1
    for mid, q in acct.positions.items():
        p = world.truth(0)[mid]
        val = abs(q) * (p if q > 0 else 1 - p)
        # allow slack: prices move after the order was sized
        if eq > 0 and val > 1.5 * cfg.risk.max_market_notional_frac * eq + 1000:
            bad += 1
    return bad


def _run_star(args):
    return run_one(*args)


def run_many(n: int, sim: Optional[SimParams] = None, cfg: Optional[BotConfig] = None,
             seed0: int = 1, workers: int = 8, strategy_off: bool = False) -> List[RunResult]:
    jobs = [(seed0 + i, sim, cfg, RACES, strategy_off) for i in range(n)]
    if workers <= 1:
        return [run_one(*j) for j in jobs]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_run_star, jobs))


def summarize(results: List[RunResult], start: float = 100_000.0) -> Dict[str, float]:
    finals = [r.final for r in results]
    mult = [f / start for f in finals]
    logs = [math.log(max(m, 1e-6)) for m in mult]
    s = {
        "runs": len(results),
        "median_x": statistics.median(mult),
        "mean_x": statistics.mean(mult),
        "geo_mean_x": math.exp(statistics.mean(logs)),
        "p_loss": sum(1 for m in mult if m < 1) / len(mult),
        "p_2x": sum(1 for m in mult if m >= 2) / len(mult),
        "p_5x": sum(1 for m in mult if m >= 5) / len(mult),
        "p_10x": sum(1 for m in mult if m >= 10) / len(mult),
        "worst_x": min(mult),
        "best_x": max(mult),
        "median_max_dd": statistics.median(r.max_drawdown for r in results),
        "median_pre_night_x": statistics.median(r.pre_night_equity / start for r in results),
        "orders_per_run": statistics.mean(r.orders for r in results),
        "blocked_per_run": statistics.mean(r.blocked for r in results),
        "risk_breaches": sum(r.risk_breaches for r in results),
        "median_field_threshold_x": statistics.median(r.field_threshold / start for r in results),
        "p_top3": sum(1 for r in results if r.top3) / len(results),
    }
    for ph in ("pre", "final_week", "night"):
        vals = [r.pnl_by_phase.get(ph, 0.0) for r in results]
        s[f"mean_pnl_{ph}"] = statistics.mean(vals)
    return s


def fmt_summary(name: str, s: Dict[str, float]) -> str:
    return (f"{name:<34} median {s['median_x']:.2f}x  geo {s['geo_mean_x']:.2f}x  "
            f"P(loss) {s['p_loss']:.0%}  P(2x) {s['p_2x']:.0%}  P(5x) {s['p_5x']:.0%}  "
            f"P(10x) {s['p_10x']:.0%}  worst {s['worst_x']:.2f}x  best {s['best_x']:.2f}x  "
            f"medDD {s['median_max_dd']:.0%}  pnl pre/fw/night "
            f"{s['mean_pnl_pre']/1000:+.0f}k/{s['mean_pnl_final_week']/1000:+.0f}k/{s['mean_pnl_night']/1000:+.0f}k  "
            f"breaches {s['risk_breaches']}  top-3 bar {s['median_field_threshold_x']:.1f}x  "
            f"P(top3) {s['p_top3']:.0%}")


SWEEPS = {
    "headline (default assumptions)": {},
    "sharper MM (half the bias/open error)": {"mm_bias_sd": 0.10, "mm_open_err_sd": 0.22},
    "MM with no lag": {"mm_lag_pre": 0, "mm_lag_night": 0},
    "sharp MM, no lag (hard case)": {"mm_bias_sd": 0.08, "mm_open_err_sd": 0.15, "mm_lag_pre": 0, "mm_lag_night": 0},
    "noisier anchors (2x bias)": {"anchor_bias_sd": 0.20},
    "fewer anchors (40% coverage)": {"anchor_coverage": 0.4},
    "no student flow": {"stud_rate": 0.0},
    "bigger mirage, worse mirage guess": {"mirage_scale": 1.5, "bot_mirage_err_sd": 3.0},
    "thin MM (500/level)": {"mm_size": 500.0},
    "deep MM (4x size, 1/4 price impact)": {"mm_size": 6000.0, "mm_inv_k": 1e-5},
}


def sweep(n: int, cfg: Optional[BotConfig] = None, workers: int = 8) -> List[str]:
    lines = []
    for name, over in SWEEPS.items():
        sp = SimParams(**over)
        res = run_many(n, sp, cfg, workers=workers)
        lines.append(fmt_summary(name, summarize(res)))
    return lines
