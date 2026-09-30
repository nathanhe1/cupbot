"""Command line entry point:  python -m cupbot <command>

  model        print the race model's probabilities from data/races_2026.json
  sim          backtest the engine on simulated contests (--runs, --sweep)
  paper        paper-trade one simulated contest tick by tick, printing activity
  whoami       check the Super.Market API key works (read-only)
  live         run against the real platform; DRY RUN unless --live (and config allows it)
  measure-mm   record Cup quotes next to Kalshi/Polymarket to measure the MM's repricing lag
  mm-report    summarise a measure-mm CSV
  kill/unkill  create/remove the KILL file (the engine cancels everything and stops trading)
  cancel-all   cancel every resting order on the platform now
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time

from .config import load_config, load_dotenv


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="cupbot")
    ap.add_argument("--config", default="config/default.json")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("model")
    s = sub.add_parser("sim")
    s.add_argument("--runs", type=int, default=40)
    s.add_argument("--seed", type=int, default=1)
    s.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    s.add_argument("--sweep", action="store_true", help="re-run under harder / different assumptions")
    s.add_argument("--baseline", action="store_true", help="also run a do-nothing baseline")
    p = sub.add_parser("paper")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--speed", type=float, default=0.0, help="seconds to sleep per simulated tick")
    p.add_argument("--quiet", action="store_true")
    sub.add_parser("whoami")
    lv = sub.add_parser("live")
    lv.add_argument("--live", action="store_true", help="actually send orders (config must also allow it)")
    lv.add_argument("--seconds", type=float, default=None)
    mm = sub.add_parser("measure-mm")
    mm.add_argument("--minutes", type=float, default=60)
    mm.add_argument("--out", default="journal/mm_lag.csv")
    mr = sub.add_parser("mm-report")
    mr.add_argument("csv", nargs="?", default="journal/mm_lag.csv")
    rp = sub.add_parser("refresh-priors")
    rp.add_argument("--write", action="store_true", help="write means and anchor tickers to data/")
    dm = sub.add_parser("draft-map")
    dm.add_argument("--out", default="config/market_map.json")
    sub.add_parser("preflight")
    rc = sub.add_parser("record")
    rc.add_argument("--interval", type=float, default=2.0)
    rc.add_argument("--minutes", type=float, default=None)
    tr = sub.add_parser("tick-report")
    tr.add_argument("--dir", default="journal/ticks")
    sub.add_parser("kill")
    sub.add_parser("unkill")
    sub.add_parser("cancel-all")
    a = ap.parse_args(argv)

    load_dotenv()
    cfg = load_config(a.config)

    if a.cmd == "model":
        from .racemodel import ModelParams, RaceModel
        m = RaceModel.from_json(cfg.races_file, ModelParams(n_draws=20000))
        s_ = m.summary()
        for k in sorted(s_):
            print(f"{k:<20} {s_[k]:.3f}")
        return 0

    if a.cmd == "sim":
        from .sim.backtest import fmt_summary, run_many, summarize, sweep
        t0 = time.time()
        if a.sweep:
            for line in sweep(a.runs, cfg, a.workers):
                print(line, flush=True)
        else:
            res = run_many(a.runs, cfg=cfg, seed0=a.seed, workers=a.workers)
            print(fmt_summary("bot", summarize(res)))
            if a.baseline:
                base = run_many(a.runs, cfg=cfg, seed0=a.seed, workers=a.workers, strategy_off=True)
                print(fmt_summary("do nothing", summarize(base)))
        print(f"({time.time() - t0:.0f}s)")
        return 0

    if a.cmd == "paper":
        return asyncio.run(_paper(cfg, a.seed, a.speed, a.quiet))

    if a.cmd == "whoami":
        from .exchange.supermarket import SuperMarketExchange

        async def go():
            ex = SuperMarketExchange(cfg.venue)
            try:
                print(await ex.whoami())
            finally:
                await ex.close()
        asyncio.run(go())
        return 0

    if a.cmd == "live":
        from .live import run_live
        if a.live and not cfg.venue.live_trading_enabled:
            print("refusing: --live given but venue.live_trading_enabled is false in the config")
            return 2
        asyncio.run(run_live(cfg, live_orders=a.live, run_seconds=a.seconds))
        return 0

    if a.cmd == "measure-mm":
        from .live import measure_mm
        asyncio.run(measure_mm(cfg, a.minutes, a.out))
        return 0

    if a.cmd == "mm-report":
        from .live import mm_lag_report
        print(mm_lag_report(a.csv))
        return 0

    if a.cmd == "refresh-priors":
        from .discover import STATE_NAMES, gather, implied_means, write_anchor_index, write_priors
        from .engine import Engine
        found = gather(sorted(STATE_NAMES))
        elapsed = min(1.0, max(0.0, (time.time() - cfg.open_ts) / (cfg.election_ts - cfg.open_ts)))
        scale = (1 - cfg.fair.pre_info_frac * elapsed) ** 0.5
        means = implied_means(found, cfg.races_file, scale)
        print(f"{'race':<10} {'Kalshi P(D)':>11} {'Poly P(D)':>10} {'blend':>7} {'mean':>7}")
        for rid, f in found.items():
            k, p = f.get("kalshi") or {}, f.get("polymarket") or {}
            kd = (1 - k["p_r"]) if k.get("p_r") is not None else k.get("p_d")
            pdp = (1 - p["p_r"]) if p.get("p_r") is not None else p.get("p_d")
            fmt = lambda x: f"{x:.3f}" if x is not None else "  -  "
            mean = means.get(rid, means.get("HOUSE") if rid == "HOUSE_D" else None)
            print(f"{rid:<10} {fmt(kd):>11} {fmt(pdp):>10} {fmt(f.get('p_d')):>7} "
                  f"{(f'{mean:+.1f}' if mean is not None else '-'):>7}")
        missing = [r for r, f in found.items() if f.get("p_d") is None]
        if missing:
            print("no anchor found for:", ", ".join(missing))
        if a.write:
            write_priors(cfg.races_file, means, found)
            write_anchor_index("data/anchors_2026.json", found)
            print(f"wrote {cfg.races_file} and data/anchors_2026.json")
        return 0

    if a.cmd == "draft-map":
        from .live import draft_market_map
        return asyncio.run(draft_market_map(cfg, a.out))

    if a.cmd == "record":
        from .recorder import record
        record(a.interval, a.minutes)
        return 0

    if a.cmd == "tick-report":
        from .tickstats import report
        print(report(a.dir))
        return 0

    if a.cmd == "preflight":
        from .live import preflight
        return asyncio.run(preflight(cfg))

    if a.cmd == "kill":
        with open(cfg.risk.kill_file, "w") as f:
            f.write(time.ctime() + "\n")
        print(f"created {cfg.risk.kill_file}: the engine will cancel all orders and stop trading")
        return 0

    if a.cmd == "unkill":
        if os.path.exists(cfg.risk.kill_file):
            os.remove(cfg.risk.kill_file)
        print("kill switch cleared")
        return 0

    if a.cmd == "cancel-all":
        from .exchange.supermarket import SuperMarketExchange

        async def go():
            ex = SuperMarketExchange(cfg.venue, allow_orders=True)
            try:
                print("cancelled", await ex.cancel_all())
            finally:
                await ex.close()
        asyncio.run(go())
        return 0
    return 1


async def _paper(cfg, seed: int, speed: float, quiet: bool) -> int:
    """Paper trading: the real engine against the mock exchange inside one simulated contest."""
    from .engine import Engine
    from .journal import Journal
    from .racemodel import ModelParams, RaceModel
    from .sim.world import TICK_S, SimParams, SimWorld
    world = SimWorld(cfg.races_file, SimParams(seed=seed))
    cfg.open_ts, cfg.election_ts = 0.0, world.pre_ticks * TICK_S
    cfg.account_refresh_s = 0.0
    model = RaceModel.from_json(cfg.races_file, ModelParams(n_draws=2000))
    journal = Journal("journal/paper.sqlite")
    eng = Engine(cfg, world.exchange, world, model, world.markets, journal=journal)
    world.start()
    last_fills = 0
    while world.phase() != "closed":
        before = dict(world.exchange.account.positions)
        await eng.step()
        world.note_bot_flow(before, dict(world.exchange.account.positions))
        eq = world.exchange.account.mark_to(world.truth(0))
        new = world.exchange.fills[last_fills:]
        last_fills = len(world.exchange.fills)
        if not quiet and (new or world.t % 8 == 0):
            day = world.t / world.p.ticks_per_day if world.phase() != "night" else world.p.days
            print(f"t={world.t:>3} {world.phase():<10} day {day:5.1f}  equity {eq:>10,.0f}  "
                  f"positions {len(world.exchange.account.positions):>2}  fills this tick {len(new):>3}"
                  f"  tournament scale {eng.kelly_scale:.2f}")
        if speed:
            await asyncio.sleep(speed)
        world.advance()
    world.resolve_all()
    journal.close()
    print(f"\nresolved: final balance {world.exchange.account.cash:,.0f} "
          f"({world.exchange.account.cash / cfg.starting_cash:.2f}x); Democrats won {world.seats} Senate seats")
    print(f"orders sent {eng.stats.orders_sent}, blocked by risk {eng.stats.orders_blocked}, "
          f"by reason {eng.stats.reasons}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
