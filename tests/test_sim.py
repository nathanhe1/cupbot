import asyncio
import os
import unittest

from cupbot.config import BotConfig
from cupbot.engine import Engine
from cupbot.racemodel import ModelParams, RaceModel
from cupbot.sim.backtest import run_one
from cupbot.sim.world import TICK_S, SimParams, SimWorld

RACES = os.path.join(os.path.dirname(__file__), "..", "data", "races_2026.json")


class TestSimulation(unittest.TestCase):
    def test_full_contest_runs_without_breaches(self):
        r = run_one(5, races=RACES)
        self.assertEqual(r.risk_breaches, 0)
        self.assertGreater(r.orders, 0)
        self.assertGreater(r.final, 0)

    def test_do_nothing_keeps_cash(self):
        r = run_one(5, races=RACES, strategy_off=True)
        self.assertAlmostEqual(r.final, 100_000.0)

    def test_combo_truth_partitions(self):
        w = SimWorld(RACES, SimParams(seed=2))
        t = w.truth()
        self.assertAlmostEqual(sum(t[f"M-BOP-{k}"] for k in ("DD", "DR", "RD", "RR")), 1.0, places=9)

    def test_dry_run_sends_nothing(self):
        w = SimWorld(RACES, SimParams(seed=3))
        cfg = BotConfig()
        cfg.open_ts, cfg.election_ts = 0.0, w.pre_ticks * TICK_S
        eng = Engine(cfg, w.exchange, w, RaceModel.from_json(RACES, ModelParams(n_draws=300)),
                     w.markets, dry_run=True)
        w.start()

        async def go():
            for _ in range(10):
                await eng.step()
                w.advance()
        asyncio.run(go())
        self.assertEqual(w.exchange.order_count, 0)
        self.assertEqual(w.exchange.account.positions, {})

    def test_kill_switch_stops_trading(self):
        import tempfile
        tmp = tempfile.mkdtemp()
        open(os.path.join(tmp, "KILL"), "w").close()
        w = SimWorld(RACES, SimParams(seed=3))
        cfg = BotConfig()
        eng = Engine(cfg, w.exchange, w, RaceModel.from_json(RACES, ModelParams(n_draws=300)),
                     w.markets, run_dir=tmp)
        w.start()
        asyncio.run(eng.step())
        self.assertEqual(w.exchange.order_count, 0)


if __name__ == "__main__":
    unittest.main()
