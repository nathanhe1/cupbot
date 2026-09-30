import asyncio
import math
import os
import random
import tempfile
import unittest

from cupbot.analyst import to_signal
from cupbot.config import BotConfig, FairValueConfig, RiskConfig, StrategyConfig, load_config
from cupbot.election_night import LiveResultsModel, NightParams, RaceReport
from cupbot.exchange.mock import MockExchange
from cupbot.fairvalue import FairQuote, FairValueEngine, inverse_variance_blend
from cupbot.mathutil import (kelly_no, kelly_yes, logit, logit_blend, norm_cdf, norm_ppf,
                             poisson_binomial_at_least, sigmoid, student_t)
from cupbot.racemodel import ModelParams, Race, RaceModel
from cupbot.risk import RiskManager
from cupbot.strategy import Strategy
from cupbot.tournament import best_scale
from cupbot.types import (Account, Level, Market, OrderBook, OrderKind, OrderRequest, OrderStatus, Side,
                          Signal, TimeInForce)

RACES = os.path.join(os.path.dirname(__file__), "..", "data", "races_2026.json")


def run(coro):
    return asyncio.run(coro)


class TestMath(unittest.TestCase):
    def test_logit_roundtrip(self):
        for p in (0.01, 0.2, 0.5, 0.93):
            self.assertAlmostEqual(sigmoid(logit(p)), p, places=9)

    def test_norm(self):
        self.assertAlmostEqual(norm_cdf(0), 0.5)
        self.assertAlmostEqual(norm_cdf(norm_ppf(0.9)), 0.9, places=6)

    def test_kelly(self):
        self.assertAlmostEqual(kelly_yes(0.8, 0.65), (0.8 - 0.65) / 0.35)
        self.assertEqual(kelly_yes(0.5, 0.6), 0.0)
        self.assertAlmostEqual(kelly_no(0.3, 0.5), 0.4)

    def test_blend(self):
        self.assertAlmostEqual(logit_blend([0.75, 0.65], [0.3, 0.7]), 0.682, places=3)
        p, unc = inverse_variance_blend([(0.6, 0.1), (0.9, 100.0)])
        self.assertAlmostEqual(p, 0.6, places=3)
        self.assertGreater(unc, 0)

    def test_poisson_binomial(self):
        self.assertAlmostEqual(poisson_binomial_at_least([0.5, 0.5], 1), 0.75)
        self.assertAlmostEqual(poisson_binomial_at_least([1, 1, 0], 2), 1.0)

    def test_student_t_unit_variance(self):
        rng = random.Random(0)
        xs = [student_t(rng, 6) for _ in range(40000)]
        var = sum(x * x for x in xs) / len(xs)
        self.assertAlmostEqual(var, 1.0, delta=0.08)


class TestRaceModel(unittest.TestCase):
    def setUp(self):
        self.m = RaceModel.from_json(RACES, ModelParams(n_draws=4000, seed=3))

    def test_mc_matches_analytic(self):
        s = self.m.summary()
        for rid in ("SEN-NC", "SEN-TX", "SEN-MI", "HOUSE"):
            self.assertAlmostEqual(s[rid], self.m.analytic_prob(rid), delta=0.04)

    def test_combos_partition(self):
        s = self.m.summary()
        self.assertAlmostEqual(s["COMBO_DD"] + s["COMBO_DR"] + s["COMBO_RD"] + s["COMBO_RR"], 1.0, places=9)
        self.assertAlmostEqual(s["COMBO_DD"] + s["COMBO_RD"], s["SENATE_D"], places=9)

    def test_invert_probs_roundtrip(self):
        target = {"SEN-NC": 0.62, "SEN-ME": 0.4, "HOUSE": 0.7}
        self.m.set_means_from_probs(target)
        for rid, p in target.items():
            self.assertAlmostEqual(self.m.analytic_prob(rid), p, places=6)

    def test_senate_monotone_in_national_shift(self):
        base = self.m.summary()["SENATE_D"]
        for r in self.m.races:
            r.mean += 3
        self.assertGreater(self.m.summary()["SENATE_D"], base)

    def test_sd_scale_sharpens(self):
        fav = max(self.m.races, key=lambda r: r.mean).race_id
        p1 = self.m.analytic_prob(fav)
        self.m.sd_scale = 0.5
        self.assertGreater(self.m.analytic_prob(fav), p1)


def mk_markets():
    return [Market("A", "D wins A", "race", "SEN-NC", 1), Market("B", "R wins B", "race", "SEN-TX", -1)]


class TestMockExchange(unittest.TestCase):
    def setUp(self):
        self.ex = MockExchange(mk_markets(), starting_cash=1000.0)

    def test_price_time_priority_and_partial(self):
        ex = self.ex
        ex.submit("mm1", OrderRequest("A", Side.SELL, 10, 0.50, tif=TimeInForce.GTC))
        ex.submit("mm2", OrderRequest("A", Side.SELL, 10, 0.50, tif=TimeInForce.GTC))
        ex.submit("mm3", OrderRequest("A", Side.SELL, 10, 0.49, tif=TimeInForce.GTC))
        res = run(ex.place_order(OrderRequest("A", Side.BUY, 15, 0.50)))
        self.assertEqual(res.status, OrderStatus.FILLED)
        self.assertEqual([(f.price, f.qty) for f in res.fills], [(0.49, 10), (0.50, 5)])
        self.assertEqual(ex.counterparty_volume, {"mm3": 10, "mm1": 5})
        self.assertAlmostEqual(ex.account.cash, 1000 - 4.9 - 2.5)
        self.assertEqual(ex.account.pos("A"), 15)

    def test_ioc_does_not_rest(self):
        res = run(self.ex.place_order(OrderRequest("A", Side.BUY, 5, 0.40)))
        self.assertEqual(res.status, OrderStatus.CANCELLED)
        self.assertEqual(run(self.ex.open_orders()), [])

    def test_gtc_rests_and_fills_later(self):
        res = run(self.ex.place_order(OrderRequest("A", Side.BUY, 5, 0.40, tif=TimeInForce.GTC)))
        self.assertEqual(res.status, OrderStatus.ACKED)
        self.ex.submit("stud", OrderRequest("A", Side.SELL, 5, None, kind=OrderKind.MARKET))
        self.assertEqual(self.ex.account.pos("A"), 5)

    def test_balance_limit(self):
        self.ex.submit("mm", OrderRequest("A", Side.SELL, 10000, 0.5, tif=TimeInForce.GTC))
        res = run(self.ex.place_order(OrderRequest("A", Side.BUY, 10000, 0.5)))
        self.assertAlmostEqual(res.filled_qty, 2000)
        self.assertGreaterEqual(self.ex.account.cash, -1e-9)

    def test_short_collateral_and_resolution(self):
        self.ex.submit("mm", OrderRequest("A", Side.BUY, 5000, 0.6, tif=TimeInForce.GTC))
        res = run(self.ex.place_order(OrderRequest("A", Side.SELL, 5000, 0.6)))
        # each short needs 1 - 0.6 = 0.4 of free cash: 1000 / 0.4 = 2500
        self.assertAlmostEqual(res.filled_qty, 2500)
        self.ex.resolve("A", yes=True)
        self.assertAlmostEqual(self.ex.account.cash, 1000 + 2500 * 0.6 - 2500)

    def test_no_short_mode(self):
        ex = MockExchange(mk_markets(), 1000.0, allow_short=False)
        ex.submit("mm", OrderRequest("A", Side.BUY, 100, 0.6, tif=TimeInForce.GTC))
        res = run(ex.place_order(OrderRequest("A", Side.SELL, 10, 0.6)))
        self.assertEqual(res.status, OrderStatus.REJECTED)

    def test_self_trade_prevention(self):
        run(self.ex.place_order(OrderRequest("A", Side.SELL, 5, 0.55, tif=TimeInForce.GTC)))
        self.ex.submit("mm", OrderRequest("A", Side.SELL, 5, 0.56, tif=TimeInForce.GTC))
        res = run(self.ex.place_order(OrderRequest("A", Side.BUY, 5, 0.56)))
        self.assertEqual([f.price for f in res.fills], [0.56])  # own 0.55 ask was cancelled, not hit


class TestRisk(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.markets = {m.market_id: m for m in mk_markets()}
        self.rm = RiskManager(RiskConfig(max_order_notional=500, max_market_notional_frac=0.3,
                                         max_net_d_frac=0.5, max_orders_per_minute=3), self.markets, self.tmp)
        self.book = OrderBook("A", [Level(0.49, 100)], [Level(0.51, 100)], ts=100.0)
        self.acct = Account(1000.0)

    def chk(self, req, acct=None, equity=1000.0, now=100.0):
        return self.rm.check(req, acct or self.acct, self.book, {"A": 0.5, "B": 0.5}, equity, now)

    def test_ok(self):
        self.assertTrue(self.chk(OrderRequest("A", Side.BUY, 100, 0.51))[0])

    def test_kill_switch(self):
        open(os.path.join(self.tmp, "KILL"), "w").close()
        ok, why = self.chk(OrderRequest("A", Side.BUY, 1, 0.51))
        self.assertFalse(ok)
        self.assertEqual(why, "kill switch")

    def test_order_and_market_caps(self):
        self.assertFalse(self.chk(OrderRequest("A", Side.BUY, 1100, 0.51))[0])        # > 500 notional
        acct = Account(1000.0, {"A": 500})
        self.assertFalse(self.chk(OrderRequest("A", Side.BUY, 200, 0.51), acct)[0])   # market > 30%

    def test_reduce_only_allowed_when_halted(self):
        self.rm.update_equity(1000, 0)
        self.rm.update_equity(500, 10)
        self.assertTrue(self.rm.state.halted)
        acct = Account(500.0, {"A": 100})
        self.assertFalse(self.chk(OrderRequest("A", Side.BUY, 10, 0.51), acct)[0])
        self.assertTrue(self.chk(OrderRequest("A", Side.SELL, 50, 0.49), acct)[0])

    def test_price_band_and_stale(self):
        self.assertFalse(self.chk(OrderRequest("A", Side.BUY, 10, 0.95))[0])
        self.assertFalse(self.chk(OrderRequest("A", Side.BUY, 10, 0.51), now=1000.0)[0])

    def test_rate_limit(self):
        for _ in range(3):
            self.assertTrue(self.chk(OrderRequest("A", Side.BUY, 1, 0.51))[0])
        self.assertFalse(self.chk(OrderRequest("A", Side.BUY, 1, 0.51))[0])

    def test_net_directional(self):
        # long D in A plus 'short R' (sell YES) in B are both pro-D exposure
        acct = Account(1000.0, {"A": 500})
        self.assertFalse(self.chk(OrderRequest("B", Side.SELL, 600, 0.49), acct)[0])
        self.assertTrue(self.chk(OrderRequest("B", Side.BUY, 400, 0.51), acct)[0])  # hedges


class TestFairValue(unittest.TestCase):
    def setUp(self):
        self.markets = {m.market_id: m for m in mk_markets()}
        self.model = RaceModel.from_json(RACES, ModelParams(n_draws=500))
        self.fv = FairValueEngine(FairValueConfig(), self.markets, self.model)

    def test_anchor_dominates_and_sign(self):
        out = self.fv.compute(0, {"SEN-NC": 0.5, "SEN-TX": 0.2}, {"A": 0.7, "B": 0.75}, {})
        self.assertGreater(out["A"].p, 0.67)
        # B is 'R wins TX': model P(D)=0.2 -> P(YES)=0.8, anchor 0.75
        self.assertTrue(0.75 <= out["B"].p <= 0.8)

    def test_signal_needs_source_and_is_capped_and_decays(self):
        self.assertFalse(self.fv.add_signal(Signal("A", 0.2, "high", "x", [], ts=0)))
        self.assertTrue(self.fv.add_signal(Signal("A", 0.2, "high", "x", ["https://apnews.com/a"], ts=0, ttl_s=100)))
        self.assertAlmostEqual(self.fv.signal_shift("A", 0), FairValueConfig().signal_cap)
        self.assertAlmostEqual(self.fv.signal_shift("A", 50), FairValueConfig().signal_cap / 2)
        self.assertEqual(self.fv.signal_shift("A", 150), 0.0)


class TestStrategy(unittest.TestCase):
    def setUp(self):
        self.markets = {m.market_id: m for m in mk_markets()}
        self.s = Strategy(StrategyConfig(min_edge=0.03, uncertainty_mult=0.0), self.markets)

    def book(self, bid, ask, qty=1000, mid="A"):
        return OrderBook(mid, [Level(bid, qty)], [Level(ask, qty)])

    def test_buys_below_fair_with_limit(self):
        o = self.s.decide({"A": FairQuote(0.70, 0.0)}, {"A": self.book(0.55, 0.60)}, Account(10000), 10000, "pre")
        self.assertEqual(len(o), 1)
        self.assertEqual(o[0].side, Side.BUY)
        self.assertAlmostEqual(o[0].price, 0.67)
        self.assertLessEqual(o[0].qty * o[0].price, 0.05 * 10000 + 1)  # per-order cap

    def test_no_trade_inside_threshold(self):
        self.assertEqual(self.s.decide({"A": FairQuote(0.62, 0.0)}, {"A": self.book(0.58, 0.60)},
                                       Account(10000), 10000, "pre"), [])

    def test_sells_above_fair(self):
        o = self.s.decide({"A": FairQuote(0.30, 0.0)}, {"A": self.book(0.40, 0.42)}, Account(10000), 10000, "pre")
        self.assertEqual(o[0].side, Side.SELL)
        self.assertAlmostEqual(o[0].price, 0.33)

    def test_threshold_rules(self):
        s = Strategy(StrategyConfig(min_edge=0.02, gap_buffer=0.01, lag_min_edge=0.015, lag_move=0.02,
                                    uncertainty_mult=0.0), self.markets)
        fq = FairQuote(0.6, 0.0, {"anchor": 0.6})
        self.assertAlmostEqual(s.threshold("A", fq, 0.02), 0.02)
        self.assertAlmostEqual(s.threshold("A", fq, 0.02, venue_gap={"A": 0.03}), 0.04)       # gap + 1
        self.assertAlmostEqual(s.threshold("A", fq, 0.02, venue_gap={"A": 0.03},
                                           anchor_moves={"A": 0.025}), 0.015)                  # lag trade

    def test_first_move_halves_target_not_each_order(self):
        o = self.s.decide({"A": FairQuote(0.70, 0.0)}, {"A": self.book(0.55, 0.60, qty=1e6)},
                          Account(10000, {"A": 0}), 10000, "pre", damp={"A": 0.5})
        full = self.s.decide({"A": FairQuote(0.70, 0.0)}, {"A": self.book(0.55, 0.60, qty=1e6)},
                             Account(10000, {"A": 0}), 10000, "pre", damp={})
        # with a position already at the halved target, no more is bought until damp lifts
        held = self.s.decide({"A": FairQuote(0.70, 0.0)}, {"A": self.book(0.55, 0.60, qty=1e6)},
                             Account(10000, {"A": o[0].qty + 5}), 10000, "pre", damp={"A": 0.5})
        self.assertLessEqual(o[0].qty, full[0].qty)
        self.assertEqual([x for x in held if x.side == Side.BUY], [])

    def test_recycle_on_night(self):
        books = {"A": self.book(0.96, 0.98), "B": self.book(0.40, 0.42, mid="B")}
        fairs = {"A": FairQuote(0.985, 0.0), "B": FairQuote(0.60, 0.0)}
        o = self.s.decide(fairs, books, Account(100, {"A": 500}), 10000, "night")
        self.assertIn(("A", Side.SELL), [(x.market_id, x.side) for x in o])

    def test_combo_arb(self):
        ms = {f"C{i}": Market(f"C{i}", "", "combo", f"COMBO_{i}", 1, group="G") for i in range(4)}
        s = Strategy(StrategyConfig(), ms)
        books = {k: OrderBook(k, [Level(0.18, 100)], [Level(0.22, 100)]) for k in ms}
        o = s.decide({}, books, Account(10000), 10000, "pre")
        self.assertEqual(len(o), 4)
        self.assertTrue(all(x.side == Side.BUY for x in o))


class TestElectionNight(unittest.TestCase):
    def test_mirage_and_swing(self):
        p = NightParams(mirage={"PA": -5.0})
        m = LiveResultsModel({"SEN-PA": 0.0, "SEN-X": 0.0}, {"SEN-PA": "PA", "SEN-X": "XX"}, p)
        # PA at 40% reporting shows R+3 in counted votes, but early counts lean R by 5*(0.6)=3 -> even
        out = m.update({"SEN-PA": RaceReport("PA", 0.4, -3.0)})
        self.assertAlmostEqual(out["SEN-PA"][0], 0.5, delta=0.02)
        # a D+4 swing in a fully reported race should lift an unreported race
        out = m.update({"SEN-PA": RaceReport("PA", 1.0, 4.0)})
        self.assertGreater(out["SEN-X"][0], 0.6)


class TestAnalystAndTournament(unittest.TestCase):
    def test_to_signal_validation(self):
        good = {"relevant": True, "direction": "toward_no", "delta_points": 4, "confidence": "high",
                "big": False, "reason": "poll", "sources": ["https://www.apnews.com/x", "https://evil.example/y"]}
        s = to_signal("A", good, ["apnews.com"])
        self.assertAlmostEqual(s.delta, -0.04)
        self.assertEqual(s.sources, ["https://www.apnews.com/x"])
        bad = dict(good, sources=["https://evil.example/y"])
        self.assertIsNone(to_signal("A", bad, ["apnews.com"]))
        self.assertIsNone(to_signal("A", dict(good, relevant=False), ["apnews.com"]))

    def test_best_scale(self):
        # far below target: more risk helps; above target: less risk
        pnl = [100.0] * 30 + [-100.0] * 70
        lam, _ = best_scale(1000, pnl, 1250, grid=(0.5, 1, 2, 3))
        self.assertEqual(lam, 3)
        lam, p = best_scale(1000, pnl, 900, grid=(0.5, 1, 2, 3))
        self.assertEqual(lam, 0.5)
        self.assertEqual(p, 1.0)


class TestBasket(unittest.TestCase):
    def test_plan_prefers_basket_that_reaches_target(self):
        from cupbot.tournament import candidate_baskets, plan_basket
        ms = {"A": Market("A", "", "race", "R1", 1), "B": Market("B", "", "race", "R2", 1),
              "C": Market("C", "", "combo", "COMBO_DD", 1, group="G")}
        fairs = {"A": 0.3, "B": 0.3, "C": 0.1}
        asks = {"A": 0.31, "B": 0.31, "C": 0.11}
        bids = {"A": 0.29, "B": 0.29, "C": 0.09}
        cands = candidate_baskets(ms, fairs, asks, bids)
        self.assertIn("D outperforms polls", cands)
        self.assertIn("single C", cands)
        # 4 scenarios: both D win together half the time, DD combo pays in one
        outcomes = {"R1": [1, 1, 0, 0], "R2": [1, 1, 0, 0], "COMBO_DD": [1, 0, 0, 0]}
        name, frac, p, qty = plan_basket(1000, [0, 0, 0, 0], 1500, cands, ms, outcomes, 4)
        self.assertEqual(name, "D outperforms polls")
        self.assertAlmostEqual(p, 0.5)
        self.assertTrue(all(q > 0 for q in qty.values()))


class TestConfig(unittest.TestCase):
    def test_default_json_loads_and_live_disabled(self):
        cfg = load_config(os.path.join(os.path.dirname(__file__), "..", "config", "default.json"))
        self.assertFalse(cfg.venue.live_trading_enabled)

    def test_unknown_key_rejected(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            f.write('{"risk": {"no_such_key": 1}}')
        with self.assertRaises(KeyError):
            load_config(f.name)


if __name__ == "__main__":
    unittest.main()
