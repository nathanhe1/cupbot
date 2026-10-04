import unittest

from cupbot.config import LadderConfig
from cupbot.fairvalue import FairQuote
from cupbot.journal import NullJournal
from cupbot.ladder import Ladder, ask_tick, bid_tick
from cupbot.types import Account, Level, Market, OrderBook, Side


def book(mid, bid, ask):
    return OrderBook(mid, [Level(bid, 1000)], [Level(ask, 1000)])


class NoSharesAccountTest(unittest.TestCase):
    def test_no_position_valued_at_one_minus_mark(self):
        a = Account(cash=1000, positions={"x": -100}, no_shares=True)
        self.assertAlmostEqual(a.mark_to({"x": 0.2}), 1000 + 100 * 0.8)
        old = Account(cash=1000, positions={"x": -100})
        self.assertAlmostEqual(old.mark_to({"x": 0.2}), 1000 - 100 * 0.2)

    def test_fills_follow_venue_cash(self):
        a = Account(cash=1000, positions={}, no_shares=True)
        a.apply_fill("x", Side.SELL, 100, 0.3)        # opens NO at 0.70
        self.assertAlmostEqual(a.cash, 930)
        self.assertEqual(a.pos("x"), -100)
        a.apply_fill("x", Side.BUY, 150, 0.2)         # closes NO at 0.80, opens 50 YES at 0.20
        self.assertAlmostEqual(a.cash, 930 + 100 * 0.8 - 50 * 0.2)
        self.assertEqual(a.pos("x"), 50)
        # round trip sold YES at 0.3, bought back at 0.2: +10 on 100, equity at mark 0.2
        self.assertAlmostEqual(a.mark_to({"x": 0.2}), 1000 + 10)


class LadderTest(unittest.TestCase):
    def setUp(self):
        self.mk = {"a": Market("a", "A"), "b": Market("b", "B")}
        self.cfg = LadderConfig(markets_file="", total_frac=0.05)
        self.t = [1000.0]
        self.lad = Ladder(self.cfg, self.mk, NullJournal(), clock=lambda: self.t[0])

    def test_ticks(self):
        self.assertEqual(bid_tick(0.4549), 0.45)
        self.assertEqual(ask_tick(0.4501), 0.455)

    def test_levels_straddle_consensus_without_crossing(self):
        self.lad.seed({"a": 0.50})
        qs = self.lad.desired("a", None, 0.0, 100_000, book("a", 0.495, 0.50))
        bids = [p for s, p, _ in qs if s == "buy"]
        asks = [p for s, p, _ in qs if s == "sell"]
        self.assertEqual(bids, [0.455, 0.41])        # unanchored offsets 0.045, 0.09
        self.assertEqual(asks, [0.545, 0.59])
        self.assertTrue(all(b < 0.50 for b in bids) and all(a > 0.495 for a in asks))

    def test_fair_guard_blocks_wrong_side_opens(self):
        self.lad.seed({"a": 0.88})
        fq = FairQuote(0.99, 0.005, {"anchor": 0.99})  # favourite underpriced on the Cup
        qs = self.lad.desired("a", fq, 0.0, 100_000, book("a", 0.875, 0.885))
        self.assertTrue(qs and all(s == "buy" for s, _, _ in qs))   # no new shorts below fair
        # holding a long: asks come back, but only to sell what we hold
        qs = self.lad.desired("a", fq, 300.0, 100_000, book("a", 0.875, 0.885))
        sells = [q for s, _, q in qs if s == "sell"]
        self.assertTrue(sells and sum(sells) <= 300)

    def test_position_cap(self):
        self.lad.seed({"a": 0.50})
        cap = self.cfg.max_pos_frac * 100_000
        qs = self.lad.desired("a", None, cap / 0.5, 100_000, book("a", 0.495, 0.50))
        self.assertFalse([1 for s, _, _ in qs if s == "buy"])        # long at the cap: no more bids

    def test_pause_when_market_runs_away(self):
        self.lad.seed({"a": 0.50})
        self.assertEqual(self.lad.desired("a", None, 0.0, 100_000, book("a", 0.70, 0.71)), [])

    def test_plan_respects_write_budget_and_requotes_on_move(self):
        self.lad.seed({"a": 0.5, "b": 0.3})
        books = {"a": book("a", 0.495, 0.505), "b": book("b", 0.295, 0.305)}
        acct = Account(50_000, {}, no_shares=True)
        chosen, quotes = self.lad.plan({}, books, acct, 100_000)
        self.assertEqual(sorted(chosen), ["a", "b"])
        for m in chosen:
            self.lad.mark_quoted(m, quotes[m], None, 0.0, self.t[0])
        self.assertEqual(self.lad.plan({}, books, acct, 100_000)[0], [])   # nothing changed
        self.lad.ref["a"] = 0.53
        self.t[0] += 30
        self.assertEqual(self.lad.plan({}, books, acct, 100_000)[0], ["a"])

    def test_ema_ignores_wide_books(self):
        self.lad.seed({"a": 0.5})
        self.t[0] += 2700
        self.lad.update_refs({"a": book("a", 0.30, 0.70)})
        self.assertAlmostEqual(self.lad.ref["a"], 0.5)
        self.lad.update_refs({"a": book("a", 0.595, 0.605)})
        self.assertAlmostEqual(self.lad.ref["a"], 0.55, places=3)   # one half-life: halfway


if __name__ == "__main__":
    unittest.main()


class LadderEngineTest(unittest.TestCase):
    """The engine's ladder step against a fake Cup API: cancel per market, one batch create."""

    def test_step_places_batch(self):
        import asyncio
        import os
        from cupbot.config import BotConfig
        from cupbot.engine import Engine
        from cupbot.feeds import Feeds
        from cupbot.racemodel import ModelParams, RaceModel
        from tests.test_live import RACES, FakeCup

        T = "t-1"
        batches = []

        def batch():
            body = ex.sent[-1][3]
            batches.append(body)
            return (200, {}, {"results": [{"index": i, "ok": True, "status": 201,
                                           "data": {"orderId": str(i), "open": True, "quantityTraded": 0}}
                                          for i in range(len(body["orders"]))]})
        ex = FakeCup({
            ("GET", "/exchanges/prices"): (200, {}, {"data": [
                {"exchangeId": "g1", "bestBid": 0.495, "bestAsk": 0.505},
                {"exchangeId": "g2", "bestBid": 0.295, "bestAsk": 0.305}]}),
            ("GET", "/tournaments/midterm-elections"): (200, {}, {"id": T, "myBalance": 90000}),
            ("GET", "/tournaments/midterm-elections/portfolio/positions"): (200, {}, {"positions": [
                {"exchangeId": "g2", "quantity": -1000, "settled": False,
                 "lots": [{"side": "no", "quantity": 1000}]}]}),
            ("POST", "/orders/cancel-all"): (200, {}, {"cancelled": 0}),
            ("POST", "/orders/batch"): batch,
        }, cfg={"tournament_id": T, "live_trading_enabled": True}, allow_orders=True)

        class F(Feeds):
            def now(self): return 1.79e9
            def phase(self): return "pre"
            def anchors(self): return {}
            def poll_means(self): return {}
            def signals(self): return []

        cfg = BotConfig()
        cfg.ladder.markets_file = ""
        cfg.risk.kill_file = "NO-SUCH-KILL-FILE"
        mk = [Market("g1", "Gov 1", "other", "GOV-X1", 1), Market("g2", "Gov 2", "other", "GOV-X2", 1)]
        eng = Engine(cfg, ex, F(), RaceModel.from_json(RACES, ModelParams(n_draws=100)), mk)
        eng.ladder.seed({"g1": 0.50, "g2": 0.30})
        asyncio.run(eng.step())
        self.assertAlmostEqual(eng.last_equity, 90000 + 1000 * 0.70, delta=1)   # NO shares at 1 - mark
        cancels = [s for s in ex.sent if s[1].endswith("/orders/cancel-all")]
        self.assertEqual(sorted(c[3]["exchangeId"] for c in cancels), ["g1", "g2"])
        self.assertEqual(len(batches), 1)
        orders = batches[0]["orders"]
        self.assertTrue(all("expirationDate" in o and o["tournamentId"] == T for o in orders))
        g2_buys = [o for o in orders if o["exchangeId"] == "g2" and o["action"] == "buy"]
        self.assertTrue(g2_buys and all(o["price"] < 0.30 for o in g2_buys))   # bids below consensus
        g1 = sorted(o["price"] for o in orders if o["exchangeId"] == "g1")
        self.assertTrue(g1[0] < 0.495 and g1[-1] > 0.505)
