"""Offline tests of the live plumbing against a fake Cup API (no network, no aiohttp)."""

import asyncio
import os
import unittest

from cupbot.anchors import AnchorBook, _cents_or_dollars
from cupbot.config import FairValueConfig, VenueConfig
from cupbot.discover import anchors_for, guess_market
from cupbot.engine import Engine
from cupbot.exchange.supermarket import LiveTradingDisabled, SuperMarketExchange
from cupbot.fairvalue import FairValueEngine
from cupbot.racemodel import ModelParams, RaceModel
from cupbot.types import Market, OrderRequest, OrderStatus, Side

RACES = os.path.join(os.path.dirname(__file__), "..", "data", "races_2026.json")


class FakeCup(SuperMarketExchange):
    """Routes requests to canned responses; records what was sent."""

    def __init__(self, routes, **kw):
        cfg = VenueConfig(read_rps=1000, write_rps=1000, **kw.pop("cfg", {}))
        super().__init__(cfg, api_key="test-key", **kw)
        self.routes = routes
        self.sent = []

    async def _send(self, method, url, params, json_body, headers, write=False):
        self.sent.append((method, url, params, json_body, headers))
        path = url.replace(self.cfg.base_url, "")
        r = self.routes[(method, path)]
        if callable(r):
            r = r()
        if isinstance(r, Exception):
            raise r
        return r


def run(c):
    return asyncio.run(c)


class TestSuperMarketClient(unittest.TestCase):
    def test_reads_parse(self):
        ex = FakeCup({
            ("GET", "/markets"): (200, {}, {"markets": [{"id": "m1", "title": "Democrat wins NC Senate"},
                                                         {"id": "m2", "title": "x", "resolved": True}]}),
            ("GET", "/exchanges/prices"): (200, {"X-RateLimit-Remaining": "9"},
                                           [{"id": "m1", "best_bid": "41", "best_ask": "43", "best_bid_size": 10}]),
            ("GET", "/portfolio/balance"): (200, {}, {"balance": "100000.00"}),
            ("GET", "/portfolio/positions"): (200, {}, [{"market_id": "m1", "quantity": "5", "side": "no"}]),
        }, cfg={"price_scale": 100.0})
        ms = run(ex.get_markets())
        self.assertEqual([m.market_id for m in ms], ["m1", "m2"])
        self.assertTrue(ms[1].closed)
        b = run(ex.get_books(["m1"]))["m1"]
        self.assertAlmostEqual(b.best_bid, 0.41)
        self.assertAlmostEqual(b.best_ask, 0.43)
        self.assertEqual(ex.last_rate_headers, {"X-RateLimit-Remaining": "9"})
        acct = run(ex.get_account())
        self.assertEqual(acct.cash, 100000.0)
        self.assertEqual(acct.positions, {"m1": -5.0})

    def test_orders_refused_unless_enabled(self):
        ex = FakeCup({}, allow_orders=True)   # config flag still false
        with self.assertRaises(LiveTradingDisabled):
            run(ex.place_order(OrderRequest("m1", Side.BUY, 10, 0.4, client_id="c1")))
        self.assertEqual(ex.sent, [])

    def test_order_payload_and_idempotency(self):
        ex = FakeCup({("POST", "/orders"): (200, {}, {"id": "o1", "status": "filled", "filled_quantity": 10,
                                                      "average_price": 40})},
                     allow_orders=True, cfg={"live_trading_enabled": True, "price_scale": 100.0})
        res = run(ex.place_order(OrderRequest("m1", Side.BUY, 10, 0.4, client_id="c1")))
        self.assertEqual(res.status, OrderStatus.FILLED)
        self.assertAlmostEqual(res.avg_price, 0.40)
        method, url, params, body, headers = ex.sent[0]
        self.assertEqual(headers["Idempotency-Key"], "c1")
        self.assertEqual(body["limit_price"], "40.0")
        self.assertEqual(body["quantity"], "10")

    def test_timeout_is_unknown_not_resent(self):
        ex = FakeCup({("POST", "/orders"): asyncio.TimeoutError()}, allow_orders=True,
                     cfg={"live_trading_enabled": True})
        res = run(ex.place_order(OrderRequest("m1", Side.BUY, 10, 0.4, client_id="c1")))
        self.assertEqual(res.status, OrderStatus.UNKNOWN)
        self.assertEqual(len(ex.sent), 1)

    def test_429_then_success_and_5xx_retry(self):
        seq = iter([(429, {"Retry-After": "0"}, {}), (503, {}, {}), (200, {}, {"ok": True})])
        ex = FakeCup({("GET", "/me"): lambda: next(seq)})
        self.assertEqual(run(ex.whoami()), {"ok": True})
        self.assertEqual(len(ex.sent), 3)

    def test_4xx_raises(self):
        ex = FakeCup({("GET", "/me"): (401, {}, {"error": "bad key"})})
        with self.assertRaises(RuntimeError):
            run(ex.whoami())

    def test_cancel_all_filters_market(self):
        ex = FakeCup({("GET", "/orders"): (200, {}, [{"id": "a", "market_id": "m1"}, {"id": "b", "market_id": "m2"}]),
                      ("DELETE", "/orders/a"): (200, {}, {}), ("DELETE", "/orders/b"): (200, {}, {})})
        self.assertEqual(run(ex.cancel_all("m1")), 1)
        self.assertEqual([s[1].rsplit("/", 1)[-1] for s in ex.sent if s[0] == "DELETE"], ["a"])


class TestEngineUnknownOrders(unittest.TestCase):
    def test_unknown_blocks_market_until_resolved(self):
        from cupbot.config import BotConfig
        from cupbot.sim.world import TICK_S, SimParams, SimWorld
        w = SimWorld(RACES, SimParams(seed=4))
        cfg = BotConfig()
        cfg.open_ts, cfg.election_ts = 0.0, w.pre_ticks * TICK_S
        eng = Engine(cfg, w.exchange, w, RaceModel.from_json(RACES, ModelParams(n_draws=300)), w.markets)
        mid = w.markets[0].market_id
        eng.unknown["cb-x"] = mid
        looked = []

        async def lookup(cid):
            looked.append(cid)
            return {"client_order_id": cid}
        w.exchange.lookup_order = lookup
        w.start()
        run(eng._resolve_unknown())
        self.assertEqual(looked, ["cb-x"])
        self.assertEqual(eng.unknown, {})


class TestAnchorsAndDiscovery(unittest.TestCase):
    def test_anchor_book(self):
        ab = AnchorBook(max_spread=0.05)
        seen = []
        ab.on_change = seen.append
        ab.update("m", "kalshi", 0.60, 0.62)
        ab.update("m", "polymarket", 0.36, 0.38, invert=True)      # venue YES is the other side
        self.assertAlmostEqual(ab.latest()["m"], 0.62, places=2)
        ab.update("x", "kalshi", 0.2, 0.5)                          # too wide, no last -> ignored
        self.assertNotIn("x", ab.latest())
        self.assertEqual(seen, ["m", "m"])

    def test_polymarket_price_change_message(self):
        # real message shape recorded from the public market websocket on Sep 30
        from cupbot.anchors import AnchorFeed
        ab = AnchorBook()
        feed = AnchorFeed.__new__(AnchorFeed)
        feed.book = ab
        tokens = {"T1": ("m1", False), "T2": ("m2", True)}
        msg = {"market": "0x81", "event_type": "price_change", "timestamp": "1790803422968", "price_changes": [
            {"asset_id": "T1", "price": "0.012", "size": "0", "side": "BUY", "best_bid": "0.056", "best_ask": "0.067"},
            {"asset_id": "T2", "price": "0.988", "size": "0", "side": "SELL", "best_bid": "0.933", "best_ask": "0.944"}]}
        feed._poly_event(msg, tokens, {})
        got = ab.latest()
        self.assertAlmostEqual(got["m1"], 0.0615, places=4)
        self.assertAlmostEqual(got["m2"], 1 - 0.9385, places=4)

    def test_kalshi_price_formats(self):
        self.assertAlmostEqual(_cents_or_dollars({"yes_bid": 41}, "yes_bid"), 0.41)
        self.assertAlmostEqual(_cents_or_dollars({"yes_bid_dollars": "0.4100"}, "yes_bid"), 0.41)

    def test_guess_market(self):
        self.assertEqual(guess_market("Will the Democrat win the North Carolina Senate race?"),
                         {"kind": "race", "race_id": "SEN-NC", "d_sign": 1})
        self.assertEqual(guess_market("Republicans win West Virginia Senate")["race_id"], "SEN-WV")
        self.assertEqual(guess_market("Governor of Ohio: Democrat wins?")["race_id"], "GOV-OH")
        self.assertEqual(guess_market("Will a Republican win PA-08?"),
                         {"kind": "other", "race_id": "HOUSE-PA08", "d_sign": -1})
        self.assertEqual(guess_market("Democrats control the Senate")["race_id"], "SENATE_D")

    def test_anchors_for_sides(self):
        idx = {"SEN-NC": {"kalshi_d": "KD", "kalshi_r": "KR", "poly_r_token": "PR"}}
        self.assertEqual(anchors_for("SEN-NC", 1, idx),
                         {"kalshi": {"ticker": "KD", "invert": False}, "polymarket": {"token_id": "PR", "invert": True}})
        self.assertEqual(anchors_for("SEN-NC", -1, idx)["kalshi"], {"ticker": "KR", "invert": False})

    def test_anchor_only_market_gets_fair(self):
        ms = {"G": Market("G", "Governor of Ohio: Democrat", "other", "GOV-OH", 1),
              "N": Market("N", "no anchor", "other", "GOV-XX", 1)}
        fv = FairValueEngine(FairValueConfig(), ms, RaceModel.from_json(RACES, ModelParams(n_draws=200)))
        out = fv.compute(0, {}, {"G": 0.3}, {"G": 0.4, "N": 0.5})
        self.assertIn("G", out)
        self.assertNotIn("N", out)
        self.assertLess(out["G"].p, 0.33)


class TestCalls(unittest.TestCase):
    def setUp(self):
        from cupbot.calls import CallParser
        idx = {"SEN-NC": {"cand_d": "Roy Cooper", "cand_r": "Michael Whatley"},
               "SEN-NE": {"cand_d": "Cindy Burbank", "cand_r": "Pete Ricketts", "cand_other": ["Dan Osborn"]},
               "SEN-VA": {"cand_d": "Mark Warner", "cand_r": "Bert Mizusawa"}}
        self.p = CallParser(idx)

    def test_parse_calls(self):
        self.assertEqual(self.p.parse("RACE CALL: Roy Cooper wins the North Carolina Senate race. #APracecall"),
                         ("SEN-NC", "D"))
        self.assertEqual(self.p.parse("Decision Desk HQ projects Dan Osborn (I) wins the Nebraska Senate race"),
                         ("SEN-NE", "D"))
        self.assertEqual(self.p.parse("BREAKING: Republican wins West Virginia Senate race"), ("SEN-WV", "R"))

    def test_rejects_non_calls(self):
        self.assertIsNone(self.p.parse("Roy Cooper leads in the North Carolina Senate race with 30% in"))
        self.assertIsNone(self.p.parse("North Carolina Senate race is too close to call"))
        self.assertIsNone(self.p.parse("Mark Warner wins the North Carolina Senate race"))  # name/state mismatch
        self.assertIsNone(self.p.parse("Democrat wins Virginia House race"))

    def test_confirmation_rule(self):
        from cupbot.calls import Call, CallBook
        b = CallBook(min_sources=2, agree_p=0.85)
        b.add(Call("SEN-NC", "D", "ap", 0, ""))
        self.assertEqual(b.effective(), {})
        self.assertEqual(b.effective({"SEN-NC": (0.9, 0.01)}), {"SEN-NC": "D"})
        self.assertEqual(b.effective({"SEN-NC": (0.6, 0.01)}), {})
        b.add(Call("SEN-NC", "D", "ddhq", 0, ""))
        self.assertEqual(b.effective(), {"SEN-NC": "D"})
        b.add(Call("SEN-NC", "R", "someone", 0, ""))
        self.assertEqual(b.effective(), {})   # conflicting calls cancel


if __name__ == "__main__":
    unittest.main()


class TestUnattended(unittest.TestCase):
    def test_tee_journal_forwards_alerts(self):
        from cupbot.journal import NullJournal
        from cupbot.notify import Notifier, TeeJournal
        sent = []
        n = Notifier("t", "tok")
        n.send = lambda title, body, priority="default", key="": sent.append((title, priority))
        j = TeeJournal(NullJournal(), n)
        j.log("halt", {"reason": "drawdown"})
        j.log("signal", {"x": 1})
        j.log("order", {"filled": 20000, "avg": 0.5, "side": "buy", "market": "m", "reason": "take"})
        self.assertEqual(sent, [("cupbot: halt", "urgent"), ("cupbot: big fill", "default")])

    def test_phone_commands(self):
        import tempfile
        from cupbot.config import BotConfig
        from cupbot.live import _command_handler

        class E:
            last_acct = None
            last_equity = 123.0
            last_error = ""

            class risk:
                class state:
                    halted = False
                    halt_reason = ""

            def poke(self):
                pass
        cfg = BotConfig()
        cfg.risk.kill_file = os.path.join(tempfile.mkdtemp(), "KILL")
        h = _command_handler(cfg, E())
        self.assertIn("KILL set", h("kill"))
        self.assertTrue(os.path.exists(cfg.risk.kill_file))
        self.assertIn("kill ON", h("status"))
        self.assertIn("cleared", h("resume"))
        self.assertFalse(os.path.exists(cfg.risk.kill_file))
        self.assertIsNone(h("buy"))   # there is no trading command

    def test_engine_stop(self):
        from cupbot.config import BotConfig
        from cupbot.sim.world import SimParams, SimWorld
        w = SimWorld(RACES, SimParams(seed=5))
        cfg = BotConfig()
        cfg.loop_interval_s = 0.01
        eng = Engine(cfg, w.exchange, w, RaceModel.from_json(RACES, ModelParams(n_draws=200)), w.markets,
                     dry_run=True)

        async def go():
            async def stopper():
                await asyncio.sleep(0.2)
                eng.stop()
            asyncio.ensure_future(stopper())
            await asyncio.wait_for(eng.run(), timeout=5)
        run(go())
        self.assertTrue(eng.stopping)
