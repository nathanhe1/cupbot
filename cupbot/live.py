"""Live wiring: feeds from the real world, and the live run loop.

Default is DRY RUN: the engine computes everything and journals the orders it
would send, but sends nothing. Real orders need BOTH venue.live_trading_enabled
= true in the config file AND the --live flag."""

from __future__ import annotations

import asyncio
import csv
import json
import os
import signal
import threading
import time
from typing import Dict, List, Optional

from .anchors import AnchorBook, AnchorFeed
from .config import BotConfig
from .election_night import NightParams, RaceReport
from .engine import Engine
from .feeds import Feeds
from .journal import Journal
from .notify import Notifier, TeeJournal
from .racemodel import ModelParams, RaceModel
from .types import Market, Signal


def load_market_map(path: str) -> Dict[str, dict]:
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        data = json.load(f)
    return {k: v for k, v in data.items() if not k.startswith("_")}


def markets_from_map(mapping: Dict[str, dict], titles: Dict[str, str]) -> List[Market]:
    out = []
    for mid, m in mapping.items():
        out.append(Market(mid, titles.get(mid, m.get("title", mid)), m.get("kind", "race"), m.get("race_id"),
                          int(m.get("d_sign", 0)), m.get("group"), {}))
    return out


class LiveFeeds(Feeds):
    def __init__(self, cfg: BotConfig, anchors: AnchorBook, races_file: str):
        self.cfg = cfg
        self.anchor_book = anchors
        self.races_file = races_file
        self._signals: List[Signal] = []
        self._reports: Optional[Dict[str, RaceReport]] = None
        self._calls = None   # calls.CallBook, set by run_live on election night

    def now(self) -> float:
        return time.time()

    def phase(self) -> str:
        t = time.time()
        close = self.cfg.election_ts + 17 * 3600      # Nov 4 noon ET
        if t >= close:
            return "closed"
        if t >= self.cfg.election_ts:
            return "night"
        if t >= self.cfg.election_ts - 7 * 86400:
            return "final_week"
        return "pre"

    def anchors(self) -> Dict[str, float]:
        return self.anchor_book.latest()

    def anchor_disagreement(self) -> Dict[str, float]:
        return self.anchor_book.disagreement()

    def anchor_moves(self) -> Dict[str, float]:
        return self.anchor_book.recent_moves()

    def poll_means(self) -> Dict[str, float]:
        # re-read so you can edit data/races_2026.json while the bot runs
        with open(self.races_file) as f:
            data = json.load(f)
        out = {r["race_id"]: float(r["mean"]) for r in data["races"]}
        if "house" in data:
            out[data["house"].get("race_id", "HOUSE")] = float(data["house"]["mean"])
        return out

    def push_signal(self, s: Signal) -> None:
        self._signals.append(s)

    def signals(self) -> List[Signal]:
        out, self._signals = self._signals, []
        return out

    def push_reports(self, reports: Dict[str, RaceReport]) -> None:
        """Election night: results parsers call this."""
        self._reports = reports

    def live_reports(self):
        return self._reports

    def call_book(self):
        return self._calls

    def night_params(self):
        path = "data/night_2026.json"
        if not os.path.exists(path):
            return None
        with open(path) as f:
            d = json.load(f)
        return NightParams(**{k: v for k, v in d.items() if not k.startswith("_")})


async def run_live(cfg: BotConfig, live_orders: bool, run_seconds: Optional[float] = None) -> None:
    from .exchange.supermarket import SuperMarketExchange
    mapping = load_market_map(cfg.market_map_file)
    if not mapping:
        raise SystemExit(f"fill in {cfg.market_map_file} first (see config/market_map.example.json)")
    ex = SuperMarketExchange(cfg.venue, allow_orders=live_orders)
    venue_markets = {m.market_id: m.title for m in await ex.get_markets()}
    markets = markets_from_map(mapping, venue_markets)
    missing = [m.market_id for m in markets if m.market_id not in venue_markets]
    if missing:
        print(f"warning: {len(missing)} mapped markets not listed by the venue: {missing[:5]}")
    book = AnchorBook()
    feed = AnchorFeed(mapping, book)
    feeds = LiveFeeds(cfg, book, cfg.races_file)
    model = RaceModel.from_json(cfg.races_file, ModelParams())
    journal = Journal(cfg.journal_path)
    notifier = Notifier.from_env()
    if notifier:
        journal = TeeJournal(journal, notifier)
    engine = Engine(cfg, ex, feeds, model, markets, journal=journal, dry_run=not live_orders)
    if live_orders:
        n = await ex.cancel_all()   # start from a clean book after a crash/restart
        print(f"cancelled {n} leftover resting orders at start")
    stop_cmds = threading.Event()
    if notifier:
        threading.Thread(target=notifier.listen_commands, args=(_command_handler(cfg, engine), stop_cmds),
                         daemon=True).start()
    journal.log("start", {"mode": "LIVE" if live_orders else "dry run", "markets": len(markets)})
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, engine.stop)   # finish the step, then cancel orders below
        except (NotImplementedError, RuntimeError):
            pass
    book.on_change = lambda _mid: engine.poke()   # reprice the instant an anchor moves
    print(f"running {'LIVE' if live_orders else 'DRY-RUN'} on {len(markets)} markets; journal {cfg.journal_path}")
    anchor_task = asyncio.ensure_future(feed.run())
    calls_task = asyncio.ensure_future(_calls_when_due(cfg, feeds, engine))
    warm_task = asyncio.ensure_future(ex.keep_warm()) if live_orders else None
    try:
        await engine.run(stop_after_s=run_seconds)
    finally:
        if live_orders:
            try:  # leave nothing resting on the book when the bot stops
                n = await ex.cancel_all()
                print(f"cancelled {n} resting orders on shutdown")
            except Exception as e:
                print(f"WARNING: cancel on shutdown failed ({e}); run `python -m cupbot cancel-all`")
        anchor_task.cancel()
        calls_task.cancel()
        if warm_task:
            warm_task.cancel()
        stop_cmds.set()
        journal.log("stop", {"equity": round(engine.last_equity)})
        await feed.close()
        await ex.close()
        journal.close()


def _command_handler(cfg: BotConfig, engine: Engine):
    """Phone commands (see notify.py). Runs on the listener thread: only touches files
    and reads engine state; the engine loop itself acts on the KILL file."""
    def handle(cmd: str) -> Optional[str]:
        if cmd == "kill":
            with open(cfg.risk.kill_file, "w") as f:
                f.write(f"phone {time.ctime()}\n")
            engine.poke()
            return "KILL set: cancelling all orders, no new trades"
        if cmd == "resume":
            if os.path.exists(cfg.risk.kill_file):
                os.remove(cfg.risk.kill_file)
            halted = engine.risk.state.halt_reason if engine.risk.state.halted else ""
            return "KILL cleared" + (f"; still halted ({halted}): clear over SSH" if halted else "")
        if cmd == "status":
            a = engine.last_acct
            pos = ", ".join(f"{k} {v:+.0f}" for k, v in sorted((a.positions if a else {}).items())[:15])
            return (f"equity {engine.last_equity:,.0f}  cash {a.cash if a else 0:,.0f}  "
                    f"kill {'ON' if os.path.exists(cfg.risk.kill_file) else 'off'}  "
                    f"halted {engine.risk.state.halted}\n{pos}\nlast error: {engine.last_error or 'none'}")
        return None
    return handle


async def _calls_when_due(cfg: BotConfig, feeds: "LiveFeeds", engine: Engine) -> None:
    """From one hour before polls close: listen for race calls on Bluesky."""
    from .calls import CallBook, CallParser, listen
    wait = cfg.election_ts - 3600 - time.time()
    if wait > 0:
        await asyncio.sleep(wait)
    index = {}
    if os.path.exists("data/anchors_2026.json"):
        with open("data/anchors_2026.json") as f:
            index = json.load(f)
    feeds._calls = CallBook()
    await listen(CallParser(index), feeds._calls, on_call=lambda c: (engine.journal.log(
        "call_seen", {"race": c.race_id, "party": c.party, "source": c.source, "text": c.text[:200]}),
        engine.poke()))


async def measure_mm(cfg: BotConfig, minutes: float, out_csv: str) -> None:
    """Week-one measurement: record Cup top-of-book next to anchors so you can
    see how fast the sponsor's market maker reprices after Kalshi/Polymarket move."""
    from .exchange.supermarket import SuperMarketExchange
    mapping = load_market_map(cfg.market_map_file)
    ex = SuperMarketExchange(cfg.venue, allow_orders=False)
    book = AnchorBook()
    feed = AnchorFeed(mapping, book, poll_s=2.0)
    task = asyncio.ensure_future(feed.run())
    end = time.time() + minutes * 60
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ts", "market_id", "cup_bid", "cup_ask", "anchor"])
        try:
            while time.time() < end:
                books = await ex.get_books(list(mapping))
                anc = book.latest()
                ts = time.time()
                for mid, b in books.items():
                    w.writerow([f"{ts:.3f}", mid, b.best_bid, b.best_ask, anc.get(mid)])
                f.flush()
                await asyncio.sleep(max(1.0, 1.0 / max(cfg.venue.read_rps, 0.1)))
        finally:
            task.cancel()
            await feed.close()
            await ex.close()


def mm_lag_report(csv_path: str, move: float = 0.02) -> str:
    """Median seconds from an anchor move >= `move` to the Cup mid moving the same way."""
    rows: Dict[str, List[tuple]] = {}
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            if not r["cup_bid"] or not r["cup_ask"] or not r["anchor"]:
                continue
            rows.setdefault(r["market_id"], []).append(
                (float(r["ts"]), (float(r["cup_bid"]) + float(r["cup_ask"])) / 2, float(r["anchor"])))
    lags = []
    for mid, xs in rows.items():
        for i in range(1, len(xs)):
            da = xs[i][2] - xs[i - 1][2]
            if abs(da) < move:
                continue
            base = xs[i - 1][1]
            for j in range(i, len(xs)):
                if (xs[j][1] - base) * da > 0 and abs(xs[j][1] - base) >= move / 2:
                    lags.append(xs[j][0] - xs[i][0])
                    break
    if not lags:
        return "no anchor moves large enough to measure yet"
    lags.sort()
    return (f"{len(lags)} anchor moves; Cup repriced after median {lags[len(lags)//2]:.1f}s "
            f"(p25 {lags[len(lags)//4]:.1f}s, p75 {lags[3*len(lags)//4]:.1f}s)")


# ---------------------------------------------------------------- day-one tools
async def draft_market_map(cfg: BotConfig, out_path: str) -> int:
    """List the Cup's markets, guess race/party for each from its title, attach
    Kalshi/Polymarket anchors from data/anchors_2026.json, and write a draft map.
    Never overwrites an existing map: writes <out>.draft instead. Review every line."""
    from .discover import anchors_for, guess_market, kalshi_for_race_id
    from .exchange.supermarket import SuperMarketExchange
    ex = SuperMarketExchange(cfg.venue)
    try:
        markets = await ex.get_markets()
    finally:
        await ex.close()
    index = {}
    if os.path.exists("data/anchors_2026.json"):
        with open("data/anchors_2026.json") as f:
            index = {k: v for k, v in json.load(f).items() if not k.startswith("_")}
    draft, unsure = {}, []
    for m in markets:
        g = guess_market(m.title)
        entry = {"title": m.title, **g}
        if g["race_id"]:
            if g["race_id"] not in index and g["race_id"].startswith(("GOV-", "HOUSE-")):
                found = kalshi_for_race_id(g["race_id"])
                if found:
                    index[g["race_id"]] = found
            entry.update(anchors_for(g["race_id"], g["d_sign"], index))
        others = (index.get(g["race_id"] or "") or {}).get("cand_other") or []
        if others:
            entry["_check"] = (f"three-way race (also {', '.join(others)}): the model's D side means 'not the "
                               "Republican'. Make sure d_sign and anchors match what this market's YES means.")
        if g["kind"] == "other" or g["d_sign"] == 0:
            unsure.append(m.title)
            entry["_check"] = "could not tell race or party from the title: fill in by hand or delete"
        draft[m.market_id] = entry
    path = out_path if not os.path.exists(out_path) else out_path + ".draft"
    with open(path, "w") as f:
        json.dump({"_README": "Draft from `python -m cupbot draft-map`. Check every race_id and d_sign "
                              "(+1: YES = Democratic side, -1: YES = Republican side).", **draft}, f, indent=2)
    print(f"wrote {path}: {len(draft)} markets, {len(unsure)} need a manual look")
    for t in unsure[:20]:
        print("  ?", t)
    return 0


async def preflight(cfg: BotConfig) -> int:
    """Read-only checks before going live. Prints PASS/WARN/FAIL per item."""
    from .anchors import AnchorBook, AnchorFeed
    from .exchange.supermarket import SuperMarketExchange
    results = []

    def rec(level, what, detail=""):
        results.append(level)
        print(f"[{level}] {what}{': ' + detail if detail else ''}")

    rec("PASS" if not os.path.exists(cfg.risk.kill_file) else "WARN", "kill switch",
        "KILL file present, engine will not trade" if os.path.exists(cfg.risk.kill_file) else "clear")
    rec("PASS", "live orders", "ENABLED in config" if cfg.venue.live_trading_enabled else
        "disabled in config (dry run only)")
    try:
        import aiohttp  # noqa: F401
        rec("PASS", "aiohttp installed")
    except ImportError:
        rec("FAIL", "aiohttp missing", "pip3 install --user aiohttp")
        return 1
    if not os.environ.get(cfg.venue.api_key_env):
        rec("FAIL", "API key", f"{cfg.venue.api_key_env} not set (.env)")
        return 1
    ex = SuperMarketExchange(cfg.venue)
    try:
        try:
            who = await ex.whoami()
            rec("PASS", "whoami", str(who)[:120])
        except Exception as e:
            rec("FAIL", "whoami", f"{e} (check venue.paths.whoami against the docs)")
        try:
            ms = await ex.get_markets()
            rec("PASS" if ms else "FAIL", "markets", f"{len(ms)} listed")
        except Exception as e:
            ms = []
            rec("FAIL", "markets", str(e))
        mapping = load_market_map(cfg.market_map_file)
        if not mapping:
            rec("FAIL", "market map", f"{cfg.market_map_file} missing: run draft-map and review it")
        else:
            ids = {m.market_id for m in ms}
            missing = [k for k in mapping if k not in ids]
            unmapped = [m.title for m in ms if m.market_id not in mapping]
            rec("PASS" if not missing else "FAIL", "mapped markets exist", f"{len(missing)} missing {missing[:3]}")
            rec("PASS" if not unmapped else "WARN", "Cup markets not in map", f"{len(unmapped)} {unmapped[:3]}")
            bad = [k for k, v in mapping.items() if v.get("d_sign") not in (1, -1) or not v.get("race_id")]
            rec("PASS" if not bad else "FAIL", "map entries have race_id and d_sign", f"{len(bad)} bad {bad[:3]}")
            try:
                books = await ex.get_books(list(mapping)[:100])
                px = [p for b in books.values() for p in (b.best_bid, b.best_ask) if p is not None]
                ok = px and all(0 < p < 1 for p in px)
                rec("PASS" if ok else "FAIL", "prices parse into (0, 1)",
                    f"{len(books)} books, sample {px[:4]} (check venue.price_scale)")
            except Exception as e:
                rec("FAIL", "prices", f"{e} (check venue.paths.prices and _parse_bbo)")
            book = AnchorBook()
            feed = AnchorFeed(mapping, book)
            try:
                await feed.poll_kalshi_once()
                await feed.poll_polymarket_once()
            finally:
                await feed.close()
            have = book.latest()
            rec("PASS" if have else "WARN", "anchors", f"{len(have)} of {len(mapping)} mapped markets have a price")
        try:
            acct = await ex.get_account()
            rec("PASS", "account", f"cash {acct.cash:,.0f}, {len(acct.positions)} positions")
        except Exception as e:
            rec("FAIL", "account", f"{e} (check venue.paths.balance / positions)")
        rec("PASS" if ex.last_rate_headers else "WARN", "rate-limit headers",
            str(ex.last_rate_headers) if ex.last_rate_headers else "none seen: take limits from the docs")
    finally:
        await ex.close()
    fails = results.count("FAIL")
    print(f"\n{fails} FAIL, {results.count('WARN')} WARN")
    return 1 if fails else 0
