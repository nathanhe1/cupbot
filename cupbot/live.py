"""Live wiring: feeds from the real world, and the live run loop.

Default is DRY RUN: the engine computes everything and journals the orders it
would send, but sends nothing. Real orders need BOTH venue.live_trading_enabled
= true in the config file AND the --live flag."""

from __future__ import annotations

import asyncio
import csv
import json
import os
import time
from typing import Dict, List, Optional

from .anchors import AnchorBook, AnchorFeed
from .config import BotConfig
from .election_night import NightParams, RaceReport
from .engine import Engine
from .feeds import Feeds
from .journal import Journal
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
    engine = Engine(cfg, ex, feeds, model, markets, journal=journal, dry_run=not live_orders)
    book.on_change = lambda _mid: engine.poke()   # reprice the instant an anchor moves
    print(f"running {'LIVE' if live_orders else 'DRY-RUN'} on {len(markets)} markets; journal {cfg.journal_path}")
    anchor_task = asyncio.ensure_future(feed.run())
    try:
        await engine.run(stop_after_s=run_seconds)
    finally:
        anchor_task.cancel()
        await feed.close()
        await ex.close()
        journal.close()


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
