"""Tick recorder for the anchor venues (Kalshi + Polymarket), stdlib only.

    python -m cupbot record                 # poll every 2 s, forever
    python -m cupbot record --interval 1 --minutes 30

Writes journal/ticks/YYYY-MM-DD.csv with one row per top-of-book CHANGE:
    recv_ts, venue, race_id, side (D|R), bid, ask, bid_size, ask_size, rtt_ms
Each venue is fetched with one batched request per poll, so this costs Kalshi
~1 request per interval and Polymarket ~1. Public market data only.
Analyse it (with the websocket recorder's richer data) using tuning/lab.py."""

from __future__ import annotations

import csv
import json
import os
import time
import urllib.request
from typing import Dict, List, Optional, Tuple

KALSHI = "https://api.elections.kalshi.com/trade-api/v2/markets"
POLY_BOOKS = "https://clob.polymarket.com/books"


def load_targets(index_path: str = "data/anchors_2026.json"):
    """-> (kalshi {ticker: (race_id, side)}, poly {token: (race_id, side)})."""
    with open(index_path) as f:
        idx = json.load(f)
    k, p = {}, {}
    for rid, e in idx.items():
        if rid.startswith("_") or not isinstance(e, dict):
            continue
        for key, side in (("kalshi_d", "D"), ("kalshi_r", "R")):
            if e.get(key):
                k[e[key]] = (rid, side)
        for key, side in (("poly_d_token", "D"), ("poly_r_token", "R")):
            if e.get(key):
                p[e[key]] = (rid, side)
    return k, p


def _http(url: str, data: Optional[bytes] = None, timeout: float = 5.0):
    req = urllib.request.Request(url, data=data, headers={
        "Accept": "application/json", "User-Agent": "cupbot-recorder/0.1",
        **({"Content-Type": "application/json"} if data else {})})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = json.loads(r.read().decode())
    return body, (time.perf_counter() - t0) * 1000


def poll_kalshi(tickers: Dict[str, Tuple[str, str]]) -> Tuple[List[tuple], float]:
    rows, rtt = [], 0.0
    ts = list(tickers)
    for i in range(0, len(ts), 50):
        body, dt = _http(KALSHI + "?tickers=" + ",".join(ts[i:i + 50]))
        rtt = max(rtt, dt)
        for m in body.get("markets", []):
            rid, side = tickers.get(m["ticker"], (None, None))
            if rid is None:
                continue
            rows.append(("kalshi", rid, side, _f(m.get("yes_bid_dollars")), _f(m.get("yes_ask_dollars")),
                         _f(m.get("yes_bid_size_fp") or m.get("yes_bid_size")),
                         _f(m.get("yes_ask_size_fp") or m.get("yes_ask_size"))))
    return rows, rtt


def poll_poly(tokens: Dict[str, Tuple[str, str]]) -> Tuple[List[tuple], float]:
    if not tokens:
        return [], 0.0
    body, rtt = _http(POLY_BOOKS, json.dumps([{"token_id": t} for t in tokens]).encode())
    rows = []
    for b in body:
        rid, side = tokens.get(b.get("asset_id"), (None, None))
        if rid is None:
            continue
        bids = [(float(x["price"]), float(x["size"])) for x in b.get("bids", [])]
        asks = [(float(x["price"]), float(x["size"])) for x in b.get("asks", [])]
        bb = max(bids) if bids else (None, None)
        ba = min(asks) if asks else (None, None)
        rows.append(("polymarket", rid, side, bb[0], ba[0], bb[1], ba[1]))
    return rows, rtt


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def record(interval_s: float = 2.0, minutes: Optional[float] = None, out_dir: str = "journal/ticks") -> None:
    os.makedirs(out_dir, exist_ok=True)
    kt, pt = load_targets()
    last: Dict[tuple, tuple] = {}
    end = time.time() + minutes * 60 if minutes else None
    with open(os.path.join(out_dir, "recorder.pid"), "w") as f:
        f.write(str(os.getpid()))
    print(f"recording {len(kt)} Kalshi and {len(pt)} Polymarket books every {interval_s}s -> {out_dir}/")
    while end is None or time.time() < end:
        t0 = time.time()
        path = os.path.join(out_dir, time.strftime("%Y-%m-%d") + ".csv")
        new = not os.path.exists(path)
        with open(path, "a", newline="") as fh:
            w = csv.writer(fh)
            if new:
                w.writerow(["recv_ts", "venue", "race_id", "side", "bid", "ask", "bid_size", "ask_size", "rtt_ms"])
            for poll, targets in ((poll_kalshi, kt), (poll_poly, pt)):
                try:
                    rows, rtt = poll(targets)
                except Exception as e:  # keep recording through transient errors
                    w.writerow([f"{time.time():.3f}", "error", "", "", "", "", "", "", repr(e)[:80]])
                    continue
                now = time.time()
                for venue, rid, side, bid, ask, bq, aq in rows:
                    key = (venue, rid, side)
                    val = (bid, ask)
                    if last.get(key) != val:
                        last[key] = val
                        w.writerow([f"{now:.3f}", venue, rid, side, bid, ask, bq, aq, f"{rtt:.0f}"])
        time.sleep(max(0.0, interval_s - (time.time() - t0)))
