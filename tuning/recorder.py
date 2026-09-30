"""Live recorder for Kalshi + Polymarket midterm books and trades.

Writes one JSON object per line to data/live/<venue>-<UTCdate>.jsonl with a
local receive timestamp (recv_ns) on every record, so analyze.py can measure
move sizes, settle-back and cross-venue lead/lag, and latency_probe.py can
measure feed lag.

Stdlib only (Python 3.9+). Runs on Johnny's Mac, a us-east-1 VM, or this
workspace once the hosts are allowed.

  python3 recorder.py --config markets.json --minutes 0   # 0 = run forever

Sources
  Polymarket: public market websocket (book + price_change + last_trade_price).
  Kalshi: websocket if KALSHI_KEY_ID + KALSHI_PRIVATE_KEY_PATH are set and the
          `cryptography` package is installed; otherwise public REST orderbook
          polling over one warm keep-alive connection (default 1 req/s/market,
          staggered).
"""
import argparse
import asyncio
import datetime as dt
import http.client
import json
import os
import ssl
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wsmini  # noqa: E402

KALSHI_HOST = "api.elections.kalshi.com"
KALSHI_BASE = "/trade-api/v2"
KALSHI_WS = "wss://api.elections.kalshi.com/trade-api/ws/v2"
POLY_WS = "wss://ws-subscriptions-clob.polymarket.com/ws/market"

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "live")


class Sink:
    """Buffered JSONL writer, flushed off the hot path every second."""

    def __init__(self, venue):
        self.venue = venue
        self.buf = []
        os.makedirs(OUT_DIR, exist_ok=True)

    def put(self, rec):
        self.buf.append(rec)

    def flush(self):
        if not self.buf:
            return
        day = dt.datetime.utcnow().strftime("%Y%m%d")
        with open(os.path.join(OUT_DIR, f"{self.venue}-{day}.jsonl"), "a") as f:
            for r in self.buf:
                f.write(json.dumps(r, separators=(",", ":")) + "\n")
        self.buf.clear()


# ---------------------------------------------------------------- Kalshi REST
class KalshiRest:
    """One persistent HTTPS connection (TCP_NODELAY, keep-alive)."""

    def __init__(self):
        self.conn = None

    def _connect(self):
        ctx = ssl.create_default_context(cafile=os.environ.get("SSL_CERT_FILE") or None)
        proxy = os.environ.get("HTTPS_PROXY")
        if proxy:
            from urllib.parse import urlparse
            p = urlparse(proxy)
            c = http.client.HTTPSConnection(p.hostname, p.port, context=ctx, timeout=10)
            c.set_tunnel(KALSHI_HOST, 443)
        else:
            c = http.client.HTTPSConnection(KALSHI_HOST, 443, context=ctx, timeout=10)
        c.connect()
        import socket
        c.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.conn = c

    def get(self, path):
        for attempt in range(2):
            try:
                if self.conn is None:
                    self._connect()
                t0 = time.time_ns()
                self.conn.request("GET", KALSHI_BASE + path, headers={"Accept": "application/json", "Connection": "keep-alive"})
                resp = self.conn.getresponse()
                body = resp.read()
                t1 = time.time_ns()
                if resp.status != 200:
                    raise RuntimeError(f"HTTP {resp.status}: {body[:200]!r}")
                return json.loads(body), t0, t1
            except (http.client.HTTPException, OSError):
                self.conn = None
                if attempt:
                    raise
        raise RuntimeError("unreachable")


def parse_kalshi_book(ob):
    """Normalise Kalshi orderbook (cents or *_dollars variants) to dollars.
    Kalshi returns YES bids and NO bids; YES ask = 1 - best NO bid."""
    ob = ob.get("orderbook_fp") or ob.get("orderbook") or ob
    def lv(key_d, key_c):
        if ob.get(key_d):
            return [(float(p), float(q)) for p, q in ob[key_d]]
        if ob.get(key_c):
            return [(p / 100.0, float(q)) for p, q in ob[key_c]]
        return []
    yes = sorted(lv("yes_dollars", "yes"), reverse=True)
    no = sorted(lv("no_dollars", "no"), reverse=True)
    bid = yes[0][0] if yes else None
    ask = round(1 - no[0][0], 4) if no else None
    return {"bid": bid, "ask": ask, "bids": yes[:10], "asks": [(round(1 - p, 4), q) for p, q in no[:10]]}


async def kalshi_rest_loop(tickers, hz, sink, stop):
    cli = KalshiRest()
    loop = asyncio.get_running_loop()
    period = 1.0 / hz
    last_trade_ts = {}
    i = 0
    while not stop.is_set():
        t_start = time.monotonic()
        for tk in tickers:
            try:
                ob, t0, t1 = await loop.run_in_executor(None, cli.get, f"/markets/{tk}/orderbook?depth=10")
                b = parse_kalshi_book(ob)
                sink.put({"v": "kalshi", "k": "book", "m": tk, "send_ns": t0, "recv_ns": t1, **b})
                if i % 5 == 0:  # trades every 5th sweep
                    tr, t0, t1 = await loop.run_in_executor(None, cli.get, f"/markets/trades?ticker={tk}&limit=100")
                    for t in reversed(tr.get("trades", [])):
                        ts = t.get("created_time")
                        if ts and ts > last_trade_ts.get(tk, ""):
                            last_trade_ts[tk] = ts
                            sink.put({"v": "kalshi", "k": "trade", "m": tk, "recv_ns": t1, "ts": ts,
                                      "p": float(t.get("yes_price_dollars") or t.get("yes_price", 0) / 100),
                                      "q": float(t.get("count_fp") or t.get("count", 0)), "side": t.get("taker_side")})
            except Exception as e:  # keep recording other markets
                sink.put({"v": "kalshi", "k": "error", "m": tk, "recv_ns": time.time_ns(), "err": str(e)[:200]})
            await asyncio.sleep(period / max(1, len(tickers)))
        i += 1
        rem = period - (time.monotonic() - t_start)
        if rem > 0:
            await asyncio.sleep(rem)


# ------------------------------------------------------------- Kalshi WS
def kalshi_ws_headers():
    key_id = os.environ.get("KALSHI_KEY_ID")
    key_path = os.environ.get("KALSHI_PRIVATE_KEY_PATH")
    if not (key_id and key_path):
        return None
    try:
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import padding
    except ImportError:
        print("kalshi ws: `cryptography` not installed, using REST polling", file=sys.stderr)
        return None
    with open(key_path, "rb") as f:
        pk = serialization.load_pem_private_key(f.read(), password=None)
    ts = str(int(time.time() * 1000))
    msg = (ts + "GET" + "/trade-api/ws/v2").encode()
    sig = pk.sign(msg, padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH), hashes.SHA256())
    import base64
    return {"KALSHI-ACCESS-KEY": key_id, "KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig).decode(), "KALSHI-ACCESS-TIMESTAMP": ts}


async def kalshi_ws_loop(tickers, sink, stop):
    while not stop.is_set():
        hdr = kalshi_ws_headers()
        if hdr is None:
            return False
        try:
            ws = await wsmini.connect(KALSHI_WS, headers=hdr)
            await ws.send(json.dumps({"id": 1, "cmd": "subscribe", "params": {
                "channels": ["orderbook_delta", "trade", "ticker"], "market_tickers": tickers}}))
            while not stop.is_set():
                raw, t_ns = await ws.recv()
                sink.put({"v": "kalshi", "k": "ws", "recv_ns": t_ns, "raw": json.loads(raw)})
        except Exception as e:
            sink.put({"v": "kalshi", "k": "error", "recv_ns": time.time_ns(), "err": str(e)[:200]})
            await asyncio.sleep(2)
    return True


# ------------------------------------------------------------- Polymarket WS
async def poly_ws_loop(token_ids, sink, stop):
    backoff = 1
    while not stop.is_set():
        try:
            ws = await wsmini.connect(POLY_WS)
            await ws.send(json.dumps({"assets_ids": token_ids, "type": "market"}))
            backoff = 1

            async def pinger():
                while not ws.closed:
                    await asyncio.sleep(10)
                    await ws.send("PING")
            pt = asyncio.ensure_future(pinger())
            try:
                while not stop.is_set():
                    raw, t_ns = await ws.recv()
                    if raw == "PONG":
                        continue
                    msg = json.loads(raw)
                    for m in (msg if isinstance(msg, list) else [msg]):
                        sink.put({"v": "poly", "k": m.get("event_type", "?"), "recv_ns": t_ns, "raw": m})
            finally:
                pt.cancel()
        except Exception as e:
            sink.put({"v": "poly", "k": "error", "recv_ns": time.time_ns(), "err": str(e)[:200]})
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30)


async def flusher(sinks, stop):
    while not stop.is_set():
        await asyncio.sleep(1)
        for s in sinks:
            s.flush()


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "markets.json"))
    ap.add_argument("--minutes", type=float, default=0, help="0 = forever")
    ap.add_argument("--kalshi-hz", type=float, default=1.0, help="REST sweeps per second")
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    kt = [m["kalshi"] for m in cfg["markets"] if m.get("kalshi")]
    pt = [m["poly_token"] for m in cfg["markets"] if m.get("poly_token")]
    stop = asyncio.Event()
    ks, ps = Sink("kalshi"), Sink("poly")
    tasks = [asyncio.ensure_future(flusher([ks, ps], stop))]
    if pt:
        tasks.append(asyncio.ensure_future(poly_ws_loop(pt, ps, stop)))
    if kt:
        async def kalshi():
            if not await kalshi_ws_loop(kt, ks, stop):
                await kalshi_rest_loop(kt, a.kalshi_hz, ks, stop)
        tasks.append(asyncio.ensure_future(kalshi()))
    if a.minutes:
        await asyncio.sleep(a.minutes * 60)
        stop.set()
        await asyncio.sleep(1.5)
        for t in tasks:
            t.cancel()
    else:
        await asyncio.gather(*tasks)
    ks.flush()
    ps.flush()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
