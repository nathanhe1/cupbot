"""Measure where the milliseconds go, from wherever this runs.

  python3 latency_probe.py                 # Kalshi + Polymarket REST, Polymarket WS
  python3 latency_probe.py --n 50 --ws-seconds 60
  python3 latency_probe.py --cup-url https://sig.thesuper.market/api/v1/<a docs-listed GET> --cup-key $CUP_API_KEY

Reports, per host: DNS, TCP connect, TLS handshake, cold first request
(new connection) vs warm keep-alive requests (median / p90), and the
serving region headers (x-vercel-id, cf-ray, x-amz-cf-pop). For the
Polymarket websocket: connect+subscribe time, time to first book, and the
exchange-timestamp -> local-receive lag (includes clock offset; run with
NTP synced and compare locations, the offset cancels in differences).
Also times JSON decoding of a real book with json vs orjson (if installed).

Stdlib only. Nothing here sends an order. The Cup probe is off unless you
pass --cup-url, and only issues the GETs you point it at, with your key,
through the official API (never the website).
"""
import argparse
import asyncio
import http.client
import json
import os
import socket
import ssl
import statistics
import sys
import time
from urllib.parse import urlparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wsmini  # noqa: E402

TARGETS = [
    ("kalshi", "https://api.elections.kalshi.com/trade-api/v2/markets/CONTROLS-2026-D/orderbook?depth=10"),
    ("poly-clob", "https://clob.polymarket.com/time"),
    ("poly-gamma", "https://gamma-api.polymarket.com/markets?limit=1"),
]
REGION_HEADERS = ("x-vercel-id", "cf-ray", "x-amz-cf-pop", "x-amz-cf-id", "server", "via")


def ms(ns):
    return round(ns / 1e6, 2)


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def ctx():
    return ssl.create_default_context(cafile=os.environ.get("SSL_CERT_FILE") or None)


def phase_timings(url):
    """DNS / TCP / TLS / TTFB for one cold request, done by hand so each
    phase is visible. Direct connection only (a proxy hides these phases)."""
    u = urlparse(url)
    t0 = time.perf_counter_ns()
    addr = socket.getaddrinfo(u.hostname, 443, proto=socket.IPPROTO_TCP)[0][4]
    t1 = time.perf_counter_ns()
    s = socket.create_connection(addr[:2], timeout=10)
    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    t2 = time.perf_counter_ns()
    ss = ctx().wrap_socket(s, server_hostname=u.hostname)
    t3 = time.perf_counter_ns()
    path = u.path + ("?" + u.query if u.query else "")
    ss.sendall(f"GET {path} HTTP/1.1\r\nHost: {u.hostname}\r\nAccept: application/json\r\nConnection: close\r\n\r\n".encode())
    ss.recv(1)
    t4 = time.perf_counter_ns()
    ss.close()
    return {"ip": addr[0], "tls": ss.version() if hasattr(ss, "version") else None,
            "dns_ms": ms(t1 - t0), "tcp_ms": ms(t2 - t1), "tls_ms": ms(t3 - t2), "ttfb_ms": ms(t4 - t3)}


def conn_for(u, headers_out):
    proxy = os.environ.get("HTTPS_PROXY")
    if proxy:
        p = urlparse(proxy)
        c = http.client.HTTPSConnection(p.hostname, p.port, context=ctx(), timeout=10)
        c.set_tunnel(u.hostname, 443)
    else:
        c = http.client.HTTPSConnection(u.hostname, 443, context=ctx(), timeout=10)
    return c


def http_probe(url, n, extra_headers=None):
    u = urlparse(url)
    path = u.path + ("?" + u.query if u.query else "")
    hdrs = {"Accept": "application/json", **(extra_headers or {})}
    cold = []
    region = {}
    for _ in range(min(5, n)):
        c = conn_for(u, None)
        t0 = time.perf_counter_ns()
        c.request("GET", path, headers=hdrs)
        r = c.getresponse()
        body = r.read()
        cold.append(time.perf_counter_ns() - t0)
        region = {h: r.getheader(h) for h in REGION_HEADERS if r.getheader(h)}
        status = r.status
        c.close()
    c = conn_for(u, None)
    c.connect()
    try:
        c.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    except OSError:
        pass
    warm = []
    for i in range(n + 1):
        t0 = time.perf_counter_ns()
        c.request("GET", path, headers=hdrs)
        r = c.getresponse()
        body = r.read()
        if i:  # first one on the fresh connection is not warm
            warm.append(time.perf_counter_ns() - t0)
    c.close()
    return {"status": status, "cold_med_ms": ms(statistics.median(cold)), "warm_med_ms": ms(statistics.median(warm)),
            "warm_p90_ms": ms(pct(warm, 0.9)), "warm_min_ms": ms(min(warm)), "region": region, "sample": body}


def json_bench(body, reps=2000):
    out = {}
    t0 = time.perf_counter_ns()
    for _ in range(reps):
        json.loads(body)
    out["json_us"] = round((time.perf_counter_ns() - t0) / reps / 1e3, 1)
    try:
        import orjson
        t0 = time.perf_counter_ns()
        for _ in range(reps):
            orjson.loads(body)
        out["orjson_us"] = round((time.perf_counter_ns() - t0) / reps / 1e3, 1)
    except ImportError:
        out["orjson_us"] = "not installed"
    out["bytes"] = len(body)
    return out


async def poly_ws_probe(token_ids, seconds):
    t0 = time.perf_counter_ns()
    ws = await wsmini.connect("wss://ws-subscriptions-clob.polymarket.com/ws/market")
    t1 = time.perf_counter_ns()
    await ws.send(json.dumps({"assets_ids": token_ids, "type": "market"}))
    first = None
    lags, gaps, n = [], [], 0
    last = None
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        try:
            raw, t_ns = await asyncio.wait_for(ws.recv(), timeout=max(0.1, end - time.monotonic()))
        except asyncio.TimeoutError:
            break
        if raw in ("PONG",):
            continue
        n += 1
        if first is None:
            first = time.perf_counter_ns()
        msg = json.loads(raw)
        for m in (msg if isinstance(msg, list) else [msg]):
            ts = m.get("timestamp")
            if ts:
                lags.append(t_ns / 1e6 - float(ts))
        if last:
            gaps.append((t_ns - last) / 1e6)
        last = t_ns
    await ws.close()
    res = {"connect_ms": ms(t1 - t0), "first_msg_ms": ms(first - t1) if first else None, "messages": n}
    if lags:
        res.update({"exch_to_recv_med_ms": round(statistics.median(lags), 1), "exch_to_recv_p90_ms": round(pct(lags, 0.9), 1),
                    "note": "includes local clock offset; compare across locations"})
    if gaps:
        res["msg_gap_med_ms"] = round(statistics.median(gaps), 1)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--ws-seconds", type=float, default=30)
    ap.add_argument("--poly-tokens", default="", help="comma list; default reads markets.json")
    ap.add_argument("--cup-url", default="")
    ap.add_argument("--cup-key", default=os.environ.get("CUP_API_KEY", ""))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "latency.jsonl"))
    a = ap.parse_args()
    report = {"at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "host": socket.gethostname(),
              "proxy": bool(os.environ.get("HTTPS_PROXY")), "targets": {}}
    targets = list(TARGETS)
    if a.cup_url:
        targets.append(("cup", a.cup_url))
    sample = None
    for name, url in targets:
        entry = {}
        try:
            if not os.environ.get("HTTPS_PROXY"):
                entry["phases"] = phase_timings(url)
            hdr = {"Authorization": f"Bearer {a.cup_key}"} if name == "cup" and a.cup_key else None
            h = http_probe(url, a.n, hdr)
            if name == "kalshi":
                sample = h["sample"]
            h.pop("sample")
            entry.update(h)
        except Exception as e:
            entry["error"] = str(e)[:200]
        report["targets"][name] = entry
        print(name, json.dumps(entry))
    if sample:
        report["json_decode"] = json_bench(sample)
        print("json", report["json_decode"])
    toks = [t for t in a.poly_tokens.split(",") if t]
    if not toks:
        try:
            cfg = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "markets.json")))
            toks = [m["poly_token"] for m in cfg["markets"] if m.get("poly_token")]
        except Exception:
            toks = []
    if toks:
        try:
            report["poly_ws"] = asyncio.run(poly_ws_probe(toks, a.ws_seconds))
        except Exception as e:
            report["poly_ws"] = {"error": str(e)[:200]}
        print("poly_ws", report["poly_ws"])
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "a") as f:
        f.write(json.dumps(report) + "\n")


if __name__ == "__main__":
    main()
