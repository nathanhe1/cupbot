"""Local micro-benchmarks for the order path (no network beyond loopback).

  python3 bench_local.py

Measures, with a local TLS server that adds a configurable fake RTT:
  1. new TLS connection per order vs one warm keep-alive connection
  2. json.dumps(order) + header build per order vs a pre-built payload template
  3. the in-memory decision (reference tick -> threshold compare -> payload)
  4. HTTP/1.1 head-of-line blocking: an order queued behind a slow read on one
     connection vs a dedicated order connection
Results go to results/bench_local.json. Loopback numbers are lower bounds; the
point is the relative cost of each step.
"""
import http.client
import json
import os
import socket
import ssl
import statistics
import subprocess
import tempfile
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
RTT_S = float(os.environ.get("FAKE_RTT_MS", "10")) / 1000  # added per request/handshake leg


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    disable_nagle_algorithm = True

    def log_message(self, *a):
        pass

    def _reply(self):
        n = int(self.headers.get("Content-Length") or 0)
        if n:
            self.rfile.read(n)
        if self.path.startswith("/slow"):
            time.sleep(0.2)
        time.sleep(RTT_S)
        body = b'{"ok":true}'
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    do_GET = do_POST = _reply


def make_server():
    d = tempfile.mkdtemp()
    crt, key = os.path.join(d, "c.pem"), os.path.join(d, "k.pem")
    subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-keyout", key, "-out", crt,
                    "-days", "1", "-subj", "/CN=localhost"], check=True, capture_output=True)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(crt, key)

    # emulate RTT on handshake: TLS1.3 = 1 extra RTT, TCP = 1 RTT
    orig = srv.get_request

    def get_request():
        s, a = orig()
        time.sleep(2 * RTT_S)
        return ctx.wrap_socket(s, server_side=True), a
    srv.get_request = get_request
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    cctx = ssl.create_default_context(cafile=crt)
    cctx.check_hostname = False
    return srv, cctx


def med(xs):
    return round(statistics.median(xs) * 1e3, 3)


def main():
    srv, cctx = make_server()
    port = srv.server_address[1]
    res = {"fake_rtt_ms": RTT_S * 1e3}
    order = {"market_id": "SEN-GA-D", "side": "buy", "outcome": "yes", "price": 0.62, "size": 1500, "type": "limit",
             "client_order_id": ""}
    auth = "Bearer " + "x" * 40

    # 1. cold vs warm
    cold = []
    for _ in range(10):
        t0 = time.perf_counter()
        c = http.client.HTTPSConnection("127.0.0.1", port, context=cctx)
        c.request("POST", "/orders", body=b"{}", headers={"Content-Type": "application/json"})
        c.getresponse().read()
        cold.append(time.perf_counter() - t0)
        c.close()
    c = http.client.HTTPSConnection("127.0.0.1", port, context=cctx)
    c.connect()
    c.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    warm = []
    for _ in range(30):
        t0 = time.perf_counter()
        c.request("POST", "/orders", body=b"{}", headers={"Content-Type": "application/json"})
        c.getresponse().read()
        warm.append(time.perf_counter() - t0)
    res["order_cold_ms"], res["order_warm_ms"] = med(cold), med(warm)

    # 2. payload build
    N = 20000
    t0 = time.perf_counter()
    for i in range(N):
        o = dict(order, price=0.62, size=1500, client_order_id=str(uuid.uuid4()))
        b = json.dumps(o).encode()
        h = {"Content-Type": "application/json", "Authorization": auth, "Content-Length": str(len(b))}
    res["build_json_us"] = round((time.perf_counter() - t0) / N * 1e6, 2)
    tmpl = {}
    for side in ("buy", "sell"):
        for outc in ("yes", "no"):
            s = json.dumps(dict(order, side=side, outcome=outc, price=0, size=0, client_order_id="")).encode()
            tmpl[(side, outc)] = s.replace(b'"price": 0', b'"price": %s').replace(b'"size": 0', b'"size": %d').replace(b'"client_order_id": ""', b'"client_order_id": "%s"')
    ids = [uuid.uuid4().hex.encode() for _ in range(1000)]  # pre-generated ids
    t0 = time.perf_counter()
    for i in range(N):
        b = tmpl[("buy", "yes")] % (b"0.62", 1500, ids[i % 1000])
    res["build_template_us"] = round((time.perf_counter() - t0) / N * 1e6, 2)
    json.loads(tmpl[("buy", "yes")] % (b"0.62", 1500, ids[0]))  # template must stay valid JSON

    # 3. decision path: tick -> compare against precomputed trigger -> payload
    triggers = {"KX-SEN-GA": (0.645, ("buy", "yes"), b"0.63", 1500)}
    tick = {"m": "KX-SEN-GA", "bid": 0.65, "ask": 0.66}
    t0 = time.perf_counter()
    for i in range(N):
        tr = triggers.get(tick["m"])
        if tr and (tick["bid"] + tick["ask"]) * 0.5 >= tr[0]:
            b = tmpl[tr[1]] % (tr[2], tr[3], ids[i % 1000])
    res["decision_us"] = round((time.perf_counter() - t0) / N * 1e6, 2)
    raw = json.dumps({"event_type": "price_change", "asset_id": "1" * 70, "price_changes": [{"price": "0.62", "size": "1200", "side": "BUY"}] * 3, "timestamp": "1790000000000"})
    t0 = time.perf_counter()
    for _ in range(N):
        json.loads(raw)
    res["parse_ws_msg_us"] = round((time.perf_counter() - t0) / N * 1e6, 2)

    # 4. head-of-line blocking
    def order_after_slow(shared):
        conn = http.client.HTTPSConnection("127.0.0.1", port, context=cctx)
        conn.connect()
        oconn = conn if shared else http.client.HTTPSConnection("127.0.0.1", port, context=cctx)
        if not shared:
            oconn.connect()
        lock = threading.Lock()
        out = {}

        def slow():
            with lock:
                conn.request("GET", "/slow")
                conn.getresponse().read()
        th = threading.Thread(target=slow)
        th.start()
        time.sleep(0.01)
        t0 = time.perf_counter()
        if shared:
            with lock:
                conn.request("POST", "/orders", body=b"{}")
                conn.getresponse().read()
        else:
            oconn.request("POST", "/orders", body=b"{}")
            oconn.getresponse().read()
        out["t"] = time.perf_counter() - t0
        th.join()
        return out["t"]
    res["order_behind_slow_read_shared_conn_ms"] = med([order_after_slow(True) for _ in range(5)])
    res["order_dedicated_conn_ms"] = med([order_after_slow(False) for _ in range(5)])
    srv.shutdown()
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(res, open(os.path.join(HERE, "results", "bench_local.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
