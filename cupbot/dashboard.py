"""Local live dashboard: `python3 -m cupbot dashboard`, then open http://127.0.0.1:8790.

Reads the bot's journal (SQLite), its pid file, the kill switch and the recorders'
output files, so it adds no load to the bot's own API budget. With --account it
also reads the Cup balance and positions every 60 s (2 of the account's 100
reads a minute, shared with the bot), which is off by default. It can switch the kill switch on and off, and has
no way to place an order. Listens on localhost only."""

from __future__ import annotations

import asyncio
import glob
import json
import os
import sqlite3
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional

from .config import BotConfig

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PID_FILE = "journal/live.pid"
ORDER_KINDS = ("order", "dry_order", "blocked")
ALERT_KINDS = ("start", "stop", "halt", "kill", "error", "basket", "unknown_resolved")


def _alive(pid_file: str) -> Optional[int]:
    try:
        pid = int(open(pid_file).read().strip())
        os.kill(pid, 0)
        return pid
    except (OSError, ValueError):
        return None


def _newest(pattern: str) -> Optional[Dict[str, Any]]:
    files = glob.glob(pattern)
    if not files:
        return None
    f = max(files, key=os.path.getmtime)
    return {"file": os.path.relpath(f, ROOT), "age_s": time.time() - os.path.getmtime(f),
            "mb": os.path.getsize(f) / 1e6}


class Account:
    """Cup balance and positions, refreshed in the background every `every_s`."""

    def __init__(self, cfg: BotConfig, every_s: float = 60.0):
        self.cfg, self.every_s = cfg, every_s
        self.data: Dict[str, Any] = {"ok": False, "error": "not read yet"}
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self) -> None:
        while True:
            try:
                self.data = asyncio.run(self._read())
            except Exception as e:  # keep the last good values visible
                self.data = {**self.data, "ok": False, "error": repr(e)[:200]}
            time.sleep(self.every_s)

    async def _read(self) -> Dict[str, Any]:
        from .exchange.supermarket import SuperMarketExchange
        ex = SuperMarketExchange(self.cfg.venue)
        try:
            t = await ex._request("GET", "balance")
            p = await ex._request("GET", "positions")
        finally:
            await ex.close()
        return {"ok": True, "ts": time.time(), "balance": t.get("myBalance"), "status": t.get("status"),
                "initial": t.get("initialBalance"), "positions": p.get("positions", []),
                "summary": p.get("summary", {})}


class State:
    def __init__(self, cfg: BotConfig, account: Optional[Account]):
        self.cfg, self.account = cfg, account
        self.titles: Dict[str, str] = {}
        try:
            with open(cfg.market_map_file) as f:
                self.titles = {k: v.get("title", k) for k, v in json.load(f).items() if not k.startswith("_")}
        except (OSError, ValueError):
            pass

    def snapshot(self) -> Dict[str, Any]:
        now = time.time()
        out: Dict[str, Any] = {
            "now": now,
            "pid": _alive(PID_FILE),
            "kill": os.path.exists(self.cfg.risk.kill_file),
            "live_enabled": self.cfg.venue.live_trading_enabled,
            "open_ts": self.cfg.open_ts,
            "recorders": {"tuning": _newest("tuning/data/live/*.jsonl"), "ticks": _newest("journal/ticks/*")},
            "account": self.account.data if self.account else None,
        }
        if not os.path.exists(self.cfg.journal_path):
            out["journal"] = None
            return out
        db = sqlite3.connect(f"file:{self.cfg.journal_path}?mode=ro", uri=True, timeout=2)
        try:
            last = {}
            for kind in ALERT_KINDS:
                r = db.execute("SELECT ts, data FROM events WHERE kind=? ORDER BY ts DESC LIMIT 1", (kind,)).fetchone()
                if r:
                    last[kind] = {"ts": r[0], "data": _loads(r[1])}
            counts = dict(db.execute("SELECT kind, COUNT(*) FROM events WHERE ts > ? GROUP BY kind",
                                     (now - 86400,)).fetchall())
            # orders per minute over the last hour, by kind
            per_min: Dict[str, list] = {k: [0] * 60 for k in ORDER_KINDS}
            for ts, kind in db.execute("SELECT ts, kind FROM events WHERE ts > ? AND kind IN (?,?,?)",
                                       (now - 3600, *ORDER_KINDS)):
                i = min(59, int((ts - (now - 3600)) // 60))
                per_min[kind][i] += 1
            recent = []
            for ts, kind, data in db.execute(
                    "SELECT ts, kind, data FROM events WHERE kind IN (?,?,?) ORDER BY ts DESC LIMIT 60", ORDER_KINDS):
                d = _loads(data)
                d["title"] = self.titles.get(str(d.get("market")), str(d.get("market")))
                recent.append({"ts": ts, "kind": kind, **d})
            alerts = []
            for ts, kind, data in db.execute(
                    f"SELECT ts, kind, data FROM events WHERE kind IN ({','.join('?' * len(ALERT_KINDS))}) "
                    "ORDER BY ts DESC LIMIT 20", ALERT_KINDS):
                alerts.append({"ts": ts, "kind": kind, "data": _loads(data)})
            newest = db.execute("SELECT MAX(ts) FROM events").fetchone()[0]
        finally:
            db.close()
        out["journal"] = {"last": last, "counts_24h": counts, "per_min": per_min, "recent": recent,
                          "alerts": alerts, "newest_ts": newest}
        return out


def _loads(s: str) -> Dict[str, Any]:
    try:
        v = json.loads(s)
        return v if isinstance(v, dict) else {"value": v}
    except ValueError:
        return {"raw": s}


def make_handler(state: State):
    page = open(os.path.join(os.path.dirname(__file__), "dashboard.html"), "rb").read()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):  # quiet
            pass

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                return self._send(200, page, "text/html; charset=utf-8")
            if self.path == "/api/state":
                try:
                    body = json.dumps(state.snapshot()).encode()
                    return self._send(200, body, "application/json")
                except Exception as e:
                    return self._send(500, json.dumps({"error": repr(e)}).encode(), "application/json")
            self._send(404, b"not found", "text/plain")

        def do_POST(self):
            # Only the kill switch. Same-origin check so another site can't flip it.
            origin = self.headers.get("Origin", "")
            if origin and not origin.startswith(("http://127.0.0.1", "http://localhost")):
                return self._send(403, b"forbidden", "text/plain")
            kf = state.cfg.risk.kill_file
            if self.path == "/api/kill":
                with open(kf, "w") as f:
                    f.write(time.ctime() + " (dashboard)\n")
            elif self.path == "/api/unkill":
                if os.path.exists(kf):
                    os.remove(kf)
            else:
                return self._send(404, b"not found", "text/plain")
            self._send(200, b'{"ok":true}', "application/json")

    return H


def serve(cfg: BotConfig, port: int = 8790, with_account: bool = False) -> None:
    os.chdir(ROOT)
    account = None
    if with_account and os.environ.get(cfg.venue.api_key_env):
        account = Account(cfg)
    srv = ThreadingHTTPServer(("127.0.0.1", port), make_handler(State(cfg, account)))
    print(f"cupbot dashboard on http://127.0.0.1:{port}  (Ctrl-C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
