"""Phone alerts and a phone kill switch via ntfy (https://ntfy.sh), stdlib only.

Setup (once): install the ntfy app on your phone, subscribe to your alert topic,
and put these in .env on the server:

    NTFY_TOPIC=cupbot-<long random string>      # alerts go here
    NTFY_CMD_TOKEN=<another random string>      # commands must include it

Commands: publish to the topic `<NTFY_TOPIC>-cmd` (from the ntfy app) a message
    kill <token>      create KILL: cancel every resting order, place nothing
    resume <token>    remove KILL (a drawdown halt still needs `cupbot unkill` over SSH)
    status <token>    reply with equity, cash, positions and last error

Anyone who knows both the topic and the token can stop the bot, but nobody can make
it place a trade from here: there is deliberately no "buy" command."""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.request
from typing import Callable, Dict, Optional

NTFY = "https://ntfy.sh"
ALERT_KINDS = {"kill", "halt", "error", "basket", "race_call", "unknown", "start", "stop", "blocked_burst"}


class Notifier:
    def __init__(self, topic: str, cmd_token: str = "", server: str = NTFY, min_gap_s: float = 30.0):
        self.topic = topic
        self.cmd_token = cmd_token
        self.server = server.rstrip("/")
        self.min_gap_s = min_gap_s
        self._last: Dict[str, float] = {}

    @classmethod
    def from_env(cls) -> Optional["Notifier"]:
        t = os.environ.get("NTFY_TOPIC", "")
        return cls(t, os.environ.get("NTFY_CMD_TOKEN", ""), os.environ.get("NTFY_SERVER", NTFY)) if t else None

    def send(self, title: str, body: str, priority: str = "default", key: str = "") -> None:
        """Fire-and-forget on a background thread; rate-limited per key."""
        k = key or title
        now = time.time()
        if now - self._last.get(k, 0) < self.min_gap_s and priority != "urgent":
            return
        self._last[k] = now

        def go():
            try:
                req = urllib.request.Request(f"{self.server}/{self.topic}", data=body.encode()[:3000],
                                             headers={"Title": title[:200], "Priority": priority})
                urllib.request.urlopen(req, timeout=10).read()
            except Exception:
                pass
        threading.Thread(target=go, daemon=True).start()

    def listen_commands(self, handler: Callable[[str], Optional[str]], stop: threading.Event) -> None:
        """Blocking loop (run in a thread): stream `<topic>-cmd` and call handler(command)."""
        since = "now"
        while not stop.is_set():
            try:
                url = f"{self.server}/{self.topic}-cmd/json?since={since}"
                with urllib.request.urlopen(url, timeout=90) as r:
                    for line in r:
                        if stop.is_set():
                            return
                        ev = json.loads(line.decode() or "{}")
                        if ev.get("event") != "message":
                            continue
                        since = ev.get("id", since)
                        parts = (ev.get("message") or "").strip().split()
                        if len(parts) != 2 or not self.cmd_token or parts[1] != self.cmd_token:
                            continue   # wrong or missing token: ignore silently
                        reply = handler(parts[0].lower())
                        if reply:
                            self.send("cupbot", reply, key=f"reply-{time.time()}")
            except Exception:
                time.sleep(5)


class TeeJournal:
    """Journal wrapper: writes everything to the real journal, and pushes the
    events worth waking up for to the phone."""

    def __init__(self, journal, notifier: Notifier):
        self.j = journal
        self.n = notifier
        self.fills = 0

    def log(self, kind: str, data: Dict) -> None:
        self.j.log(kind, data)
        if kind in ALERT_KINDS:
            pri = "urgent" if kind in ("halt", "kill") else "high" if kind in ("error", "unknown") else "default"
            self.n.send(f"cupbot: {kind}", json.dumps(data, default=str)[:1500], pri, key=kind)
        elif kind == "order" and data.get("filled"):
            self.fills += 1
            if abs(float(data.get("filled", 0)) * float(data.get("avg") or data.get("price") or 0)) >= 5000:
                self.n.send("cupbot: big fill", f"{data.get('side')} {data.get('filled'):.0f} {data.get('market')} "
                            f"@ {data.get('avg') or data.get('price')}  ({data.get('reason')})", key="bigfill")

    def close(self) -> None:
        self.j.close()
