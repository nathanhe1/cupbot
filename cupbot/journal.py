"""Order/fill/signal journal. SQLite in WAL mode, written from a background
thread so no disk I/O happens on the order path."""

from __future__ import annotations

import json
import os
import queue
import sqlite3
import threading
import time
from typing import Any, Dict


class NullJournal:
    def log(self, kind: str, data: Dict[str, Any]) -> None:
        pass

    def close(self) -> None:
        pass


class Journal(NullJournal):
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.path = path
        self.q: "queue.Queue" = queue.Queue(maxsize=100_000)
        self._t = threading.Thread(target=self._run, name="journal", daemon=True)
        self._t.start()

    def log(self, kind: str, data: Dict[str, Any]) -> None:
        try:
            self.q.put_nowait((time.time(), kind, data))
        except queue.Full:
            pass  # never block trading on the journal

    def _run(self) -> None:
        db = sqlite3.connect(self.path)
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("CREATE TABLE IF NOT EXISTS events (ts REAL, kind TEXT, data TEXT)")
        db.commit()
        while True:
            item = self.q.get()
            if item is None:
                break
            batch = [item]
            while len(batch) < 500:
                try:
                    nxt = self.q.get_nowait()
                except queue.Empty:
                    break
                if nxt is None:
                    self.q.put(None)
                    break
                batch.append(nxt)
            db.executemany("INSERT INTO events VALUES (?,?,?)",
                           [(ts, k, json.dumps(d, default=str)) for ts, k, d in batch])
            db.commit()
        db.close()

    def close(self) -> None:
        self.q.put(None)
        self._t.join(timeout=5)
