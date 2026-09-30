"""Election-night race calls from Bluesky (Jetstream websocket, free, no key).

Watches posts from a short allow-list of call desks (AP, Decision Desk HQ by
default; verify the handles before Nov 3), parses "X wins the <state> Senate
race" style posts, and turns them into race calls the engine treats as
near-certain (P = 0.995 for the called side, never 1.0).

A misread call is the most expensive mistake on election night, so a call only
takes effect when either
  * `min_sources` different desks have called the same winner, or
  * one desk has called it AND our live results model already agrees (P >= 0.85).
Conflicting calls cancel each other and the race is left to the live model."""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set, Tuple

from .discover import ALL_STATES

try:
    import aiohttp
except ImportError:  # pragma: no cover
    aiohttp = None

JETSTREAM = "wss://jetstream2.us-east.bsky.network/subscribe"
RESOLVE = "https://public.api.bsky.app/xrpc/com.atproto.identity.resolveHandle"
DEFAULT_DESKS = ["apnews.com", "decisiondeskhq.bsky.social"]

CALL_WORDS = re.compile(r"\b(wins?|won|projects?|projected winner|race call|we can call|reelected|re-elected|defeats?)\b",
                        re.I)
NOT_CALL = re.compile(r"\b(leads?|leading|ahead|too (early|close) to call|expected to|could|if|poll|primary|runoff)\b",
                      re.I)


@dataclass
class Call:
    race_id: str
    party: str          # "D" (D side, incl. main anti-R independent) or "R"
    source: str
    ts: float
    text: str


@dataclass
class CallBook:
    min_sources: int = 2
    agree_p: float = 0.85
    calls: Dict[str, List[Call]] = field(default_factory=dict)

    def add(self, c: Call) -> None:
        lst = self.calls.setdefault(c.race_id, [])
        if not any(x.source == c.source and x.party == c.party for x in lst):
            lst.append(c)

    def effective(self, live: Optional[Dict[str, Tuple[float, float]]] = None) -> Dict[str, str]:
        """race_id -> party for calls that pass the confirmation rule."""
        out = {}
        for rid, lst in self.calls.items():
            parties = {c.party for c in lst}
            if len(parties) != 1:
                continue                        # desks disagree: ignore
            party = parties.pop()
            n = len({c.source for c in lst})
            agree = False
            if live and rid in live:
                p_d = live[rid][0]
                agree = (p_d >= self.agree_p) if party == "D" else (p_d <= 1 - self.agree_p)
            if n >= self.min_sources or (n >= 1 and agree):
                out[rid] = party
        return out


class CallParser:
    """Maps post text to (race_id, party) using candidate names from
    data/anchors_2026.json plus party words. Returns None when unsure."""

    def __init__(self, anchors_index: Dict[str, dict]):
        self.names: Dict[str, Tuple[str, str]] = {}
        for rid, e in anchors_index.items():
            if not rid.startswith("SEN-") or not isinstance(e, dict):
                continue
            for key, party in (("cand_d", "D"), ("cand_r", "R")):
                if e.get(key):
                    self.names[_last(e[key])] = (rid, party)
            for o in e.get("cand_other") or []:
                self.names.setdefault(_last(o), (rid, "D"))   # main independent counted on the D side

    def parse(self, text: str) -> Optional[Tuple[str, str]]:
        if not CALL_WORDS.search(text) or NOT_CALL.search(text):
            return None
        low = text.lower()
        if "senate" not in low:
            return None
        hits = {v for k, v in self.names.items() if re.search(rf"\b{re.escape(k)}\b", low)}
        if len(hits) == 1:
            rid, party = hits.pop()
            st = _state_in(low)
            if st is None or st == rid[4:]:
                return rid, party
            return None
        st = _state_in(low)
        if st is None:
            return None
        d = re.search(r"\((d|dem)\)|\bdemocrat", low)
        r = re.search(r"\((r|rep)\)|\brepublican|\bgop\b", low)
        if bool(d) == bool(r):
            return None
        return f"SEN-{st}", "D" if d else "R"


def _last(name: str) -> str:
    return name.strip().split()[-1].lower()


def _state_in(low: str) -> Optional[str]:
    for st, name in sorted(ALL_STATES.items(), key=lambda kv: -len(kv[1])):
        if re.search(rf"\b{name.lower()}\b", low):
            return st
    return None


async def resolve_dids(handles: List[str]) -> Dict[str, str]:
    sess = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10))
    out = {}
    try:
        for h in handles:
            async with sess.get(RESOLVE, params={"handle": h}) as r:
                if r.status == 200:
                    out[(await r.json())["did"]] = h
    finally:
        await sess.close()
    return out


async def listen(parser: CallParser, book: CallBook, handles: List[str] = None,
                 on_call: Optional[Callable[[Call], None]] = None) -> None:
    """Run forever: stream posts from the desks, parse calls into `book`."""
    if aiohttp is None:
        raise RuntimeError("pip3 install --user aiohttp")
    dids = await resolve_dids(handles or DEFAULT_DESKS)
    if not dids:
        raise RuntimeError("could not resolve any call-desk handles")
    params = [("wantedCollections", "app.bsky.feed.post")] + [("wantedDids", d) for d in dids]
    while True:
        try:
            async with aiohttp.ClientSession() as s:
                async with s.ws_connect(JETSTREAM, params=params, heartbeat=30) as ws:
                    async for msg in ws:
                        if msg.type != aiohttp.WSMsgType.TEXT:
                            continue
                        ev = json.loads(msg.data)
                        rec = (ev.get("commit") or {}).get("record") or {}
                        text = rec.get("text", "")
                        got = parser.parse(text) if text else None
                        if got:
                            c = Call(got[0], got[1], dids.get(ev.get("did"), "?"), time.time(), text)
                            book.add(c)
                            if on_call:
                                on_call(c)
        except Exception:
            await asyncio.sleep(2)
