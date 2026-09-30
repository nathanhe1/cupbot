"""Discover external anchor markets and refresh race priors (stdlib only).

    python -m cupbot refresh-priors            # show Kalshi / Polymarket prices per race
    python -m cupbot refresh-priors --write    # also write means into data/races_2026.json
                                               # and tickers/tokens into data/anchors_2026.json

Public, read-only market data from Kalshi and Polymarket. Nothing here touches
the Predictions Cup site.

Race markets are two-party in our model: P(D side) = 1 - P(Republican wins), so
an independent who is the main anti-Republican candidate (e.g. NE, MT, SD, ID)
is counted on the D side. That matters for Senate control; check how each
independent says they would caucus."""

from __future__ import annotations

import json
import math
import re
import time
import urllib.parse
import urllib.request
from typing import Dict, List, Optional, Tuple

from .mathutil import clip_p, logit_blend, norm_ppf

KALSHI = "https://api.elections.kalshi.com/trade-api/v2"
GAMMA = "https://gamma-api.polymarket.com"

STATE_NAMES = {
    "AL": "Alabama", "AK": "Alaska", "AR": "Arkansas", "CO": "Colorado", "DE": "Delaware", "FL": "Florida",
    "GA": "Georgia", "ID": "Idaho", "IL": "Illinois", "IA": "Iowa", "KS": "Kansas", "KY": "Kentucky",
    "LA": "Louisiana", "ME": "Maine", "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota",
    "MS": "Mississippi", "MT": "Montana", "NE": "Nebraska", "NH": "New Hampshire", "NJ": "New Jersey",
    "NM": "New Mexico", "NC": "North Carolina", "OH": "Ohio", "OK": "Oklahoma", "OR": "Oregon",
    "RI": "Rhode Island", "SC": "South Carolina", "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas",
    "VA": "Virginia", "WV": "West Virginia", "WY": "Wyoming",
}
ALL_STATES = dict(STATE_NAMES, **{
    "AZ": "Arizona", "CA": "California", "CT": "Connecticut", "HI": "Hawaii", "IN": "Indiana", "MD": "Maryland",
    "MO": "Missouri", "NV": "Nevada", "NY": "New York", "ND": "North Dakota", "PA": "Pennsylvania", "UT": "Utah",
    "VT": "Vermont", "WA": "Washington", "WI": "Wisconsin"})
SPECIALS = {"FL", "OH"}   # 2026 special elections (Kalshi uses SENATE{ST}S)


def _get(url: str, params: Optional[Dict] = None, timeout: float = 10.0):
    if params:
        url += "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": "cupbot/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _mid(bid, ask) -> Optional[float]:
    try:
        b, a = float(bid), float(ask)
    except (TypeError, ValueError):
        return None
    if 0 < b <= a < 1 and a - b <= 0.10:
        return (a + b) / 2
    return None


# ---------------------------------------------------------------- Kalshi
def kalshi_event_markets(event_ticker: str) -> List[dict]:
    try:
        return _get(f"{KALSHI}/markets", {"event_ticker": event_ticker}).get("markets", [])
    except Exception:
        return []


def kalshi_race(st: str) -> Optional[Dict]:
    """-> {"r_ticker", "d_ticker", "p_r"} for a 2026 Senate race, or None."""
    cands = ([f"SENATE{st}S-26"] if st in SPECIALS else []) + [f"SENATE{st}-26"]
    for ev in cands:
        ms = kalshi_event_markets(ev)
        r = next((m for m in ms if m["ticker"].endswith("-R")), None)
        if r is None:
            continue
        d = next((m for m in ms if m["ticker"].endswith("-D")), None)
        p_r = _mid(r.get("yes_bid_dollars"), r.get("yes_ask_dollars"))
        # third-party candidates (e.g. independents) are recorded too, with the D side
        others = [m.get("yes_sub_title") for m in ms if not m["ticker"].endswith(("-R", "-D"))]
        return {"event": ev, "r_ticker": r["ticker"], "d_ticker": d["ticker"] if d else None, "p_r": p_r,
                "cand_r": r.get("yes_sub_title"), "cand_d": d.get("yes_sub_title") if d else None,
                "cand_other": [o for o in others if o]}
    return None


def kalshi_control(chamber: str) -> Optional[Dict]:
    ev = "CONTROLS-2026" if chamber == "senate" else "CONTROLH-2026"
    ms = kalshi_event_markets(ev)
    d = next((m for m in ms if m["ticker"].endswith("-D")), None)
    if d is None:
        return None
    return {"event": ev, "d_ticker": d["ticker"], "p_d": _mid(d.get("yes_bid_dollars"), d.get("yes_ask_dollars"))}


# ------------------------------------------------------------ Polymarket
def poly_event(slug: str) -> Optional[dict]:
    try:
        evs = _get(f"{GAMMA}/events", {"slug": slug})
    except Exception:
        return None
    return evs[0] if evs else None


def _poly_market(ev: dict, needle: str) -> Optional[Dict]:
    for m in ev.get("markets", []):
        q = m.get("question", "")
        if needle in q and not m.get("closed"):
            toks = json.loads(m.get("clobTokenIds") or "[]")
            p = _mid(m.get("bestBid"), m.get("bestAsk"))
            if p is None and m.get("outcomePrices"):
                try:
                    p = float(json.loads(m["outcomePrices"])[0])
                except (ValueError, IndexError):
                    p = None
            return {"question": q, "yes_token": toks[0] if toks else None, "p_yes": p}
    return None


def poly_race(st: str) -> Optional[Dict]:
    ev = poly_event(f"{STATE_NAMES[st].lower().replace(' ', '-')}-senate-election-winner")
    if ev is None:
        return None
    r = _poly_market(ev, "Republicans win")
    if r is None:
        return None
    return {"slug": ev["slug"], "r_token": r["yes_token"], "p_r": r["p_yes"]}


def poly_control(chamber: str) -> Optional[Dict]:
    ev = poly_event(f"which-party-will-win-the-{chamber}-in-2026")
    if ev is None:
        return None
    d = _poly_market(ev, "Democratic Party control")
    return None if d is None else {"slug": ev["slug"], "d_token": d["yes_token"], "p_d": d["p_yes"]}


# ------------------------------------------------------------ refresh
def gather(states: List[str], pause_s: float = 0.15) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    for st in states:
        k, p = kalshi_race(st), poly_race(st)
        ps = [1 - x["p_r"] for x in (k, p) if x and x.get("p_r") is not None]
        out[f"SEN-{st}"] = {"kalshi": k, "polymarket": p,
                            "p_d": logit_blend(ps, [1.0] * len(ps)) if ps else None}
        time.sleep(pause_s)
    for ch, rid in (("senate", "SENATE_D"), ("house", "HOUSE_D")):
        k, p = kalshi_control(ch), poly_control(ch)
        ps = [x["p_d"] for x in (k, p) if x and x.get("p_d") is not None]
        out[rid] = {"kalshi": k, "polymarket": p, "p_d": logit_blend(ps, [1.0] * len(ps)) if ps else None}
    return out


def implied_means(found: Dict[str, Dict], races_path: str, sd_scale: float = 1.0) -> Dict[str, float]:
    """Race means (D margin, points) that reproduce the anchor probabilities under
    the model's own error structure."""
    from .racemodel import ModelParams, RaceModel
    m = RaceModel.from_json(races_path, ModelParams(n_draws=200))
    m.sd_scale = sd_scale
    means = {}
    for r in m.races:
        f = found.get(r.race_id)
        if f and f.get("p_d") is not None:
            means[r.race_id] = round(m.total_sd(r) * norm_ppf(clip_p(f["p_d"], 0.003)), 2)
    if m.house and found.get("HOUSE_D", {}).get("p_d") is not None:
        means[m.house.race_id] = round(m.total_sd(m.house) * norm_ppf(clip_p(found["HOUSE_D"]["p_d"], 0.003)), 2)
    return means


def write_priors(races_path: str, means: Dict[str, float], found: Dict[str, Dict]) -> None:
    with open(races_path) as f:
        data = json.load(f)
    for r in data["races"]:
        if r["race_id"] in means:
            r["mean"] = means[r["race_id"]]
            r["p_d_anchor"] = round(found[r["race_id"]]["p_d"], 4)
    if "house" in data and data["house"].get("race_id", "HOUSE") in means:
        data["house"]["mean"] = means[data["house"].get("race_id", "HOUSE")]
    data["_README"] = ("Race means backed out of Kalshi/Polymarket prices by `python -m cupbot refresh-priors "
                       f"--write` on {time.strftime('%Y-%m-%d %H:%M')}. Edit by hand to disagree with the market; "
                       "'sd' is the race-level error on top of the shared national and regional errors.")
    with open(races_path, "w") as f:
        json.dump(data, f, indent=2)


def write_anchor_index(path: str, found: Dict[str, Dict]) -> None:
    """race_id -> Kalshi tickers and Polymarket tokens, used to build market_map.json."""
    idx = {}
    for rid, f in found.items():
        k, p = f.get("kalshi") or {}, f.get("polymarket") or {}
        idx[rid] = {
            "kalshi_d": k.get("d_ticker"), "kalshi_r": k.get("r_ticker"),
            "poly_d_token": p.get("d_token"), "poly_r_token": p.get("r_token"),
            "p_d": f.get("p_d"),
            "cand_d": k.get("cand_d"), "cand_r": k.get("cand_r"), "cand_other": k.get("cand_other", []),
        }
    with open(path, "w") as fh:
        json.dump({"_README": "Generated by refresh-priors. race_id -> anchor tickers/tokens.", **idx}, fh, indent=2)


def anchors_for(race_id: str, d_sign: int, index: Dict[str, dict]) -> Dict[str, dict]:
    """Anchor entries for a Cup market whose YES is the D side (d_sign=+1) or R side (-1)."""
    e = index.get(race_id) or {}
    out: Dict[str, dict] = {}
    if d_sign > 0:
        if e.get("kalshi_d"):
            out["kalshi"] = {"ticker": e["kalshi_d"], "invert": False}
        elif e.get("kalshi_r"):
            out["kalshi"] = {"ticker": e["kalshi_r"], "invert": True}
        if e.get("poly_d_token"):
            out["polymarket"] = {"token_id": e["poly_d_token"], "invert": False}
        elif e.get("poly_r_token"):
            out["polymarket"] = {"token_id": e["poly_r_token"], "invert": True}
    elif d_sign < 0:
        if e.get("kalshi_r"):
            out["kalshi"] = {"ticker": e["kalshi_r"], "invert": False}
        elif e.get("kalshi_d"):
            out["kalshi"] = {"ticker": e["kalshi_d"], "invert": True}
        if e.get("poly_r_token"):
            out["polymarket"] = {"token_id": e["poly_r_token"], "invert": False}
        elif e.get("poly_d_token"):
            out["polymarket"] = {"token_id": e["poly_d_token"], "invert": True}
    return out


# ----------------------------------------------------- Cup title matching
DEM_WORDS = ("democrat", "democratic", "dem ", "(d)", " d ")
REP_WORDS = ("republican", "gop", "(r)", " r ")


def guess_market(title: str) -> Dict:
    """Best guess of {kind, race_id, d_sign} from a Cup market title. Always review."""
    t = " " + re.sub(r"[^a-z0-9'()]+", " ", title.lower()) + " "
    d = any(w in t for w in DEM_WORDS)
    r = any(w in t for w in REP_WORDS)
    sign = 1 if d and not r else -1 if r and not d else 0
    if "senate" in t and ("control" in t or "majority" in t or "win the senate" in t):
        return {"kind": "chamber", "race_id": "SENATE_D", "d_sign": sign}
    if "house" in t and ("control" in t or "majority" in t or "win the house" in t):
        return {"kind": "chamber", "race_id": "HOUSE_D", "d_sign": sign}
    st = _find_state(t)
    if "senate" in t and st:
        return {"kind": "race", "race_id": f"SEN-{st}", "d_sign": sign}
    if "governor" in t and st:
        return {"kind": "other", "race_id": f"GOV-{st}", "d_sign": sign}
    m = re.search(r"\b([A-Z]{2})[- ]?0?(\d{1,2})\b", title)
    if m and m.group(1) in ALL_STATES:
        return {"kind": "other", "race_id": f"HOUSE-{m.group(1)}{int(m.group(2)):02d}", "d_sign": sign}
    m = re.search(r"(\d{1,2})(st|nd|rd|th)", t)
    if m and st and ("house" in t or "district" in t or "congress" in t):
        return {"kind": "other", "race_id": f"HOUSE-{st}{int(m.group(1)):02d}", "d_sign": sign}
    return {"kind": "other", "race_id": None, "d_sign": sign}


def _find_state(t: str) -> Optional[str]:
    for st, name in sorted(ALL_STATES.items(), key=lambda kv: -len(kv[1])):
        if f" {name.lower()} " in t or f" {name.lower()}'s " in t:
            return st
    for st in ALL_STATES:
        if f" {st.lower()} " in t:
            return st
    return None


def kalshi_for_race_id(race_id: str) -> Optional[Dict]:
    """Kalshi D/R tickers for governor (GOV-XX) and House district (HOUSE-XXNN) ids."""
    if race_id.startswith("GOV-"):
        ev = f"GOVPARTY{race_id[4:]}-26"
    elif race_id.startswith("HOUSE-") and len(race_id) == 10:
        ev = f"HOUSE{race_id[6:8]}{int(race_id[8:])}-26"
    else:
        return None
    ms = kalshi_event_markets(ev)
    d = next((m["ticker"] for m in ms if m["ticker"].endswith("-D")), None)
    r = next((m["ticker"] for m in ms if m["ticker"].endswith("-R")), None)
    return {"kalshi_d": d, "kalshi_r": r} if (d or r) else None
