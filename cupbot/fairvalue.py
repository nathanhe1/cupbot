"""Fair value: one probability per market.

  race market   = blend in log-odds of (our model, external anchor), or (model, Cup mid)
                  when no anchor exists; on election night the live results model
                  dominates. Capped, decaying news shifts are added last.
  chamber/combo = computed from the race fair values through the correlated
                  model (so they are consistent with the races), then blended with
                  any chamber anchor.

Claude never sets a fair value. It can only propose a Signal, which is clamped
here to +/- signal_cap per item and +/- signal_total_cap per market and decays
to zero over its TTL."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .config import FairValueConfig
from .mathutil import clamp, clip_p, logit_blend
from .racemodel import RaceModel
from .types import Market, Signal


@dataclass
class FairQuote:
    p: float                  # fair probability of YES
    unc: float                # rough 1-sd uncertainty of p (prob points)
    parts: Dict[str, float] = field(default_factory=dict)


def inverse_variance_blend(srcs: List[Tuple[float, float]]) -> Tuple[float, float]:
    """Blend (probability, logit sd) sources in log-odds with weights 1/sd^2.
    Returns (p, uncertainty in probability points)."""
    ws = [1.0 / (sd * sd) for _, sd in srcs]
    p = logit_blend([q for q, _ in srcs], ws)
    logit_sd = (1.0 / sum(ws)) ** 0.5
    return p, logit_sd * p * (1 - p)


def to_yes(p_d: float, d_sign: int) -> float:
    return p_d if d_sign >= 0 else 1 - p_d


def to_d(p_yes: float, d_sign: int) -> float:
    return p_yes if d_sign >= 0 else 1 - p_yes


CHAMBER_KEYS = {"SENATE_D", "HOUSE_D", "COMBO_DD", "COMBO_DR", "COMBO_RD", "COMBO_RR"}


class FairValueEngine:
    def __init__(self, cfg: FairValueConfig, markets: Dict[str, Market], model: RaceModel):
        self.cfg = cfg
        self.markets = markets
        # private copy whose means we overwrite with blended race fairs
        self.consistency = copy.deepcopy(model)
        self.signals: Dict[str, List[Signal]] = {}
        self.rejected_signals = 0

    # ------------------------------------------------------------ signals
    def add_signal(self, sig: Signal) -> bool:
        if sig.market_id not in self.markets or not sig.sources:
            self.rejected_signals += 1
            return False
        sig.delta = clamp(sig.delta, -self.cfg.signal_cap, self.cfg.signal_cap)
        self.signals.setdefault(sig.market_id, []).append(sig)
        return True

    def signal_shift(self, market_id: str, now: float) -> float:
        total = 0.0
        live = []
        for s in self.signals.get(market_id, []):
            age = now - s.ts
            if age < 0 or age >= s.ttl_s:
                if age < 0:
                    live.append(s)
                continue
            live.append(s)
            total += s.delta * (1 - age / s.ttl_s)
        self.signals[market_id] = live
        return clamp(total, -self.cfg.signal_total_cap, self.cfg.signal_total_cap)

    # ------------------------------------------------------------ compute
    def compute(self, now: float, model_pd: Dict[str, float], anchors: Dict[str, float],
                mids: Dict[str, float],
                live: Optional[Dict[str, Tuple[float, float]]] = None) -> Dict[str, FairQuote]:
        """model_pd: race_id -> P(D) from our model (incl. SENATE_D etc).
        anchors: market_id -> P(YES) from Kalshi/Polymarket.
        mids: market_id -> Cup book mid.
        live: race_id -> (P(D) from live results, sd) on election night."""
        c = self.cfg
        out: Dict[str, FairQuote] = {}
        race_fair_d: Dict[str, float] = {}

        for mid, m in self.markets.items():
            if m.closed or m.kind != "race" or not m.race_id or m.race_id not in model_pd:
                continue
            p_model = to_yes(model_pd[m.race_id], m.d_sign)
            srcs = [(p_model, c.model_logit_sd)]
            parts = {"model": p_model}
            a = anchors.get(mid)
            if a is not None:
                srcs.append((a, c.anchor_logit_sd))
                parts["anchor"] = a
            elif mids.get(mid) is not None:
                srcs.append((mids[mid], c.cup_mid_logit_sd))
                parts["cup_mid"] = mids[mid]
            if live and m.race_id in live:
                lp = to_yes(live[m.race_id][0], m.d_sign)
                srcs.append((lp, c.live_logit_sd))
                parts["live"] = lp
                srcs[0] = (p_model, 1e3)   # pre-election model is stale once results arrive
            p, unc = inverse_variance_blend(srcs)
            shift = self.signal_shift(mid, now)
            if shift:
                parts["news"] = shift
                p = clip_p(p + shift, 0.005)
            out[mid] = FairQuote(clip_p(p, 0.002), unc, parts)
            race_fair_d[m.race_id] = to_d(out[mid].p, m.d_sign)

        # markets the model doesn't cover (governors, House districts): anchor + Cup mid only
        for mid, m in self.markets.items():
            if m.closed or mid in out or m.kind in ("chamber", "combo"):
                continue
            if m.race_id in model_pd:
                continue
            a = anchors.get(mid)
            if a is None:
                continue   # no model and no anchor: don't trade it
            srcs = [(a, c.anchor_logit_sd)]
            parts = {"anchor": a}
            if mids.get(mid) is not None:
                srcs.append((mids[mid], c.cup_mid_logit_sd))
                parts["cup_mid"] = mids[mid]
            p, unc = inverse_variance_blend(srcs)
            shift = self.signal_shift(mid, now)
            if shift:
                parts["news"] = shift
                p = clip_p(p + shift, 0.005)
            out[mid] = FairQuote(clip_p(p, 0.002), unc, parts)

        # model-only races that have no Cup market still feed the consistency model
        for rid, pd in model_pd.items():
            if rid not in CHAMBER_KEYS and not rid.startswith("_"):
                race_fair_d.setdefault(rid, pd)

        chamber_needed = any(m.kind in ("chamber", "combo") and not m.closed for m in self.markets.values())
        if chamber_needed:
            if live:  # races without a Cup market still update from live results
                for rid, (lp, _) in live.items():
                    if rid not in race_fair_d or rid == "HOUSE":
                        race_fair_d[rid] = lp
            self.consistency.set_means_from_probs(race_fair_d)
            summ = self.consistency.summary()
            for mid, m in self.markets.items():
                if m.closed or m.kind not in ("chamber", "combo") or not m.race_id:
                    continue
                if m.race_id not in summ:
                    continue
                p_model = to_yes(summ[m.race_id], m.d_sign)
                parts = {"consistency": p_model}
                srcs = [(p_model, c.consistency_logit_sd)]
                a = anchors.get(mid)
                if a is not None:
                    srcs.append((a, c.anchor_logit_sd))
                    parts["anchor"] = a
                elif mids.get(mid) is not None:
                    srcs.append((mids[mid], c.cup_mid_logit_sd))
                    parts["cup_mid"] = mids[mid]
                p, unc = inverse_variance_blend(srcs)
                shift = self.signal_shift(mid, now)
                if shift:
                    parts["news"] = shift
                    p = clip_p(p + shift, 0.005)
                out[mid] = FairQuote(clip_p(p, 0.002), unc, parts)
        return out
