"""Election-night live model (deterministic, no LLM).

Input per race: reporting fraction rho in [0, 1] and the margin among counted
votes. Output per race: P(D wins) and a rough sd.

  reporting races   projected = counted - mirage_s * (1 - rho)
                    (mirage_s: how much early counts in state s lean vs the final,
                     e.g. mail counted late -> early counts lean R -> mirage < 0)
                    sd = sqrt((1 - rho) * race_night_var + resid_var + count noise)
  unreported races  shift the pre-election mean by the swing observed so far in
                    reporting races, shrunk by how much has reported; this is the
                    'reweight the simulations toward early states' step.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Tuple

from .mathutil import norm_cdf


@dataclass
class NightParams:
    national_night_sd: float = 1.9   # national error still unknown when polls close
    region_night_sd: float = 1.0     # regional error still unknown when polls close
    race_night_sd: float = 2.5       # race-level error still unknown when polls close
    resid_sd: float = 0.8            # error not knowable before the count finishes
    count_noise_sd: float = 1.0
    mirage_err_sd: float = 1.5       # our uncertainty about each state's mirage
    shrink_k: float = 1.0            # swing shrinkage: beta = W / (W + k), W = sum of rho
    mirage: Dict[str, float] = field(default_factory=dict)   # state -> points (D minus R)

    def sd_close(self) -> float:
        return (self.national_night_sd ** 2 + self.region_night_sd ** 2 + self.race_night_sd ** 2
                + self.resid_sd ** 2) ** 0.5


@dataclass
class RaceReport:
    state: str
    rho: float
    counted_margin: float


class LiveResultsModel:
    def __init__(self, pre_means: Dict[str, float], states: Dict[str, str], params: NightParams):
        self.pre = dict(pre_means)     # race_id -> pre-election mean margin (our estimate)
        self.states = states           # race_id -> state
        self.p = params
        sd_close = params.sd_close()
        self.informative_margin = 2.5 * sd_close

    def update(self, reports: Dict[str, RaceReport]) -> Dict[str, Tuple[float, float]]:
        p = self.p
        out: Dict[str, Tuple[float, float]] = {}
        num = den = 0.0
        proj: Dict[str, float] = {}
        for rid, r in reports.items():
            if r.rho <= 0:
                continue
            m = p.mirage.get(r.state, 0.0)
            pm = r.counted_margin - m * (1 - r.rho)
            proj[rid] = pm
            if rid in self.pre and abs(self.pre[rid]) < self.informative_margin:
                num += r.rho * (pm - self.pre[rid])
                den += r.rho
        swing = num / den if den > 0 else 0.0
        # (priors of safe races are clipped, so only competitive races inform the swing)
        beta = den / (den + p.shrink_k) if den > 0 else 0.0
        for rid, mean in self.pre.items():
            r = reports.get(rid)
            if r is not None and r.rho > 0:
                var = ((1 - beta) * p.national_night_sd ** 2 + (1 - r.rho) * p.region_night_sd ** 2
                       + (1 - r.rho) * p.race_night_sd ** 2 + p.resid_sd ** 2
                       + ((1 - r.rho) * p.count_noise_sd) ** 2 + ((1 - r.rho) * p.mirage_err_sd) ** 2)
                mu = proj[rid]
            else:
                mu = mean + beta * swing
                var = ((1 - beta) * p.national_night_sd ** 2 + p.region_night_sd ** 2
                       + p.race_night_sd ** 2 + p.resid_sd ** 2)
            sd = math.sqrt(max(var, 1e-6))
            pd = norm_cdf(mu / sd)
            # probability sd (delta method) for the fair-value uncertainty
            dens = math.exp(-0.5 * (mu / sd) ** 2) / math.sqrt(2 * math.pi)
            out[rid] = (pd, max(0.003, dens * 0.5))
        return out
