"""Correlated Monte Carlo race model.

Margin (D minus R, points) for race i in draw s:

    m[s,i] = mean_i + national[s] + region[s, r(i)] + race_sd_i * e[s,i]

with fat-tailed (Student-t, unit-variance) shocks at every layer. Shocks are
drawn ONCE with a fixed seed (common random numbers) and reused, so repricing
after a mean change is just arithmetic and results don't jitter between calls.

The same scenarios give: P(win) per race, P(Senate control), P(House control),
the four balance-of-power combos, and a per-draw outcome matrix for sizing.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from .mathutil import clip_p, norm_cdf, norm_ppf, student_t


@dataclass
class Race:
    race_id: str
    state: str
    region: str
    mean: float
    sd: float
    holder: str = ""


@dataclass
class ModelParams:
    national_sd: float = 3.0
    region_sd: float = 1.5
    df: float = 6.0
    n_draws: int = 4000
    seed: int = 7
    d_holdover: int = 34
    d_seats_needed: int = 51


class RaceModel:
    def __init__(self, races: Sequence[Race], params: Optional[ModelParams] = None,
                 house: Optional[Race] = None):
        self.params = params or ModelParams()
        self.races: List[Race] = list(races)
        self.house = house
        self.index = {r.race_id: i for i, r in enumerate(self.races)}
        # Multiplies every error term. Set < 1 as election day approaches and
        # uncertainty that polls will resolve has already been resolved.
        self.sd_scale = 1.0
        self.regions = sorted({r.region for r in self.races} | ({house.region} if house else set()))
        self._draw_shocks()

    # ---------------------------------------------------------------- setup
    @classmethod
    def from_json(cls, path: str, params: Optional[ModelParams] = None) -> "RaceModel":
        with open(path) as f:
            data = json.load(f)
        params = params or ModelParams()
        sen = data.get("senate", {})
        params.d_holdover = sen.get("d_holdover", params.d_holdover)
        params.d_seats_needed = sen.get("d_seats_needed", params.d_seats_needed)
        races = [Race(r["race_id"], r["state"], r["region"], float(r["mean"]), float(r["sd"]),
                      r.get("holder", "")) for r in data["races"]]
        house = None
        if "house" in data:
            h = data["house"]
            house = Race(h.get("race_id", "HOUSE"), "US", "national", float(h["mean"]), float(h["sd"]))
        return cls(races, params, house)

    def _draw_shocks(self) -> None:
        p = self.params
        rng = random.Random(p.seed)
        n = p.n_draws
        self._nat = [student_t(rng, p.df) for _ in range(n)]
        self._reg = {reg: [student_t(rng, p.df) for _ in range(n)] for reg in self.regions}
        self._eps = [[student_t(rng, p.df) for _ in range(n)] for _ in self.races]
        self._eps_house = [student_t(rng, p.df) for _ in range(n)] if self.house else None

    # -------------------------------------------------------------- queries
    def total_sd(self, race: Race) -> float:
        p = self.params
        reg = 0.0 if race.region == "national" else p.region_sd
        return self.sd_scale * math.sqrt(p.national_sd ** 2 + reg ** 2 + race.sd ** 2)

    def margins(self, i: int) -> List[float]:
        r = self.races[i]
        p = self.params
        k = self.sd_scale
        reg = self._reg[r.region]
        eps = self._eps[i]
        nat = self._nat
        a, b, c = k * p.national_sd, k * p.region_sd, k * r.sd
        return [r.mean + a * nat[s] + b * reg[s] + c * eps[s] for s in range(p.n_draws)]

    def house_margins(self) -> Optional[List[float]]:
        if not self.house:
            return None
        h, p, k = self.house, self.params, self.sd_scale
        return [h.mean + k * p.national_sd * self._nat[s] + k * h.sd * self._eps_house[s]
                for s in range(p.n_draws)]

    def outcome_matrix(self) -> Dict[str, List[int]]:
        """race_id -> list of 0/1 (D wins) per draw. Includes HOUSE if present."""
        out = {r.race_id: [1 if m > 0 else 0 for m in self.margins(i)]
               for i, r in enumerate(self.races)}
        hm = self.house_margins()
        if hm is not None:
            out[self.house.race_id] = [1 if m > 0 else 0 for m in hm]
        return out

    def summary(self) -> Dict[str, float]:
        """P(D wins) per race, plus SENATE_D, HOUSE_D and the four combos."""
        n = self.params.n_draws
        outcomes = self.outcome_matrix()
        res = {rid: sum(v) / n for rid, v in outcomes.items()}
        seats = [0] * n
        for r in self.races:
            v = outcomes[r.race_id]
            for s in range(n):
                seats[s] += v[s]
        need = self.params.d_seats_needed - self.params.d_holdover
        sen = [1 if seats[s] >= need else 0 for s in range(n)]
        res["SENATE_D"] = sum(sen) / n
        res["_expected_d_seats"] = self.params.d_holdover + sum(seats) / n
        if self.house:
            hv = outcomes[self.house.race_id]
            res["HOUSE_D"] = sum(hv) / n
            res["COMBO_DD"] = sum(1 for s in range(n) if hv[s] and sen[s]) / n
            res["COMBO_DR"] = sum(1 for s in range(n) if hv[s] and not sen[s]) / n  # D House, R Senate
            res["COMBO_RD"] = sum(1 for s in range(n) if not hv[s] and sen[s]) / n
            res["COMBO_RR"] = sum(1 for s in range(n) if not hv[s] and not sen[s]) / n
        return res

    # ---------------------------------------------------------- inversion
    def set_means_from_probs(self, probs: Dict[str, float]) -> None:
        """Back out race means from race win probabilities (normal approx on
        the total sd). Used to make chamber prices consistent with race prices."""
        for r in self.races:
            if r.race_id in probs:
                r.mean = self.total_sd(r) * norm_ppf(clip_p(probs[r.race_id], 1e-4))
        if self.house and self.house.race_id in probs:
            self.house.mean = self.total_sd(self.house) * norm_ppf(clip_p(probs[self.house.race_id], 1e-4))

    def analytic_prob(self, race_id: str) -> float:
        """Normal-approximation win probability (sanity check vs the MC)."""
        if self.house and race_id == self.house.race_id:
            r = self.house
        else:
            r = self.races[self.index[race_id]]
        return norm_cdf(r.mean / self.total_sd(r))
