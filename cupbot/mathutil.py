"""Small numeric helpers, stdlib only."""

from __future__ import annotations

import math
import random
from statistics import NormalDist

_N = NormalDist()
EPS = 1e-4


def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def clip_p(p: float, eps: float = EPS) -> float:
    return clamp(p, eps, 1 - eps)


def logit(p: float) -> float:
    p = clip_p(p)
    return math.log(p / (1 - p))


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1 / (1 + z)
    z = math.exp(x)
    return z / (1 + z)


def norm_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def norm_ppf(p: float) -> float:
    return _N.inv_cdf(clip_p(p, 1e-9))


def student_t(rng: random.Random, df: float) -> float:
    """Standard Student-t draw scaled to unit variance (df > 2)."""
    z = rng.gauss(0, 1)
    chi2 = rng.gammavariate(df / 2, 2)
    t = z / math.sqrt(chi2 / df)
    return t * math.sqrt((df - 2) / df)


def logit_blend(ps, weights) -> float:
    """Weighted average in log-odds space. Weights are renormalised."""
    tw = sum(weights)
    if tw <= 0:
        raise ValueError("weights must sum to > 0")
    return sigmoid(sum(w * logit(p) for p, w in zip(ps, weights)) / tw)


def kelly_yes(q: float, price: float) -> float:
    """Full-Kelly fraction of wealth to spend buying YES at `price` with belief q."""
    if price <= 0 or price >= 1:
        return 0.0
    return max(0.0, (q - price) / (1 - price))


def kelly_no(q: float, price: float) -> float:
    """Full-Kelly fraction to sell YES (buy NO at 1-price) with belief q."""
    if price <= 0 or price >= 1:
        return 0.0
    return max(0.0, (price - q) / price)


def normal_quadrature(n: int = 64):
    """Equal-weight quantile nodes for E[f(Z)], Z~N(0,1). Simple and robust;
    accurate to ~1e-3 for the smooth integrands used here."""
    xs = [norm_ppf((i + 0.5) / n) for i in range(n)]
    return xs, [1.0 / n] * n


def poisson_binomial_at_least(ps, k: int) -> float:
    """P(sum of independent Bernoulli(ps) >= k), exact DP."""
    dist = [1.0]
    for p in ps:
        nxt = [0.0] * (len(dist) + 1)
        for i, v in enumerate(dist):
            nxt[i] += v * (1 - p)
            nxt[i + 1] += v * p
        dist = nxt
    if k <= 0:
        return 1.0
    return sum(dist[k:]) if k < len(dist) else 0.0
