"""Final-week sizing for the tournament objective.

Only the top three are paid, so late in the contest the goal is the
probability that final balance clears the leaderboard threshold b, not expected
log growth. Given our book and the model's correlated scenarios we compute

    W_s(lam) = equity + lam * sum_i q_i * (outcome_is - price_i)

(lam scales every position at today's prices) and pick the lam in the allowed
range that maximises P(W >= b). If we are far below b with little time left
the answer is usually 'more risk'; if we are already above b it is 'less'.
The engine turns lam into a Kelly multiplier for the final week."""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

from .types import Market


def scenario_payoffs(positions: Dict[str, float], marks: Dict[str, float], markets: Dict[str, Market],
                     outcomes: Dict[str, List[int]], n: int) -> List[float]:
    """Per-scenario P&L of holding `positions` to resolution vs marking at `marks`."""
    pnl = [0.0] * n
    for mid, q in positions.items():
        m = markets.get(mid)
        if m is None or not q or m.race_id not in outcomes:
            continue
        price = marks.get(mid, 0.5)
        o = outcomes[m.race_id]
        for s in range(n):
            yes = o[s] if m.d_sign >= 0 else 1 - o[s]
            pnl[s] += q * (yes - price)
    return pnl


def best_scale(equity: float, pnl: Sequence[float], target: float,
               grid: Iterable[float] = (0.5, 0.75, 1.0, 1.5, 2.0, 3.0)) -> Tuple[float, float]:
    """(lam, P(W >= target)) maximising the probability of clearing target.
    Ties go to the smaller lam (less risk for the same odds)."""
    best = (1.0, -1.0)
    n = len(pnl) or 1
    for lam in grid:
        p = sum(1 for x in pnl if equity + lam * x >= target) / n
        if p > best[1] + 1e-9:
            best = (lam, p)
    return best


def chamber_outcomes(outcomes: Dict[str, List[int]], d_needed_from_class: int) -> Dict[str, List[int]]:
    """Add SENATE_D / HOUSE_D / COMBO_* rows to a race outcome matrix."""
    races = [k for k in outcomes if k.startswith("SEN-")]
    n = len(next(iter(outcomes.values())))
    sen = [1 if sum(outcomes[r][s] for r in races) >= d_needed_from_class else 0 for s in range(n)]
    out = dict(outcomes)
    out["SENATE_D"] = sen
    if "HOUSE" in outcomes:
        h = outcomes["HOUSE"]
        out["HOUSE_D"] = h
        out["COMBO_DD"] = [h[s] & sen[s] for s in range(n)]
        out["COMBO_DR"] = [h[s] & (1 - sen[s]) for s in range(n)]
        out["COMBO_RD"] = [(1 - h[s]) & sen[s] for s in range(n)]
        out["COMBO_RR"] = [(1 - h[s]) & (1 - sen[s]) for s in range(n)]
    return out


# ----------------------------------------------------------------------------
# Scenario baskets: the deliberate concentrated bet for the final days.
# A basket is a set of legs that all pay in one correlated scenario (e.g. every
# competitive race breaking toward one party). We evaluate each candidate at
# several sizes over the model's joint scenarios and pick the one that maximises
# P(final balance >= target). Buying longshots at the ask usually has slightly
# negative expected value; this trades that for a better shot at the top three.
# ----------------------------------------------------------------------------

Leg = Tuple[str, str, float]   # (market_id, "yes" | "no", price paid per contract)


def candidate_baskets(markets: Dict[str, Market], fairs: Dict[str, float], asks: Dict[str, float],
                      bids: Dict[str, float], lo: float = 0.08, hi: float = 0.45) -> Dict[str, List[Leg]]:
    """Underdog legs grouped by the party that has to outperform, plus single chamber/combo longshots."""
    d_legs: List[Leg] = []
    r_legs: List[Leg] = []
    out: Dict[str, List[Leg]] = {}
    for mid, m in markets.items():
        if m.closed or mid not in fairs or not m.d_sign:
            continue
        p_yes = fairs[mid]
        p_d = p_yes if m.d_sign > 0 else 1 - p_yes
        # cost of the D side and of the R side of this market
        d_side = ("yes", asks.get(mid)) if m.d_sign > 0 else ("no", _no_cost(bids.get(mid)))
        r_side = ("no", _no_cost(bids.get(mid))) if m.d_sign > 0 else ("yes", asks.get(mid))
        if m.kind == "race":
            if lo <= p_d <= hi and d_side[1]:
                d_legs.append((mid, d_side[0], d_side[1]))
            if lo <= 1 - p_d <= hi and r_side[1]:
                r_legs.append((mid, r_side[0], r_side[1]))
        elif m.kind in ("chamber", "combo") and asks.get(mid) and lo / 2 <= p_yes <= hi:
            out[f"single {mid}"] = [(mid, "yes", asks[mid])]
    if d_legs:
        out["D outperforms polls"] = d_legs
    if r_legs:
        out["R outperforms polls"] = r_legs
    return out


def _no_cost(bid):
    return (1 - bid) if bid else None


def basket_payoffs(legs: Sequence[Leg], notional: float, markets: Dict[str, Market],
                   outcomes: Dict[str, List[int]], n: int) -> Tuple[List[float], Dict[str, float]]:
    """P&L per scenario of spending `notional` split equally across legs; also the
    signed YES quantities to hold (positive = long YES, negative = short YES / long NO)."""
    pnl = [0.0] * n
    qty: Dict[str, float] = {}
    per = notional / max(1, len(legs))
    for mid, side, price in legs:
        m = markets[mid]
        o = outcomes.get(m.race_id)
        if o is None or price <= 0:
            continue
        q = per / price
        qty[mid] = q if side == "yes" else -q
        for s in range(n):
            yes = o[s] if m.d_sign >= 0 else 1 - o[s]
            win = yes if side == "yes" else 1 - yes
            pnl[s] += q * (win - price)
    return pnl, qty


def plan_basket(equity: float, base_pnl: Sequence[float], target: float, baskets: Dict[str, List[Leg]],
                markets: Dict[str, Market], outcomes: Dict[str, List[int]], n: int,
                fracs: Sequence[float] = (0.0, 0.1, 0.2, 0.35, 0.5)) -> Tuple[str, float, float, Dict[str, float]]:
    """-> (basket name, fraction of equity, P(W >= target), target quantities).

    If no candidate can reach `target` in any scenario, the target is lowered
    step by step (maximise the highest reachable bar) rather than doing nothing."""
    units = {name: basket_payoffs(legs, 1.0, markets, outcomes, n) for name, legs in baskets.items()}
    bar = target
    while True:
        best = _best_for(equity, base_pnl, bar, units, n, fracs)
        if best[2] > 0 or bar <= equity:
            return best
        bar = max(equity, bar * 0.8)


def _best_for(equity, base_pnl, target, units, n, fracs):
    p0 = sum(1 for x in base_pnl if equity + x >= target) / max(1, n)
    best = ("none", 0.0, p0, {})
    for name, (unit, qty) in units.items():
        for f in fracs:
            if f <= 0:
                continue
            spend = f * equity
            p = sum(1 for s in range(n) if equity + base_pnl[s] + spend * unit[s] >= target) / max(1, n)
            if p > best[2] + 1e-9:
                best = (name, f, p, {k: v * spend for k, v in qty.items()})
    return best
