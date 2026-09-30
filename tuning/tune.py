"""Tune cupbot's lag-sniping parameters on REAL Kalshi/Polymarket price paths.

  python3 tune.py [--res 1m|60m] [--lags 1,5,15,60] [--value-h 1440]

What is real and what is assumed
  Real:     the reference fair path R(t) (log-odds blend of Kalshi and Polymarket,
            weight from lab.anchor_weight), including every overshoot, settle-back
            and reversal that actually happened.
  Assumed:  the Cup book is a stale copy of R, lagging L steps (Q(t) = R(t-L)),
            and we pay `cost` (one tick) over Q to take the stale order.
            L is swept as a scenario because the Cup does not open until Oct 1.
  Value:    each position is valued at V = R(t + value_h) (the reference fair price
            later), and scored by exact expected log growth for a binary bet:
            G = V*log(1 + f*(1-p)/p) + (1-V)*log(1-f)  (YES at price p, stake f).

Parameters tuned
  theta   minimum edge (R - Q - cost) to trade
  kelly   Kelly multiplier on f* = (q - p)/(1 - p)
  phi1    fraction of the target size taken on the first move
  settle  steps to wait before topping up (or skipping the top-up if R retraced)
Output: results/tune_<res>.json with the full grid and the robust best set.
"""
import argparse
import itertools
import json
import math
import os
import statistics

import lab

HERE = os.path.dirname(os.path.abspath(__file__))
F_CAP = 0.20  # hard cap per leg as a fraction of balance (mirrors the risk limit)


def blended(pairs, w_k):
    """market -> (sorted ts list, dict ts->fair) using the log-odds blend where both exist."""
    out = {}
    for mid, kg, pg, step in pairs:
        ts = sorted(set(kg) | set(pg))
        fair = {}
        lastk = lastp = None
        for t in ts:
            lastk = kg.get(t, lastk)
            lastp = pg.get(t, lastp)
            if lastk is not None and lastp is not None:
                fair[t] = lab.sigmoid(w_k * lab.logit(lastk) + (1 - w_k) * lab.logit(lastp))
            elif lastk is not None or lastp is not None:
                fair[t] = lastk if lastk is not None else lastp
        if len(fair) > 20:
            out[mid] = (sorted(fair), fair, step)
    return out


def kelly_yes(q, p):
    return (q - p) / (1 - p) if q > p else 0.0


def growth(v, p, f, yes):
    """Expected log growth of staking f at price p (YES) or buying NO at 1-p."""
    if f <= 0:
        return 0.0
    if not yes:
        v, p = 1 - v, 1 - p
    f = min(f, F_CAP, 0.999)
    return v * math.log(1 + f * (1 - p) / p) + (1 - v) * math.log(1 - f)


def simulate(series, L, theta, kelly, phi1, settle, cost, value_h):
    trades = []
    G = 0.0
    for mid, (ts, R, step) in series.items():
        tset = R
        i = 0
        n = len(ts)
        next_ok = ts[0]
        for i in range(n):
            t = ts[i]
            if t < next_ok:
                continue
            tl = t - L * step
            tv = t + value_h * step
            if tl not in R or tv not in R:
                continue
            q, stale = R[t], R[tl]
            up = q - stale
            yes = up > 0
            p = stale + cost if yes else stale - cost  # price we pay (YES) / sell at (NO)
            edge = (q - p) if yes else (p - q)
            if edge < theta or not (0.01 < p < 0.99):
                continue
            fstar = kelly_yes(q, p) if yes else kelly_yes(1 - q, 1 - p)
            f1 = kelly * phi1 * fstar
            v = R[tv]
            g1 = growth(v, p, f1, yes)
            leg = {"m": mid, "t": t, "yes": yes, "p": round(p, 4), "q": round(q, 4), "v": round(v, 4), "f": round(f1, 4), "g": g1}
            G += g1
            trades.append(leg)
            # settle-back check, then top up only if the edge is still there
            ts2 = t + settle * step
            if phi1 < 1 and ts2 in R and (ts2 - L * step) in R and (ts2 + value_h * step) in R:
                q2, stale2 = R[ts2], R[ts2 - L * step]
                p2 = stale2 + cost if yes else stale2 - cost
                edge2 = (q2 - p2) if yes else (p2 - q2)
                if edge2 >= theta and 0.01 < p2 < 0.99:
                    fstar2 = kelly_yes(q2, p2) if yes else kelly_yes(1 - q2, 1 - p2)
                    f2 = kelly * (1 - phi1) * fstar2
                    v2 = R[ts2 + value_h * step]
                    g2 = growth(v2, p2, f2, yes)
                    G += g2
                    trades.append({"m": mid, "t": ts2, "yes": yes, "p": round(p2, 4), "q": round(q2, 4), "v": round(v2, 4), "f": round(f2, 4), "g": g2, "topup": True})
            next_ok = t + max(L, settle) * step + step
    real = sum(((tr["v"] - tr["p"]) if tr["yes"] else (tr["p"] - tr["v"])) for tr in trades)
    pred = sum(((tr["q"] - tr["p"]) if tr["yes"] else (tr["p"] - tr["q"])) for tr in trades)
    return {"G": G, "n": len(trades), "edge_realised_ratio": (real / pred) if pred else None,
            "mean_pred_edge": pred / len(trades) if trades else None, "mean_real_edge": real / len(trades) if trades else None}, trades


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default="1m")
    ap.add_argument("--lags", default="")
    ap.add_argument("--value-h", type=int, default=0, help="steps ahead used as value proxy (default 1 day)")
    ap.add_argument("--cost", type=float, default=0.01)
    ap.add_argument("--w-kalshi", type=float, default=-1)
    a = ap.parse_args()
    pairs = lab.load_pairs(a.res)
    step_min = {"live": 5 / 60, "1m": 1, "60m": 60}[a.res]
    # value proxy horizon: 4h ahead for 1m data (covers ~48h), 1 day for hourly, 1h for live 5s data
    value_h = a.value_h or {"live": 720, "1m": 240, "60m": 24}[a.res]
    lags = [int(x) for x in a.lags.split(",")] if a.lags else {"live": [1, 6, 12, 60, 180], "1m": [1, 5, 15, 60], "60m": [1, 2, 4, 8]}[a.res]
    w_k = a.w_kalshi
    if w_k < 0:
        aw = lab.anchor_weight(pairs, value_h)
        w_k = aw["best_w_kalshi"] if aw else 0.5
    series = blended(pairs, w_k)
    grid = {
        "theta": [0.01, 0.015, 0.02, 0.03, 0.04, 0.06],
        "kelly": [0.25, 0.5, 0.75, 1.0],
        "phi1": [0.25, 0.5, 1.0],
        "settle": {"live": [2, 6, 12, 36], "1m": [1, 3, 5, 15], "60m": [1, 2, 4]}[a.res],
    }
    rows = []
    for L in lags:
        for th, k, ph, st in itertools.product(*grid.values()):
            r, _ = simulate(series, L, th, k, ph, st, a.cost, value_h)
            rows.append({"L": L, "theta": th, "kelly": k, "phi1": ph, "settle": st, **r})
    # robust pick: best mean rank across lag scenarios
    keyf = lambda r: (r["theta"], r["kelly"], r["phi1"], r["settle"])
    ranks = {}
    for L in lags:
        sub = sorted([r for r in rows if r["L"] == L], key=lambda r: -r["G"])
        for i, r in enumerate(sub):
            ranks.setdefault(keyf(r), []).append(i)
    best_key = min(ranks, key=lambda k: statistics.mean(ranks[k]))
    best = [r for r in rows if keyf(r) == best_key]
    per_L_best = {L: max((r for r in rows if r["L"] == L), key=lambda r: r["G"]) for L in lags}
    # sensitivity: realised-edge ratio by theta at the best other params
    sens = [r for r in rows if (r["kelly"], r["phi1"], r["settle"]) == best_key[1:]]
    out = {"res": a.res, "markets": list(series), "w_kalshi": w_k, "value_h_steps": value_h, "cost": a.cost,
           "robust_best": dict(zip(["theta", "kelly", "phi1", "settle"], best_key)), "robust_best_by_L": best,
           "per_L_best": per_L_best, "theta_sensitivity": sens, "grid_rows": len(rows)}
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", f"tune_{a.res}.json"), "w") as f:
        json.dump({**out, "grid": rows}, f, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k not in ("theta_sensitivity",)}, indent=1, default=str))


if __name__ == "__main__":
    main()
