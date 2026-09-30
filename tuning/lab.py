"""Data loading and real-data analysis for tuning cupbot.

  python3 lab.py            # runs every analysis on whatever data exists, writes results/analysis.json

Inputs (any subset):
  data/kalshi/candles_{1m,60m}/<ticker>.csv   ts,yes_bid,yes_ask,last,volume
  data/polymarket/hist_{1m,60m}/<slug>.csv     ts,price
  data/live/*.jsonl                            from recorder.py (resampled to 1 s / 1 m)
  markets.json                                 pairs a Kalshi ticker with a Polymarket slug/token

Analyses
  gaps      Kalshi vs Polymarket gap: size, persistence (AR(1) half-life)
  leadlag   cross-correlation of 1-step changes at lags -k..k; who moves first
  jumps     moves >= J within w, then how much is kept at +h (overshoot / settle-back)
  anchor_w  best Kalshi weight in the log-odds anchor blend, scored on predicting
            the future consensus
Stdlib only.
"""
import csv
import glob
import json
import math
import os
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
EPS = 1e-4


def logit(p):
    p = min(1 - EPS, max(EPS, p))
    return math.log(p / (1 - p))


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def parse_ts(s):
    s = s.strip()
    try:
        v = float(s)
        return v / 1000 if v > 1e12 else v
    except ValueError:
        import datetime as dt
        s = s.replace("Z", "+00:00")
        if "." in s:  # trim sub-microsecond digits
            head, tail = s.split(".", 1)
            frac = "".join(ch for ch in tail if ch.isdigit())
            tz = tail[len(frac):]
            s = f"{head}.{frac[:6]}{tz}"
        return dt.datetime.fromisoformat(s).timestamp()


def fnum(x):
    try:
        v = float(x)
        return v / 100 if v > 1.0 else v  # cents -> dollars
    except (TypeError, ValueError):
        return None


def load_kalshi(path):
    """-> sorted [(ts, mid)] ; mid from bid/ask when both sane, else last."""
    out = []
    with open(path) as f:
        for r in csv.DictReader(f):
            b, a, l = fnum(r.get("yes_bid")), fnum(r.get("yes_ask")), fnum(r.get("last"))
            if b is not None and a is not None and 0 < b <= a < 1 and a - b <= 0.10:
                m = (a + b) / 2
            elif l is not None and 0 < l < 1:
                m = l
            else:
                continue
            out.append((parse_ts(r["ts"]), m, (a - b) if (a is not None and b is not None) else None))
    out.sort()
    return out


def load_poly(path):
    out = []
    with open(path) as f:
        for r in csv.DictReader(f):
            p = fnum(r.get("price") or r.get("p"))
            if p is not None and 0 < p < 1:
                out.append((parse_ts(r.get("ts") or r.get("t")), p, None))
    out.sort()
    return out


def resample(series, step, t0=None, t1=None):
    """Last-observation-carried-forward onto a grid. -> dict ts->price"""
    if not series:
        return {}
    t0 = t0 or math.ceil(series[0][0] / step) * step
    t1 = t1 or series[-1][0]
    out, i, last = {}, 0, None
    t = t0
    while t <= t1:
        while i < len(series) and series[i][0] <= t:
            last = series[i][1]
            i += 1
        if last is not None:
            out[t] = last
        t += step
    return out


def load_markets():
    p = os.path.join(HERE, "markets.json")
    return json.load(open(p))["markets"] if os.path.exists(p) else []



def load_live(step=5):
    """recorder.py JSONL -> {("kalshi", ticker) | ("poly", token): [(ts, mid, spread)]}"""
    out = {}
    for path in sorted(glob.glob(os.path.join(DATA, "live", "*.jsonl"))):
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                t = r.get("recv_ns", 0) / 1e9
                if r.get("v") == "kalshi" and r.get("k") == "book" and r.get("bid") is not None and r.get("ask") is not None:
                    out.setdefault(("kalshi", r["m"]), []).append((t, (r["bid"] + r["ask"]) / 2, r["ask"] - r["bid"]))
                elif r.get("v") == "poly":
                    m = r.get("raw") or {}
                    if r.get("k") == "book" and m.get("bids") and m.get("asks"):
                        b = max(float(x["price"]) for x in m["bids"])
                        a = min(float(x["price"]) for x in m["asks"])
                        out.setdefault(("poly", m.get("asset_id")), []).append((t, (a + b) / 2, a - b))
                    elif r.get("k") == "price_change":
                        for c in m.get("price_changes", []):
                            if c.get("best_bid") and c.get("best_ask"):
                                b, a = float(c["best_bid"]), float(c["best_ask"])
                                out.setdefault(("poly", c.get("asset_id")), []).append((t, (a + b) / 2, a - b))
    for k in out:
        out[k].sort()
    return out

def load_pairs(res):
    """res '1m' or '60m' -> list of (market_id, kalshi_grid, poly_grid, step)."""
    if res == "live":
        step = 5
        lv = load_live(step)
        pairs = []
        for m in load_markets():
            ks = lv.get(("kalshi", m.get("kalshi")), [])
            ps = lv.get(("poly", m.get("poly_token")), [])
            if m.get("poly_invert"):
                ps = [(t, 1 - p, s) for t, p, s in ps]
            if m.get("kalshi_invert"):
                ks = [(t, 1 - p, s) for t, p, s in ks]
            pairs.append((m["id"], resample(ks, step), resample(ps, step), step))
        return pairs
    step = 60 if res == "1m" else 3600
    kdir = os.path.join(DATA, "kalshi", f"candles_{res}")
    pdir = os.path.join(DATA, "polymarket", f"hist_{res}")
    pairs = []
    for m in load_markets():
        kp = os.path.join(kdir, f"{m.get('kalshi')}.csv")
        pp = os.path.join(pdir, f"{m.get('poly_slug')}.csv")
        ks = load_kalshi(kp) if m.get("kalshi") and os.path.exists(kp) else []
        ps = load_poly(pp) if m.get("poly_slug") and os.path.exists(pp) else []
        if m.get("poly_invert"):
            ps = [(t, 1 - p, s) for t, p, s in ps]
        if m.get("kalshi_invert"):
            ks = [(t, 1 - p, s) for t, p, s in ks]
        kg, pg = resample(ks, step), resample(ps, step)
        pairs.append((m["id"], kg, pg, step))
    return pairs


def single_series(res):
    """Every series at this resolution, for per-venue stats. -> [(name, grid, step)]"""
    if res == "live":
        return [(f"{v}:{k}", resample(sr, 5), 5) for (v, k), sr in load_live().items()]
    step = 60 if res == "1m" else 3600
    out = []
    for p in sorted(glob.glob(os.path.join(DATA, "kalshi", f"candles_{res}", "*.csv"))):
        out.append(("kalshi:" + os.path.basename(p)[:-4], resample(load_kalshi(p), step), step))
    for p in sorted(glob.glob(os.path.join(DATA, "polymarket", f"hist_{res}", "*.csv"))):
        out.append(("poly:" + os.path.basename(p)[:-4], resample(load_poly(p), step), step))
    return out


# ------------------------------------------------------------------ analyses
def gap_stats(pairs):
    rows = []
    for mid, kg, pg, step in pairs:
        ts = sorted(set(kg) & set(pg))
        if len(ts) < 10:
            continue
        g = [kg[t] - pg[t] for t in ts]
        ag = [abs(x) for x in g]
        # AR(1) on the gap for persistence
        x, y = g[:-1], g[1:]
        mx, my = statistics.mean(x), statistics.mean(y)
        vx = sum((a - mx) ** 2 for a in x)
        phi = sum((a - mx) * (b - my) for a, b in zip(x, y)) / vx if vx > 0 else float("nan")
        hl = (math.log(0.5) / math.log(phi) * step / 60) if 0 < phi < 1 else None
        rows.append({"market": mid, "n": len(ts), "mean_gap_k_minus_p": round(statistics.mean(g), 4),
                     "abs_gap_p50": round(statistics.median(ag), 4), "abs_gap_p90": round(sorted(ag)[int(0.9 * len(ag))], 4),
                     "ar1": round(phi, 3) if phi == phi else None, "gap_half_life_min": round(hl, 1) if hl else None})
    return rows


def lead_lag(pairs, max_lag=10):
    """corr(dK_t, dP_{t+lag}). Peak at lag>0 => Kalshi leads Polymarket."""
    agg = {l: [] for l in range(-max_lag, max_lag + 1)}
    per = []
    for mid, kg, pg, step in pairs:
        ts = sorted(set(kg) & set(pg))
        if len(ts) < 60:
            continue
        dk = [kg[ts[i]] - kg[ts[i - 1]] for i in range(1, len(ts))]
        dp = [pg[ts[i]] - pg[ts[i - 1]] for i in range(1, len(ts))]
        res = {}
        for l in agg:
            if l >= 0:
                a, b = dk[: len(dk) - l], dp[l:]
            else:
                a, b = dk[-l:], dp[: len(dp) + l]
            c = corr(a, b)
            if c is not None:
                res[l] = c
                agg[l].append(c)
        if res:
            best = max(res, key=lambda l: res[l])
            per.append({"market": mid, "best_lag_steps": best, "corr_at_best": round(res[best], 3), "corr_at_0": round(res.get(0, 0), 3),
                        "k_leads_sum": round(sum(res[l] for l in res if l > 0), 3), "p_leads_sum": round(sum(res[l] for l in res if l < 0), 3)})
    avg = {l: round(statistics.mean(v), 3) for l, v in agg.items() if v}
    return {"per_market": per, "avg_corr_by_lag": avg}


def corr(a, b):
    if len(a) < 10:
        return None
    ma, mb = statistics.mean(a), statistics.mean(b)
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((x - mb) ** 2 for x in b)
    if va <= 0 or vb <= 0:
        return None
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / math.sqrt(va * vb)


def jumps(series_list, J, w_steps, horizons):
    """Find moves |p(t)-p(t-w)| >= J (non-overlapping), report fraction of the move
    still held at t+h: kept = (p(t+h)-p(t-w)) / (p(t)-p(t-w)). 1 = stuck, <1 = retraced,
    >1 = kept going."""
    events = []
    for name, g, step in series_list:
        ts = sorted(g)
        i = w_steps
        while i < len(ts):
            t, t0 = ts[i], ts[i - w_steps]
            move = g[t] - g[t0]
            if abs(move) >= J:
                # extend to the local extreme within the window so t is the peak
                j = i
                while j + 1 < len(ts) and (g[ts[j + 1]] - g[ts[j]]) * move > 0:
                    j += 1
                t = ts[j]
                move = g[t] - g[t0]
                ev = {"series": name, "t": t, "move": round(move, 4), "from": round(g[t0], 4), "to": round(g[t], 4)}
                for h in horizons:
                    th = t + h * step
                    if th in g:
                        ev[f"kept_{h}"] = round((g[th] - g[t0]) / move, 3)
                events.append(ev)
                i = j + w_steps
            else:
                i += 1
    summary = {}
    for h in horizons:
        ks = [e[f"kept_{h}"] for e in events if f"kept_{h}" in e]
        if ks:
            summary[h] = {"n": len(ks), "kept_median": round(statistics.median(ks), 3),
                          "kept_mean": round(statistics.mean(ks), 3), "share_retraced_half": round(sum(k < 0.5 for k in ks) / len(ks), 3)}
    return events, summary


def anchor_weight(pairs, horizon_steps, grid=None):
    """Weight on Kalshi (1-w on Polymarket) in logit space that best predicts the
    average of both venues `horizon_steps` later. Also scores each venue alone."""
    grid = grid or [i / 10 for i in range(11)]
    sse = {w: 0.0 for w in grid}
    n = 0
    for mid, kg, pg, step in pairs:
        ts = sorted(set(kg) & set(pg))
        tset = set(ts)
        for t in ts:
            th = t + horizon_steps * step
            if th not in tset:
                continue
            target = 0.5 * (logit(kg[th]) + logit(pg[th]))
            lk, lp = logit(kg[t]), logit(pg[t])
            for w in grid:
                sse[w] += (w * lk + (1 - w) * lp - target) ** 2
            n += 1
    if not n:
        return None
    best = min(sse, key=sse.get)
    return {"n": n, "best_w_kalshi": best, "rmse_by_w": {str(w): round(math.sqrt(s / n), 4) for w, s in sse.items()}}


def main():
    out = {}
    for res, J, w, hz in (("live", 0.01, 12, [1, 6, 12, 60, 180, 720]), ("1m", 0.02, 5, [1, 5, 15, 60, 240]), ("60m", 0.03, 3, [1, 3, 6, 24, 72])):
        pairs = load_pairs(res)
        singles = single_series(res)
        ev, summ = jumps(singles, J, w, hz)
        out[res] = {
            "series": len(singles),
            "gaps": gap_stats(pairs),
            "leadlag": lead_lag(pairs, 10 if res == "1m" else 6),
            "jumps": {"J": J, "window_steps": w, "summary_by_horizon_steps": summ, "events": ev[:200]},
            "anchor_weight": {str(h): anchor_weight(pairs, h) for h in hz},
        }
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "analysis.json"), "w") as f:
        json.dump(out, f, indent=1, default=str)
    for res in out:
        print(f"== {res}: {out[res]['series']} series")
        print(" gaps:", json.dumps(out[res]["gaps"]))
        print(" leadlag avg:", out[res]["leadlag"]["avg_corr_by_lag"])
        print(" jumps:", out[res]["jumps"]["summary_by_horizon_steps"])
        print(" anchor w:", {h: (v or {}).get("best_w_kalshi") for h, v in out[res]["anchor_weight"].items()})


if __name__ == "__main__":
    main()
