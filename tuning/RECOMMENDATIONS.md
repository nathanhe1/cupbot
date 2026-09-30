# cupbot changes from real Kalshi/Polymarket data (round 1, Sep 30 2026)

Data: Kalshi and Polymarket history pulled Sep 30 for Senate control, House control and 7–10 Senate races. That is 30 days hourly and the last 48 h by the minute, plus about 48 h of Kalshi trades on 5 markets (`data/*/README.md`). Analysis: `lab.py` → `results/analysis.json`; tuning: `tune.py --res 60m` → `results/tune_60m.json`.

Limits: the Cup isn't open yet, so its lag behind the reference venues is swept as an assumption (1–8 h at hourly resolution). Second-scale behaviour needs `recorder.py` running (see README). The month had strong trends (Senate D 0.49 → 0.62, TX D 0.50 → 0.62), so momentum flatters the second half.

## What the data shows

1. **The venues disagree persistently.** Kalshi prices the Democratic side lower than Polymarket in every market. The mean gap is -0.1 to -2.8 points, and the Senate-control gap is -2.8 points hourly. The median absolute gap is 1–3 points and the 90th percentile is 2–5 points. Gaps decay slowly, with a half-life of 13–50 hours on hourly data.
2. **Neither venue predicts the other's future price better.** In a log-odds blend, a 50/50 weight minimises error at every horizon from 1 to 72 hours. The error curve is flat around 0.4–0.6.
3. **Kalshi leads slightly.** On hourly changes, the cross-correlation peaks at +1 hour, with Kalshi first (0.13 vs 0.09 at lag 0). This is weak and needs the live recorder to confirm at second scale.
4. **Hourly moves stick.** There were 28 moves of 3 points or more within 3 hours. The median share of the move still held was 1.0 at 1, 6, 24 and 72 hours. Only 7% gave back half within 6 hours, and 18% within 3 days.
5. **Big trade prints overshoot, then half-retrace within minutes.** On Sep 30 at 19:10 UTC, 77k-contract sweeps hit Texas and then, 95 s later, Senate control. Both printed 0.64 from 0.62, the mid was back to 0.62 within 5 minutes, and the price settled at 0.63. Moving Kalshi's price 1–2 cents takes 50–80k contracts.
6. **Quiet markets hardly move.** Over 48 h, Kalshi's Senate-control mid changed in 11 of 2,880 minutes, and Maine's changed in none. The lag edge comes almost entirely around news and events.
7. **Tuned parameters (hourly, lag swept 1–8 h, one-tick cost):**
   - A minimum edge of 1–1.5 points over the blended fair value gave the most log growth at every assumed lag. At 2 points trade count fell 2–10x; at 4 points or more, growth was 0 at a 1-hour lag and less than half the peak at longer lags.
   - Realised edge divided by predicted edge was 0.78 in the first half and 1.16 in the second, so edges survive.
   - Kelly 1.0 vs 1.5 was equal in the first half. The 1.5 only won in the trending half. Full Kelly stands; don't go above it.
   - Taking half the size first, then topping up after one settle step, was best or tied.

## Changes to make in cupbot

| Setting / code | Change | Why |
|---|---|---|
| Anchor blend | `fair_anchor = sigmoid(0.5*logit(kalshi_mid) + 0.5*logit(poly_mid))`; fall back to the single venue when one is stale for more than 60 s | Finding 2 |
| Model weight | Keep `w_model ≈ 0.3` against that anchor | Can't be tuned before resolution; literature value |
| Price input | Use the **quote mid** (bid/ask), never last-trade price, for both venues | Finding 5: prints overshoot by 1–2 cents |
| Min edge, lag trades (both venues moved together, Cup hasn't) | **1.5 points** after one-tick cost | Finding 7 |
| Min edge, static fair-value trades | **max(2 points, abs(kalshi − poly) + 1 point)** | Finding 1: a 1–3 point venue gap is normal noise, not signal |
| Kelly | Keep full Kelly (1.0), per-leg cap 20% of balance, and cap the sum across markets that move together (chamber + its races) | Finding 7; the month's correlated trend |
| First move | 50% of target size at trigger; top up after **60–120 s** only if the reference mid still holds at least half the move; otherwise hold the first leg | Findings 4 and 5 |
| Trigger | The reference mid must move at least 2 ticks; ignore 1-tick flicker | Finding 6 |
| Cross-market | When a race's anchor moves, reprice the chamber markets through the race model at once | Finding 5: Texas moved 95 s before Senate control |
| Event calendar | Pre-arm triggers for debates (ME Oct 6, 8, 13, 15; TX Oct 6, 13; IA Oct 7, 13; GA Oct 8; MI Oct 8, 21) and FEC deadlines (Oct 15, 22); keep the recorder running around them | Finding 6: the edge is event-driven |
| Latency | See `LATENCY.md` (dedicated warm order connection, separate read connection, TCP_NODELAY, batched Kalshi REST until a websocket key) | Measured |

## Next rounds

- Run `recorder.py` on the Mac (or a VM) continuously, then `lab.py` and `tune.py --res live`, to get second-scale overshoot and lead/lag.
- From Oct 1: add the Cup's prices to the recorder through the official API and measure its real lag, which replaces the swept L.
- Re-tune weekly as data accumulates.
