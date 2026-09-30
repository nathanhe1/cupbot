# cupbot tuning harness

Standalone, stdlib-only (Python 3.9+), no installs. Copy this folder to `~/Trading/tuning/` on the Mac (or a VM) and run from inside it. Nothing here places orders.

| File | What it does |
|---|---|
| `markets.json` | Pairs each market: `{"id", "kalshi": ticker, "poly_slug", "poly_token": Dem-YES CLOB token id, "poly_invert": false}` |
| `recorder.py` | Records live books/trades: Polymarket public websocket; Kalshi websocket if `KALSHI_KEY_ID` + `KALSHI_PRIVATE_KEY_PATH` are set (needs `cryptography`), else REST polling on one warm connection. Writes `data/live/*.jsonl` with local receive timestamps. |
| `latency_probe.py` | DNS/TCP/TLS/TTFB, cold vs warm request timing, serving-region headers, Polymarket websocket exchange-to-receive lag. Appends to `results/latency.jsonl`. `--cup-url <a GET from the official API docs> --cup-key $KEY` adds the Cup (API only, never the website). |
| `bench_local.py` | Loopback benchmarks of the order path (cold vs warm, head-of-line blocking, Nagle trap, payload build). |
| `lab.py` | Real-data analysis: Kalshi-Polymarket gap and its half-life, lead/lag, jump overshoot and settle-back, best Kalshi weight in the anchor blend. Writes `results/analysis.json`. |
| `tune.py` | Grid-searches min edge, Kelly multiplier, first-move fraction and settle window on real price paths, scoring exact expected log growth, across assumed Cup lags. Writes `results/tune_<res>.json`. |
| `LATENCY.md`, `RECOMMENDATIONS.md` | Findings and the changes to make in cupbot. |

## Run on the Mac

```
cd ~/Trading/tuning
python3 latency_probe.py --n 30 --ws-seconds 60      # from the Mac; repeat later from a us-east-1 VM
python3 bench_local.py
nohup python3 recorder.py --minutes 0 > recorder.log 2>&1 &   # leave running; the more hours the better
python3 lab.py && python3 tune.py --res 1m            # any time, on data/ so far
```

`lab.py`/`tune.py` read historical CSVs from `data/kalshi/candles_{1m,60m}/`, `data/polymarket/hist_{1m,60m}/` and the live recorder output. After the recorder has run for a few hours, analyse it with `python3 lab.py` and `python3 tune.py --res live` (5-second grid built from `data/live`).

Keep the recorder running through the Cup's first week: its timestamps are what `measure-mm` needs to compare the Cup's price moves against Kalshi/Polymarket.
