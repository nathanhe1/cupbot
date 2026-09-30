# Latency: ranked changes for cupbot

Measured = numbers from `bench_local.py` (loopback TLS server, fake 10 ms RTT) or from real Kalshi/Polymarket data. Estimated = from the Edge Playbook research, not yet measured from a real location. Re-run `latency_probe.py` on the Mac and on a us-east-1 VM to replace the estimates.

| # | Change | Saves | Evidence |
|---|---|---|---|
| 1 | **Run the bot on a small AWS us-east-1 VM, not the Mac on campus Wi-Fi.** Keep the Mac as read-only standby. | ~10–40 ms per round trip plus Wi-Fi jitter spikes of 100 ms+; also removes sleep/offline risk (the Mac session dropped offline more than a dozen times on Sep 29) | Estimated; the RC session's offline log is measured |
| 2 | **One dedicated, always-warm order connection** (TCP_NODELAY, keep-alive, a cheap authenticated GET every 20 s). Never open a connection on the order path. | 2 round trips per order (TCP + TLS 1.3): measured 33 ms cold vs 11 ms warm at a 10 ms RTT | Measured |
| 3 | **Separate connections for orders and for reads/polling.** HTTP/1.1 serialises requests on one connection, so an order queued behind a slow read waits for the whole read. | Measured 212 ms (order behind a 200 ms read) vs 11 ms on its own connection | Measured |
| 4 | **Check for the Nagle / delayed-ACK trap.** If headers and body go out in separate writes without TCP_NODELAY, each order waits ~40 ms for a delayed ACK. aiohttp sets TCP_NODELAY by default; `urllib`/`http.client` connections do not, and `urllib.request` also opens a fresh connection per call. | Measured 52 ms vs 11 ms per request in the bench before disabling Nagle | Measured |
| 5 | **Kalshi: one batched REST call for all mapped markets** (`GET /markets?tickers=A,B,C...` returns yes_bid/yes_ask for all) instead of one orderbook call per market, until the websocket key arrives. Then switch to the websocket (`orderbook_delta`, `ticker`). | Staleness from N x (1/rate) to 1/rate; the websocket removes polling staleness entirely (avg 500 ms at 1 Hz) | Estimated; the markets response fetched this session carries yes_bid/yes_ask per market |
| 6 | **Precompute triggers.** For every Cup market, compute "if reference mid crosses X, send side S at limit Y for size Z" whenever fair value or balance changes, so the tick handler is one dict lookup and one compare. | Keeps decision at ~0.3 µs (measured) instead of running the Monte Carlo/Kelly path on each tick | Measured |
| 7 | **Pre-built order templates and pre-generated client order ids.** | ~5 µs per order (5.1 µs json.dumps vs 0.19 µs template). Negligible next to network; do it only because it is free | Measured |
| 8 | **Record the order intent in memory before send; persist it to disk off the hot path.** Keep the rule that a client order id exists before every send (so a timeout reconciles instead of resending), but append it to an in-memory journal and let a background task commit to SQLite. | A synchronous SQLite commit with fsync is typically 1–10 ms on a laptop SSD | Estimated |
| 9 | **Polymarket: public market websocket, subscribe only to mapped tokens, ping every 10 s; never the RTDS trade feed.** | RTDS lags by up to seconds | Playbook research |
| 10 | **Clock sync (chrony/NTP) and timestamps on every tick and order.** Needed to measure the Cup's lag vs Kalshi in week one. | Enables item 1 of the week-one measurement plan | n/a |
| 11 | uvloop / orjson | Parsing a Polymarket price_change: 2.75 µs with stdlib json. Not worth adding dependencies on the Mac; fine on the VM | Measured |

What does not help: a Rust rewrite, HTTP/2 multiplexing unless the Cup API supports it (not known), hedged duplicate orders (never; double fills), DNS tricks (a warm connection never resolves).
