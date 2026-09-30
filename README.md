# cupbot: a Predictions Cup trading bot

A trading bot for the Susquehanna Predictions Cup (2026 US midterms, trading Oct 1 noon ET to Nov 4 noon ET).
It follows the project's Game Plan and Edge Playbook: a rule-based engine with hard risk limits on the order path,
a correlated Monte Carlo race model, Kalshi and Polymarket as price anchors, Claude only as a capped news
analyst, an election-night live results model, and a final-week sizer aimed at finishing top three.

**Status:** the engine, risk manager, simulator, paper trading and tests work today on this Mac with no installs.
The live platform client is a skeleton until the API docs are read (see "Before Oct 1"). **Nothing sends a real
order unless `venue.live_trading_enabled` is `true` in the config file AND you pass `--live`.**

## Quick start

```bash
cd ~/Trading
python3 -m unittest discover -s tests -t .      # 62 tests, ~20 s
python3 -m cupbot model                          # race model probabilities from data/races_2026.json
python3 -m cupbot paper --seed 1                 # paper-trade one simulated contest, tick by tick
python3 -m cupbot sim --runs 40 --baseline       # backtest 40 simulated contests vs doing nothing
python3 -m cupbot sim --runs 30 --sweep          # same, under harder / different assumptions
```

In VS Code: open the `Trading` folder, then use Run and Debug (the configurations in `.vscode/launch.json`)
or the Testing panel (unittest is preconfigured).

The core needs only the Python standard library (works on the macOS system Python 3.9). Live trading needs
one package, which also works on Python 3.9:

```bash
pip3 install --user aiohttp
cp .env.example .env        # then put your Predictions Cup API key in it
```

The Claude news analyst needs the Anthropic SDK, which needs Python 3.10+. This Mac has no Homebrew, so install
Python 3.12 from python.org, then `python3.12 -m venv .venv && source .venv/bin/activate && pip install -r requirements-dev.txt`.

## What you need to supply

| Item | Where | Needed for |
|---|---|---|
| Predictions Cup API key (read + trade scopes) | `.env` → `SUPERMARKET_API_KEY` | `whoami`, `live`, `measure-mm` |
| Tournament id, rate limits, price format | `config/default.json` → `venue` | live |
| Real endpoint paths and field names | `config/default.json` → `venue.paths`, and the `_parse_*` helpers in `cupbot/exchange/supermarket.py` | live |
| Review of the draft market map | `python3 -m cupbot draft-map` writes `config/market_map.json`; check each `race_id` / `d_sign` | live, anchors |
| Anthropic API key | `.env` → `ANTHROPIC_API_KEY`, then `analyst.enabled: true` | news analyst (optional) |
| Kalshi API key (optional) | `.env` → `KALSHI_API_KEY` | Kalshi websocket; REST polling works without it |

## Commands

| Command | What it does |
|---|---|
| `python3 -m cupbot refresh-priors --write` | Pulls every Senate race, Senate and House control from Kalshi and Polymarket (public data), backs out race means and writes them plus anchor tickers to `data/` |
| `python3 -m cupbot draft-map` | Lists the Cup's markets, guesses race and party from each title, attaches Kalshi/Polymarket anchors, writes `config/market_map.json` for you to review |
| `python3 -m cupbot preflight` | Read-only PASS/FAIL checks: key, endpoints, market map, price parsing, anchors, account, rate-limit headers |
| `python3 -m cupbot status` | Kill-switch state, last start/stop/halt/error, recent orders (from the journal) |
| `python3 -m cupbot record` | Records Kalshi + Polymarket top-of-book changes to `journal/ticks/` (public data) |
| `python3 -m cupbot whoami` | Read-only check that the API key works |
| `python3 -m cupbot live` | Runs against the real platform in **dry run**: journals the orders it would send |
| `python3 -m cupbot live --live` | Sends real orders, only if the config also allows it |
| `python3 -m cupbot measure-mm --minutes 120` then `mm-report` | Week-one measurement of how fast the sponsor's market maker reprices after Kalshi/Polymarket move |
| `python3 -m cupbot kill` / `unkill` | Creates / removes `KILL`. With it present the engine cancels everything and places nothing |
| `python3 -m cupbot cancel-all` | Standalone cancel of every resting order |

## Running it unattended on a server

The bot should not depend on the MacBook being open. Run it on a small always-on server in AWS **us-east-1
(Virginia)**: Kalshi, Polymarket and (most likely) the Cup platform are all served from there, so it is also
the lowest-latency place to trade from. From this Mac the measured round trips are Kalshi 35 ms (p90 114 ms)
and Polymarket's feed 51 ms (p90 140 ms), over campus Wi-Fi that also drops.

**Cost:** an AWS Lightsail Linux instance, 1 GB plan, about **$5/month** (roughly $6 for the whole contest).
An EC2 t4g.micro is similar. Nothing else is paid; ntfy alerts are free.

**Setup (about 20 minutes, once):**
1. AWS console → Lightsail → Create instance → Linux, **Ubuntu 24.04**, region **Virginia (us-east-1)**, 1 GB plan.
   Download the SSH key it offers.
2. On the Mac: `bash deploy/push.sh ubuntu@<server-ip> ~/Downloads/<key>.pem`
3. On the server (`ssh -i <key>.pem ubuntu@<server-ip>`): `bash ~/Trading/deploy/setup_vm.sh`
4. `sudo nano /home/cupbot/Trading/.env`: API key, `NTFY_TOPIC`, `NTFY_CMD_TOKEN` (long random strings).
   Copy your reviewed `config/market_map.json` and `config/default.json` over with `push.sh` too.
5. `cupbot preflight`, then `sudo systemctl start cupbot` and `cupbot logs`. It runs as a dry run.
6. When you decide to trade for real: set `live_trading_enabled: true` in `config/default.json` on the server
   **and** `CUPBOT_ARGS=--live` in `.env`, then `cupbot restart`.

What keeps it safe with nobody watching:
- systemd restarts it within 10 s after a crash or reboot; on every start it cancels leftover resting orders,
  and on every stop it cancels its resting orders before exiting.
- All hard risk limits stay in force (per order, per market, gross, net direction, orders per minute, price band,
  stale book, 35% 24-hour drawdown halt). A drawdown halt stays on until you clear it over SSH.
- **Phone alerts** (ntfy app, subscribe to your `NTFY_TOPIC`): start/stop, errors, halts, kill, big fills,
  race calls, the final-week basket.
- **Phone kill switch:** in the ntfy app publish to `<NTFY_TOPIC>-cmd` the message `kill <token>`
  (also `resume <token>`, `status <token>`). There is no command that places a trade. Over SSH from a phone
  app (e.g. Termius): `cupbot kill`, `cupbot status`, `cupbot logs`, `cupbot stop`.
- The tick recorder runs as its own service (`cupbot-recorder`), so tuning data keeps accumulating.

Update the code later from the Mac with `bash deploy/push.sh ubuntu@<server-ip> <key>.pem`.

## How it works

```
Kalshi / Polymarket ──(websocket or poll)──▶ AnchorBook ──┐ on change: poke engine
data/races_2026.json ──▶ RaceModel (correlated MC) ───────┤
Claude analyst ──▶ Signal (capped Δp, sources, decays) ───┤
election-night results ──▶ LiveResultsModel ──────────────┤
                                                           ▼
Cup books ──▶ Engine.step: FairValue ──▶ Strategy ──▶ RiskManager ──▶ Exchange ──▶ Journal (SQLite, background thread)
```

- **Fair value** (`fairvalue.py`): each source (model, anchor, Cup mid, live results) is blended in log-odds with
  inverse-variance weights. Chamber and combo markets are priced from the race fair values through the correlated
  model, so they stay consistent with the races. News shifts are clamped to ±4 points per item and ±10 per market.
- **Strategy** (`strategy.py`): buys when the ask is below fair by more than a threshold, sells when the bid is
  above; limit prices stop the sweep before the edge is gone; fractional Kelly sizing with per-market and
  per-order caps; smaller size on the first tick after a big fair-value jump; exits; election-night capital
  recycling (sell near-certain winners when there is a better use for the cash); arbitrage when mutually
  exclusive combo markets don't sum to 1.
- **Risk** (`risk.py`): kill switch, per-order / per-market / gross / net-Democratic-direction caps, orders per
  minute, price sanity band, stale-book check, and a halt after a large 24-hour drawdown. Orders that only reduce a
  position are always allowed (except under the kill switch).
- **Tournament sizer** (`tournament.py`): in the final week and on election night it scales position sizes to
  maximise P(final balance ≥ target), using the model's correlated scenarios. Set `tournament.target_multiple`
  from the leaderboard in late October.
- **Election night** (`election_night.py`): for reporting races, the counted margin corrected for each state's
  counting-order "mirage"; for unreported races, the pre-close market-implied margin shifted by the swing seen
  so far. Race calls from AP and Decision Desk HQ on Bluesky (`calls.py`) pin a race at 99.5% once two desks
  agree, or one desk plus our live model. State results-feed parsers still need writing before Nov 3; they push
  into `LiveFeeds.push_reports()`.
- **Markets outside the model** (governors, House districts): traded off their Kalshi anchor blended with the Cup
  mid; with no anchor they are not traded.

## The simulator, and what it does and does not prove

`cupbot/sim/world.py` simulates a whole contest: a latent truth for every race, a sponsor market maker that
quotes around a lagged and biased view of it and moves its price when traded against, students who send random
orders and leave stale limit orders, anchor venues that track the truth closely, poll and news feeds for the
bot, and an election night where states report in order with counting-order bias. The real engine trades
against a real limit order book (`exchange/mock.py`), then everything resolves.

It is useful for: checking the mechanics end to end, checking the risk limits are never breached, choosing
sizing and thresholds, and seeing **which assumptions the profit depends on**. It cannot tell you whether the
real Cup market maker is as slow or as biased as the simulated one; the bot only profits in the simulator
because the simulator was built with those edges. The week-one `measure-mm` data is what tells you which
sweep row looks like reality.

### After folding in the real-data tuning (Sep 30 evening, 40 contests per row)

Edge rules from 30 days of Kalshi/Polymarket history (`tuning/RECOMMENDATIONS.md`): trade only past
max(2 points, venue gap + 1), 1.5 points when the anchor has just moved; quote mids only; half size on a
fresh move, the rest after 90 s if it holds; 20% per-market cap at full Kelly.

| Scenario | Median result | P(loss) | P(2x) | P(top 3) |
|---|---|---|---|---|
| Default assumptions | 1.39x (was 1.27x) | 2% (was 12%) | 0% | 0% |
| MM reprices with no lag | 1.02x | 45% | 0% | 0% |
| Deep MM | 1.65x | 8% | 10% | 2% |

### Results with real market priors (Sep 30 build, 40 contests per row)

After `refresh-priors` the race priors match Kalshi/Polymarket (Senate D about 62%, House D about 92%). Real
markets have more lopsided races, so there are more cheap longshots for rivals to buy and the simulated top-3
bar rises to about 4.5x.

| Scenario | Median result | P(loss) | P(2x) | P(top 3) |
|---|---|---|---|---|
| Default assumptions | 1.27x | 12% | 0% | 0% |
| MM reprices with no lag | 1.02x | 45% | 0% | 0% |
| Deep MM (4x size, 1/4 price impact) | 1.69x | 5% | 28% | 2% |
| Deep MM, no race-call desks | 1.79x | 10% | 25% | 5% |

With an instant market maker the bot loses on election night (-6k on average, worst run 0.45x): if week-one
`measure-mm` shows the Cup reprices within seconds, cut `night_kelly_mult` and `night_max_market_frac`.
Race calls barely change the result in the sim because the live model already reaches 95%+ by the time desks call.

### Results (Sep 29 build, placeholder priors, 40 simulated contests per row, `python3 -m cupbot sim --runs 40 --sweep`)

"Top-3 bar" is the third-best final balance among 2,000 simulated rivals (half casual, the rest betting most of
their balance on longshots at final-week prices). That field is an assumption too.

| Scenario | Median result | P(loss) | P(top 3) |
|---|---|---|---|
| Default assumptions | 1.23x | 20% | 0% |
| Sponsor MM half as biased | 1.16x | 28% | 0% |
| MM reprices with no lag | 1.07x | 45% | 0% |
| Sharp MM with no lag (hard case) | 1.00x | 48% | 0% |
| Anchors twice as noisy | 1.15x | 28% | 0% |
| Anchors on only 40% of markets | 1.24x | 30% | 0% |
| No student order flow | 1.14x | 28% | 0% |
| Thin MM (500 contracts a level) | 1.14x | 25% | 0% |
| **Deep MM (4x size, 1/4 price impact)** | **1.52x** | 12% | **18%** |

Risk-limit breaches: 0 in every run. Median top-3 bar: 2.6x to 3.1x.

What this says:
- **The edge comes from the sponsor's market maker being slower or more biased than Kalshi/Polymarket.** If it
  reprices instantly and accurately, the bot roughly breaks even; the thresholds keep it from bleeding.
- **Liquidity decides whether a top-3 finish is reachable at all.** With the default book depth the bot's best run
  is about 2x and never clears the bar; with a deep book it places in about 1 contest in 5.
- Sizing was tuned over 54 settings (Kelly 0.5 / 1 / 2, minimum edge 1-3 points, uncertainty buffer, market cap):
  full Kelly with a 2-point minimum edge and half the fair-value uncertainty as extra buffer was best.
- A naive "buy the longshot basket" final bet *lowered* top-3 odds (12% → 5% with deep liquidity), because it
  pays in the same scenarios as the crowd's longshots. The sizer now estimates the bar per scenario from the
  field; with that, the final-week scaling and basket raised top-3 odds from 10% to 18% (60 contests each).
- Calibration in the sim: our Monte Carlo model alone is much worse than the anchors, so it gets a small weight;
  its real jobs are markets with no anchor and keeping chamber/combo prices consistent with the races.

## Oct 1 checklist (in order)

1. `pip3 install --user aiohttp`, `cp .env.example .env`, put the API key in `.env`.
2. Read https://sig.thesuper.market/api/v1/docs while logged in. Fill in `venue.paths`, `price_scale`,
   `tournament_id`, and set `read_rps` / `write_rps` below the published limits. Note position limits, whether
   you can sell a position before resolution (the recycling depends on it), and whether shorting / buying NO exists.
3. `python3 -m cupbot refresh-priors --write` (same morning, so priors are current).
4. `python3 -m cupbot draft-map`, then review `config/market_map.json`, especially three-way races (NE, MT, SD, ID).
5. `python3 -m cupbot preflight` until there are no FAILs.
6. `python3 -m cupbot live` (dry run) through the noon open; read the `dry_order` rows in `journal/journal.sqlite`.
7. `python3 -m cupbot measure-mm --minutes 120` in the first days and compare with the simulator sweep.
8. Only then set `live_trading_enabled: true`, keep small caps in `config/default.json` → `risk`, and run
   `python3 -m cupbot live --live`.
