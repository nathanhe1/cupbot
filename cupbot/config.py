"""Configuration. Defaults live here; config/*.json overrides them; secrets come
from environment variables (or a .env file) and are never written to disk by the bot."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Any, Dict


@dataclass
class FairValueConfig:
    # Sources are blended in log-odds with inverse-variance weights (w = 1/sd^2).
    # These sds are each source's typical error in log-odds. Defaults are set from
    # simulation calibration (see README); re-estimate them from week-one data.
    model_logit_sd: float = 0.45     # our Monte Carlo model
    anchor_logit_sd: float = 0.12    # Kalshi / Polymarket
    cup_mid_logit_sd: float = 0.35   # the Cup's own mid (used only when no anchor)
    live_logit_sd: float = 0.40      # election-night live results model
    signal_cap: float = 0.04         # max shift from any single news signal (prob points)
    signal_total_cap: float = 0.10   # max summed news shift per market
    consistency_logit_sd: float = 0.30  # chamber/combo price implied by race fair values
    pre_info_frac: float = 0.5       # share of race uncertainty resolved before election night


@dataclass
class StrategyConfig:
    depth_reads_per_step: int = 4    # live venue without sizes in its snapshot: books read before trading
    min_edge: float = 0.02           # minimum edge after crossing the spread (prob points)
    gap_buffer: float = 0.01         # ...and at least (Kalshi-Polymarket gap + this) on static trades
    lag_min_edge: float = 0.015      # when the anchor just moved >= 2 ticks and the Cup hasn't followed
    lag_move: float = 0.02           # anchor move (prob, within ~2 min) that counts as "just moved"
    uncertainty_mult: float = 0.5    # extra edge required per unit of fair-value uncertainty
    kelly_mult: float = 1.0          # full Kelly on the blended fair (sim-tuned; halve if week-one calibration is worse)
    max_market_frac: float = 0.20    # max notional in one market, fraction of equity
    max_order_frac: float = 0.05     # max notional per order
    jump_damp: float = 0.5           # first move: take this share of target size...
    jump_threshold: float = 0.02     # ...when fair value jumps by at least this (2 ticks)
    settle_s: float = 90.0           # ...and add the rest after this long only if half the move still holds
    recycle_price: float = 0.95      # election night: sell called winners at or above this bid
    recycle_min_alt_edge: float = 0.05  # ...if a better use of the cash exists
    combo_arb_margin: float = 0.02   # mutually exclusive sets: trade if prices sum off 1 by this
    passive_quotes: bool = False     # rest limit orders around fair value (base building)
    passive_edge: float = 0.03
    passive_frac: float = 0.01
    # final week / election night overrides
    final_week_kelly_mult: float = 1.0
    final_week_max_market_frac: float = 0.25
    night_kelly_mult: float = 1.0
    night_max_market_frac: float = 0.30
    night_min_edge: float = 0.02


@dataclass
class RiskConfig:
    max_order_notional: float = 15_000
    max_market_notional_frac: float = 0.35   # hard ceiling above the strategy's own caps
    max_gross_frac: float = 1.5              # sum |notional| / equity
    max_net_d_frac: float = 0.8              # net Democratic-direction exposure / equity
    max_orders_per_minute: int = 60
    max_price_deviation: float = 0.30        # limit price vs book mid sanity band
    max_book_age_s: float = 30.0
    daily_loss_halt_frac: float = 0.35       # halt new risk after this drawdown in 24h
    kill_file: str = "KILL"


@dataclass
class VenueConfig:
    base_url: str = "https://sig.thesuper.market/api/v1"
    api_key_env: str = "SUPERMARKET_API_KEY"
    # Midterm Elections tournament (GET /tournaments). Every read and order is scoped to it.
    tournament_id: str = "bda92870-621e-47b0-bc3c-3602c5c26f55"
    tournament_slug: str = "midterm-elections"
    # Published limits are per account: 100 reads and 30 writes per minute.
    read_rps: float = 1.5
    write_rps: float = 0.45
    timeout_s: float = 5.0
    price_scale: float = 1.0         # the API quotes YES prices in 0-1
    tick: float = 0.005              # limit prices must be on this tick, 0.005..0.995
    live_trading_enabled: bool = False  # must be true AND --live passed to send real orders
    # Checked against /api/v1/openapi.json (saved in docs/openapi.json) on Oct 1.
    paths: Dict[str, str] = field(default_factory=lambda: {
        "whoami": "/account",
        "markets": "/markets",
        "prices": "/exchanges/prices",
        "book": "/exchanges/{market_id}/orderbook",
        "positions": "/tournaments/{slug}/portfolio/positions",
        "balance": "/tournaments/{slug}",
        "orders": "/orders",
        "cancel": "/orders/{order_id}",
        "cancel_all": "/orders/cancel-all",
    })


@dataclass
class AnalystConfig:
    enabled: bool = False
    model: str = "claude-opus-5-5"
    triage_effort: str = "low"
    review_effort: str = "high"
    max_calls_per_hour: int = 60
    news_domains: list = field(default_factory=lambda: [
        "apnews.com", "reuters.com", "politico.com", "thehill.com", "nytimes.com",
        "washingtonpost.com", "cookpolitical.com", "natesilver.net", "fivethirtyeight.com",
        "decisiondeskhq.com", "realclearpolling.com", "270towin.com", "ballotpedia.org",
    ])


@dataclass
class TournamentConfig:
    enabled: bool = True
    target_multiple: float = 10.0    # fixed bar (x starting cash), used when field_model is off
    field_model: bool = True         # estimate the top-3 bar per scenario from an assumed field of rivals
    field_size: int = 2000           # update from the leaderboard
    field_concentrated_frac: float = 0.35
    min_scale: float = 0.5
    max_scale: float = 3.0
    phases: list = field(default_factory=lambda: ["final_week", "night"])
    # scenario basket: one concentrated correlated bet placed in the last days before the election
    basket_enabled: bool = True
    basket_days_before: float = 5.0
    basket_max_frac: float = 0.5     # most of equity the basket may use
    basket_max_premium: float = 0.05  # pay at most this much over fair value per contract
    basket_lo: float = 0.08          # underdog legs: fair probability between lo and hi
    basket_hi: float = 0.45
    final_max_net_d_frac: float = 1.5  # risk cap on net directional exposure once the basket is on


@dataclass
class BotConfig:
    starting_cash: float = 100_000.0
    races_file: str = "data/races_2026.json"
    market_map_file: str = "config/market_map.json"
    loop_interval_s: float = 2.0
    account_refresh_s: float = 10.0   # re-read balance/positions from the venue this often; between
                                      # refreshes the engine tracks them from its own fills (0 = every step)
    open_ts: float = 1790870400.0      # 2026-10-01 16:00 UTC (noon ET)
    election_ts: float = 1793746800.0  # 2026-11-03 23:00 UTC (6pm ET, first polls close)
    journal_path: str = "journal/journal.sqlite"
    fair: FairValueConfig = field(default_factory=FairValueConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    venue: VenueConfig = field(default_factory=VenueConfig)
    analyst: AnalystConfig = field(default_factory=AnalystConfig)
    tournament: TournamentConfig = field(default_factory=TournamentConfig)


def _merge(obj: Any, overrides: Dict[str, Any]) -> Any:
    names = {f.name for f in fields(obj)}
    for k, v in overrides.items():
        if k.startswith("_"):
            continue
        if k not in names:
            raise KeyError(f"unknown config key: {k}")
        cur = getattr(obj, k)
        if is_dataclass(cur) and isinstance(v, dict):
            _merge(cur, v)
        else:
            setattr(obj, k, v)
    return obj


def load_config(path: str = "") -> BotConfig:
    cfg = BotConfig()
    if path and os.path.exists(path):
        with open(path) as f:
            _merge(cfg, json.load(f))
    return cfg


def load_dotenv(path: str = ".env") -> None:
    """Minimal .env reader (KEY=VALUE lines). Existing env vars win."""
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def to_dict(cfg: BotConfig) -> Dict[str, Any]:
    return asdict(cfg)
