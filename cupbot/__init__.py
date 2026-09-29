"""cupbot: a trading bot for the Susquehanna Predictions Cup (2026 midterms).

Layout:
  types        shared dataclasses (markets, books, orders, fills)
  mathutil     logit / normal CDF / Kelly helpers
  racemodel    correlated Monte Carlo race model (races -> chamber control)
  fairvalue    blends model, external anchors and capped news signals in log-odds
  strategy     turns fair values + books into desired orders
  risk         hard pre-trade limits and the kill switch (never bypassed)
  engine       the loop: data -> fair value -> strategy -> risk -> orders -> journal
  exchange/    exchange interface, mock exchange, live Super.Market client
  sim/         simulated contest world and backtester
"""

__version__ = "0.1.0"
