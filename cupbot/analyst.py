"""Claude news analyst. Off the order path: it reads news and returns a
proposed probability SHIFT with sources. The fair-value engine clamps it
(+/- signal_cap per item, +/- signal_total_cap per market) and decays it; the
strategy and risk manager decide whether anything trades. Claude never sizes
or sends orders.

Two tiers (per the playbook's latency revision):
  triage(): effort "low", one headline -> {relevant, direction, delta, confidence}
  review(): effort "high", only for items triage marks as big, with web search
Requires ANTHROPIC_API_KEY and `pip install anthropic` (Python >= 3.10 for SDK 1.x).
"""

from __future__ import annotations

import json
import time
from typing import Dict, List, Optional
from urllib.parse import urlparse

from .config import AnalystConfig
from .types import Signal

try:
    import anthropic
except ImportError:  # pragma: no cover
    anthropic = None

SIGNAL_SCHEMA = {
    "type": "object",
    "properties": {
        "relevant": {"type": "boolean"},
        "direction": {"type": "string", "enum": ["toward_yes", "toward_no", "none"]},
        "delta_points": {"type": "number", "description": "size of the shift in probability points, 0-10"},
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
        "big": {"type": "boolean", "description": "true only if this could move the race by 5+ points"},
        "reason": {"type": "string"},
        "sources": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["relevant", "direction", "delta_points", "confidence", "big", "reason", "sources"],
    "additionalProperties": False,
}

SYSTEM = """You assess election news for a prediction-market trading system covering the 2026 US midterms.
For each item, judge whether it changes the probability that the named market resolves YES, in which
direction, and by how many probability points. Be conservative: most headlines move nothing. Treat
polls from unknown or low-rated pollsters and early-vote turnout headlines as noise. Only cite sources
you actually read. Return JSON matching the schema."""


class ClaudeAnalyst:
    def __init__(self, cfg: AnalystConfig, race_briefs: str):
        if anthropic is None:
            raise RuntimeError("pip install anthropic (needs Python 3.10+) to use the analyst")
        self.cfg = cfg
        self.client = anthropic.AsyncAnthropic()
        self.briefs = race_briefs      # stable text, cached at the front of every call
        self._calls: List[float] = []

    def _budget_ok(self) -> bool:
        now = time.time()
        self._calls = [t for t in self._calls if t > now - 3600]
        return len(self._calls) < self.cfg.max_calls_per_hour

    async def _ask(self, market_id: str, market_title: str, item: str, effort: str,
                   with_search: bool) -> Optional[dict]:
        if not self._budget_ok():
            return None
        self._calls.append(time.time())
        tools = []
        if with_search:
            tools.append({"type": "web_search_20260209", "name": "web_search",
                          "allowed_domains": self.cfg.news_domains, "max_uses": 3})
        kwargs = {"tools": tools} if tools else {}
        resp = await self.client.beta.messages.create(
            **kwargs,
            model=self.cfg.model,
            max_tokens=4000,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            output_config={"effort": effort, "format": {"type": "json_schema", "schema": SIGNAL_SCHEMA}},
            system=[{"type": "text", "text": SYSTEM + "\n\n" + self.briefs,
                     "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user",
                       "content": f"Market {market_id}: {market_title}\nNews item: {item}"}],
        )
        if resp.stop_reason in ("refusal", "max_tokens", "pause_turn"):
            return None  # never trade on a partial or declined answer
        texts = [b.text for b in resp.content if getattr(b, "type", "") == "text"]
        if not texts:
            return None
        try:
            return json.loads(texts[-1])
        except json.JSONDecodeError:
            return None

    async def assess(self, market_id: str, market_title: str, item: str) -> Optional[Signal]:
        out = await self._ask(market_id, market_title, item, self.cfg.triage_effort, with_search=False)
        if out and out.get("big"):
            deep = await self._ask(market_id, market_title, item, self.cfg.review_effort, with_search=True)
            out = deep or out
        return to_signal(market_id, out, self.cfg.news_domains)


def to_signal(market_id: str, out: Optional[Dict], allowed_domains: List[str]) -> Optional[Signal]:
    """Validate the model's JSON and turn it into an (uncapped) Signal, or None."""
    if not out or not out.get("relevant") or out.get("direction") == "none":
        return None
    sources = [s for s in out.get("sources", []) if _allowed(s, allowed_domains)]
    if not sources:
        return None  # every shift must cite an allowed source
    pts = max(0.0, min(10.0, float(out.get("delta_points", 0))))
    conf_mult = {"low": 0.3, "medium": 0.6, "high": 1.0}.get(out.get("confidence"), 0.3)
    delta = pts / 100 * conf_mult * (1 if out["direction"] == "toward_yes" else -1)
    if abs(delta) < 0.005:
        return None
    return Signal(market_id, delta, out.get("confidence", "low"), out.get("reason", ""), sources)


def _allowed(url: str, domains: List[str]) -> bool:
    try:
        host = urlparse(url).hostname or ""
    except ValueError:
        return False
    return any(host == d or host.endswith("." + d) for d in domains)
