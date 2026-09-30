"""Data feed interface the engine reads from. The simulator and the live bot
each provide an implementation."""

from __future__ import annotations

import abc
from typing import Dict, List, Optional, Tuple

from .election_night import NightParams, RaceReport
from .types import Signal


class Feeds(abc.ABC):
    @abc.abstractmethod
    def now(self) -> float:
        """Current time in seconds (sim time in the simulator)."""

    @abc.abstractmethod
    def phase(self) -> str:
        """'pre' | 'final_week' | 'night' | 'closed'."""

    @abc.abstractmethod
    def anchors(self) -> Dict[str, float]:
        """market_id -> P(YES) from external real-money venues."""

    @abc.abstractmethod
    def poll_means(self) -> Dict[str, float]:
        """race_id -> expected D margin (points) for the race model."""

    def signals(self) -> List[Signal]:
        """New news signals since the last call (already structured, not yet capped)."""
        return []

    def live_reports(self) -> Optional[Dict[str, "RaceReport"]]:
        """Election night: race_id -> RaceReport(state, rho, counted_margin). None before results."""
        return None

    def night_params(self) -> Optional["NightParams"]:
        """Election-night model parameters (mirage table etc.). None -> defaults."""
        return None

    def call_book(self):
        """Election night: a calls.CallBook of race calls from call desks, or None."""
        return None
