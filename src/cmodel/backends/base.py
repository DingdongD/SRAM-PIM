"""Strict systolic backend interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..ir import SystolicInvocation, SystolicResult


class SystolicBackend(ABC):
    @abstractmethod
    def run(self, invocation: SystolicInvocation) -> SystolicResult:
        """Return a validated cycle result and cycle-tagged operand demands."""
