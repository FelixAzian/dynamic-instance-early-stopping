"""Forward-pass scheduling policies used in the 2025 USYD experiments."""

from __future__ import annotations

import math
from dataclasses import dataclass, field


VALID_EPOCH_SCHEDULES = {"constant", "linear", "log", "sqrt"}
VALID_LR_DECAYS = {"linear", "exponential"}


def period_for_epoch(
    epoch: int,
    mode: str,
    *,
    constant_period: int = 1,
    max_period: int = 5,
) -> int:
    """Return a bounded re-evaluation period for a zero-indexed epoch.

    The coefficients reproduce the schedules evaluated in the project report.
    """
    if epoch < 0:
        raise ValueError("epoch must be non-negative")
    if constant_period < 1 or max_period < 1:
        raise ValueError("periods must be positive")
    if mode not in VALID_EPOCH_SCHEDULES:
        raise ValueError(f"unknown epoch schedule: {mode}")

    if mode == "constant":
        value = constant_period
    elif mode == "linear":
        value = math.floor(0.05 * epoch)
    elif mode == "log":
        value = math.floor(2.5 * math.log10(epoch + 1))
    else:
        value = math.floor(0.6 * math.sqrt(epoch + 1))

    return min(max_period, max(1, value))


def period_from_learning_rate(
    learning_rate: float,
    decay: str,
    *,
    max_period: int = 5,
) -> int:
    """Map optimizer learning rate to the piecewise period used in experiments."""
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if decay not in VALID_LR_DECAYS:
        raise ValueError(f"unknown learning-rate decay: {decay}")
    if max_period < 1:
        raise ValueError("max_period must be positive")

    if decay == "linear":
        boundaries = ((0.05, 1), (0.02, 2), (0.01, 3), (0.005, 4))
    else:
        boundaries = ((0.05, 1), (0.01, 2), (0.005, 3), (0.001, 4))

    for lower_bound, period in boundaries:
        if learning_rate >= lower_bound:
            return min(period, max_period)
    return max_period


@dataclass
class AdaptiveRatioSchedule:
    """Feedback schedule driven by changes in the mastered-sample ratio.

    Call :meth:`update` after each measurement window. Stable ratios increase
    the interval; larger changes decrease it so mastered samples are checked
    more frequently.
    """

    threshold: float = 0.02
    max_period: int = 5
    period: int = 1
    history: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.threshold < 0:
            raise ValueError("threshold must be non-negative")
        if not 1 <= self.period <= self.max_period:
            raise ValueError("period must be between 1 and max_period")

    def update(self, mastered_ratio: float) -> int:
        if not 0 <= mastered_ratio <= 1:
            raise ValueError("mastered_ratio must be in [0, 1]")

        self.history.append(mastered_ratio)
        if len(self.history) < 2:
            return self.period

        delta = abs(self.history[-1] - self.history[-2])
        if delta < self.threshold:
            self.period = min(self.max_period, self.period + 1)
        else:
            self.period = max(1, self.period - 1)
        return self.period
