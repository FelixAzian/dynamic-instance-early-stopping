"""Per-instance mastery criterion based on finite loss differences."""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Iterable
from dataclasses import dataclass, field


def finite_difference(values: Iterable[float], order: int = 2) -> float:
    """Calculate the latest first- or second-order finite difference."""
    sequence = list(values)
    if order == 1:
        if len(sequence) < 2:
            raise ValueError("first-order difference requires two values")
        return sequence[-1] - sequence[-2]
    if order == 2:
        if len(sequence) < 3:
            raise ValueError("second-order difference requires three values")
        return sequence[-1] - 2 * sequence[-2] + sequence[-3]
    raise ValueError("only first- and second-order differences are supported")


def moving_average_abs(values: Iterable[float], window: int) -> list[float]:
    """Return the moving average of absolute values over complete windows."""
    sequence = [abs(value) for value in values]
    if window < 1:
        raise ValueError("window must be positive")
    if len(sequence) < window:
        return []

    window_sum = sum(sequence[:window])
    averages = [window_sum / window]
    for index in range(window, len(sequence)):
        window_sum += sequence[index] - sequence[index - window]
        averages.append(window_sum / window)
    return averages


@dataclass
class MasteryTracker:
    """Track loss dynamics and identify samples satisfying the IES criterion."""

    threshold: float = 1e-3
    derivative_order: int = 2
    smoothing_window: int = 3
    criterion_window: int = 1
    _losses: dict[int, deque[float]] = field(default_factory=lambda: defaultdict(deque))
    _differences: dict[int, deque[float]] = field(default_factory=lambda: defaultdict(deque))

    def __post_init__(self) -> None:
        if self.threshold < 0:
            raise ValueError("threshold must be non-negative")
        if self.derivative_order not in (1, 2):
            raise ValueError("derivative_order must be 1 or 2")
        if self.smoothing_window < 1 or self.criterion_window < 1:
            raise ValueError("window sizes must be positive")

    def observe(self, sample_id: int, loss: float) -> None:
        """Add a new loss observation for one sample."""
        history = self._losses[sample_id]
        history.append(float(loss))
        while len(history) > self.derivative_order + 1:
            history.popleft()

        if len(history) == self.derivative_order + 1:
            differences = self._differences[sample_id]
            differences.append(finite_difference(history, self.derivative_order))
            keep = self.smoothing_window + self.criterion_window - 1
            while len(differences) > keep:
                differences.popleft()

    def is_mastered(self, sample_id: int) -> bool:
        """Return whether a sample's recent smoothed dynamics are below threshold."""
        smoothed = moving_average_abs(
            self._differences.get(sample_id, ()), self.smoothing_window
        )
        if len(smoothed) < self.criterion_window:
            return False
        return sum(smoothed[-self.criterion_window :]) < self.threshold

    def mastered(self, sample_ids: Iterable[int]) -> set[int]:
        return {sample_id for sample_id in sample_ids if self.is_mastered(sample_id)}

    def switch_to_first_order(self, threshold: float) -> None:
        """Switch criteria and reset incompatible derivative histories."""
        if threshold < 0:
            raise ValueError("threshold must be non-negative")
        self.derivative_order = 1
        self.threshold = threshold
        self._differences.clear()
