import unittest

from dynamic_ies.schedules import (
    AdaptiveRatioSchedule,
    period_for_epoch,
    period_from_learning_rate,
)


class EpochScheduleTests(unittest.TestCase):
    def test_constant_schedule(self):
        self.assertEqual(period_for_epoch(100, "constant", constant_period=3), 3)

    def test_sqrt_schedule_is_bounded(self):
        self.assertEqual(period_for_epoch(0, "sqrt"), 1)
        self.assertEqual(period_for_epoch(10_000, "sqrt"), 5)

    def test_lr_schedule_matches_experimental_bins(self):
        self.assertEqual(period_from_learning_rate(0.1, "linear"), 1)
        self.assertEqual(period_from_learning_rate(0.03, "linear"), 2)
        self.assertEqual(period_from_learning_rate(0.007, "linear"), 4)
        self.assertEqual(period_from_learning_rate(0.0005, "linear"), 5)

    def test_adaptive_ratio_reacts_to_stability(self):
        schedule = AdaptiveRatioSchedule(threshold=0.02)
        self.assertEqual(schedule.update(0.10), 1)
        self.assertEqual(schedule.update(0.11), 2)
        self.assertEqual(schedule.update(0.20), 1)


if __name__ == "__main__":
    unittest.main()
