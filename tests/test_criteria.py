import unittest

from dynamic_ies.criteria import MasteryTracker, finite_difference, moving_average_abs


class CriterionTests(unittest.TestCase):
    def test_finite_differences(self):
        self.assertAlmostEqual(finite_difference([3.0, 2.0], order=1), -1.0)
        self.assertAlmostEqual(finite_difference([4.0, 2.0, 1.0], order=2), 1.0)

    def test_moving_average_uses_absolute_values(self):
        self.assertEqual(moving_average_abs([-1.0, 3.0, -2.0], 2), [2.0, 2.5])

    def test_flat_loss_trajectory_becomes_mastered(self):
        tracker = MasteryTracker(threshold=1e-3, smoothing_window=3)
        for loss in [1.0] * 5:
            tracker.observe(7, loss)
        self.assertTrue(tracker.is_mastered(7))

    def test_curved_loss_trajectory_is_not_mastered(self):
        tracker = MasteryTracker(threshold=1e-3, smoothing_window=2)
        for loss in [1.0, 0.8, 0.5, 0.1]:
            tracker.observe(7, loss)
        self.assertFalse(tracker.is_mastered(7))


if __name__ == "__main__":
    unittest.main()
