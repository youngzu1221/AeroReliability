import unittest

import numpy as np

from core.reliability import analyze_component, decision_from_metrics
from core.weibull_math import hazard, reliability


class ReliabilityTests(unittest.TestCase):
    def test_reliability_is_bounded(self):
        values = reliability(np.array([1.0, 10.0, 100.0]), 2.0, 50.0)
        self.assertTrue(np.all(values <= 1.0))
        self.assertTrue(np.all(values >= 0.0))

    def test_hazard_is_positive(self):
        values = hazard(np.array([1.0, 10.0, 100.0]), 2.0, 50.0)
        self.assertTrue(np.all(values > 0.0))

    def test_decision_levels(self):
        self.assertEqual(decision_from_metrics(0.7, 1000).level, "HIGH")
        self.assertEqual(decision_from_metrics(0.4, 1000).level, "MEDIUM")
        self.assertEqual(decision_from_metrics(0.1, 1000).level, "LOW")

    def test_component_analysis_produces_confidence_intervals_and_rpn(self):
        rng = np.random.default_rng(21)
        sample = np.sort(rng.weibull(2.3, 90) * 1100.0)
        result = analyze_component(
            "Pump A",
            sample,
            mttr=12.0,
            current_age=400.0,
            mission_time=150.0,
            preventive_cost=500.0,
            failure_cost=5000.0,
            severity=7,
            detectability=4,
        )
        self.assertLessEqual(result.beta_ci.lower, result.beta_ci.upper)
        self.assertLessEqual(result.eta_ci.lower, result.eta_ci.upper)
        self.assertLessEqual(result.rul_ci.lower, result.rul_ci.upper)
        self.assertEqual(len(result.distribution_fits), 7)
        self.assertEqual(result.severity, 7)
        self.assertEqual(result.detectability, 4)
        self.assertTrue(1 <= result.occurrence <= 10)
        self.assertEqual(result.rpn, result.severity * result.occurrence * result.detectability)


if __name__ == "__main__":
    unittest.main()
