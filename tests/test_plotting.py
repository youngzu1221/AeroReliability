import unittest

import matplotlib.pyplot as plt
import numpy as np

from core.reliability import analyze_component
from plotting.distribution_plots import build_forward_risk_figure
from plotting.styling import apply_axis_bounds_and_units


class PlottingTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_apply_axis_bounds_and_units_recovers_from_non_finite_limits(self):
        fig, ax = plt.subplots()
        apply_axis_bounds_and_units(ax, {}, (0.0, 10.0), (float("nan"), float("inf")))
        self.assertEqual(tuple(round(value, 6) for value in ax.get_ylim()), (0.0, 1.0))

    def test_forward_risk_figure_handles_early_life_weibull_without_crashing(self):
        rng = np.random.default_rng(17)
        sample = np.sort(rng.weibull(0.7, 80) * 1400.0)
        result = analyze_component(
            "Brake A",
            sample,
            mttr=0.0,
            current_age=0.0,
            mission_time=500.0,
            preventive_cost=500.0,
            failure_cost=5000.0,
            severity=5,
            detectability=5,
            selected_distribution="Weibull",
        )
        fig = build_forward_risk_figure(result, 0.0, 500.0, "None", 0.0, 0.0, {})
        self.assertEqual(len(fig.axes), 3)


if __name__ == "__main__":
    unittest.main()
