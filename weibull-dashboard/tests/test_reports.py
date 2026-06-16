import unittest

import pandas as pd

from core.optimization import DistributionFit
from reports.table_formatter import distribution_comparison_frame, formatted_results_frame, highest_risk_component_text


class ReportsTests(unittest.TestCase):
    def test_formatted_results_frame_formats_money_and_percent(self):
        df = pd.DataFrame(
            [
                {
                    "Component": "Pump A",
                    "Selected Distribution": "Weibull",
                    "Parameter 1 Label": "beta",
                    "Parameter 1": 2.3,
                    "Parameter 1 CI Lower": 2.1,
                    "Parameter 1 CI Upper": 2.5,
                    "Parameter 2 Label": "eta",
                    "Parameter 2": 1200.0,
                    "Parameter 2 CI Lower": 1100.0,
                    "Parameter 2 CI Upper": 1300.0,
                    "Characteristic Value": 1200.0,
                    "MTTF": 1100.0,
                    "MTBF": 1120.0,
                    "Conditional Reliability": 0.25,
                    "Conditional Reliability CI Lower": 0.20,
                    "Conditional Reliability CI Upper": 0.30,
                    "Conditional Probability of Failure": 0.75,
                    "Failure Probability CI Lower": 0.70,
                    "Failure Probability CI Upper": 0.80,
                    "RUL": 450.0,
                    "RUL CI Lower": 400.0,
                    "RUL CI Upper": 500.0,
                    "Optimal Replacement": 900.0,
                    "Min Cost Rate": 12.5,
                    "Failure Mode": "Wear-out",
                    "Risk": "HIGH",
                    "Best Fit Distribution": "Weibull",
                    "Best Fit AIC": 100.0,
                    "Best Fit BIC": 110.0,
                    "Best Fit RMSE": 0.042,
                    "Severity": 5,
                    "Occurrence": 8,
                    "Detectability": 4,
                    "RPN": 160,
                }
            ]
        )
        formatted = formatted_results_frame(df)
        self.assertEqual(formatted.loc[0, "Conditional Reliability"], "25.00%")
        self.assertEqual(formatted.loc[0, "Min Cost Rate"], "$12.50")
        self.assertEqual(formatted.loc[0, "Parameter 1 95% CI"], "2.10 to 2.50")
        self.assertNotIn("Parameter 1 CI Lower", formatted.columns)

    def test_highest_risk_component_text_joins_multiple_high_risk_components(self):
        df = pd.DataFrame(
            [
                {"Component": "A", "Conditional Probability of Failure": 0.9, "Risk": "HIGH"},
                {"Component": "B", "Conditional Probability of Failure": 0.85, "Risk": "HIGH"},
            ]
        )
        self.assertEqual(highest_risk_component_text(df), "A, B")

    def test_distribution_comparison_frame_formats_parameters(self):
        fit = DistributionFit(
            name="Weibull",
            params=(2.3, 1200.0),
            param_labels=("beta", "eta"),
            log_likelihood=-50.0,
            aic=104.0,
            bic=108.0,
            fit_score=0.965,
            rmse=0.042,
            ks_statistic=0.08,
            ks_pvalue=0.91,
            ad_statistic=0.35,
            ad_pvalue=0.72,
        )
        frame = distribution_comparison_frame((fit,))
        self.assertEqual(frame.loc[0, "Distribution"], "Weibull")
        self.assertEqual(frame.loc[0, "Best Fit"], "YES")
        self.assertEqual(frame.loc[0, "Fit % (R^2-like)"], "96.50%")
        self.assertEqual(frame.loc[0, "RMSE"], "0.0420")
        self.assertIn("beta=2.300", frame.loc[0, "Parameters"])


if __name__ == "__main__":
    unittest.main()
