import unittest

import numpy as np

from core.optimization import bootstrap_weibull_parameters, estimate_weibull_mle, fit_distribution_models
from core.weibull_math import weibull_cdf, weibull_pdf, weibull_quantile


class WeibullMathTests(unittest.TestCase):
    def test_pdf_is_non_negative(self):
        values = weibull_pdf(np.array([1.0, 5.0, 10.0]), 2.0, 8.0)
        self.assertTrue(np.all(values >= 0.0))

    def test_cdf_is_bounded(self):
        values = weibull_cdf(np.array([1.0, 5.0, 10.0]), 2.0, 8.0)
        self.assertTrue(np.all(values >= 0.0))
        self.assertTrue(np.all(values <= 1.0))

    def test_quantile_is_positive(self):
        self.assertGreater(weibull_quantile(0.5, 2.0, 1000.0), 0.0)

    def test_mle_recovers_reasonable_parameters(self):
        rng = np.random.default_rng(42)
        sample = np.sort(rng.weibull(2.4, 500) * 1200.0)
        beta, eta = estimate_weibull_mle(sample)
        self.assertAlmostEqual(beta, 2.4, delta=0.35)
        self.assertAlmostEqual(eta, 1200.0, delta=180.0)

    def test_distribution_comparison_returns_expected_models(self):
        rng = np.random.default_rng(7)
        sample = np.sort(rng.weibull(2.1, 700) * 900.0)
        fits = fit_distribution_models(sample)
        self.assertEqual(
            {fit.name for fit in fits},
            {"Weibull", "Lognormal", "Normal", "Gamma", "Gumbel", "Rayleigh", "Exponential"},
        )
        self.assertEqual(fits[0].name, "Weibull")
        self.assertGreaterEqual(fits[0].rmse, 0.0)

    def test_bootstrap_returns_parameter_samples(self):
        rng = np.random.default_rng(9)
        sample = np.sort(rng.weibull(2.0, 120) * 500.0)
        beta_samples, eta_samples = bootstrap_weibull_parameters(sample, resamples=20, random_seed=123)
        self.assertGreaterEqual(len(beta_samples), 1)
        self.assertEqual(len(beta_samples), len(eta_samples))


if __name__ == "__main__":
    unittest.main()
