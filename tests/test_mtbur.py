import unittest

from core.mtbur import calculate_mtbur


class MtburTests(unittest.TestCase):
    def test_calculates_mtbur_from_flight_hours(self):
        self.assertEqual(calculate_mtbur(12_000.0, 3, 4), 16_000.0)

    def test_allows_zero_exposure(self):
        self.assertEqual(calculate_mtbur(0.0, 1, 2), 0.0)

    def test_rejects_zero_removals(self):
        with self.assertRaises(ValueError):
            calculate_mtbur(12_000.0, 0, 4)


if __name__ == "__main__":
    unittest.main()
