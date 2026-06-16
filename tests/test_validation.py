import unittest
from io import BytesIO

import numpy as np
import pandas as pd

from core.validation import clean_life_data
from data.excel_parser import parse_excel


class ValidationTests(unittest.TestCase):
    def test_clean_life_data_removes_invalid_values(self):
        values = pd.Series([5, None, "bad", -1, 3, 0, 8])
        cleaned = clean_life_data(values)
        np.testing.assert_allclose(cleaned, np.array([3.0, 5.0, 8.0]))

    def test_parse_excel_returns_error_for_empty_sheet(self):
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            pd.DataFrame().to_excel(writer, index=False)
        result = parse_excel(buffer.getvalue())
        self.assertIsNotNone(result.error)

    def test_parse_excel_accepts_valid_component_column(self):
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            pd.DataFrame({"Pump A": [100, 120, 140], "Bad": [None, 0, -1]}).to_excel(writer, index=False)
        result = parse_excel(buffer.getvalue())
        self.assertIsNone(result.error)
        self.assertIn("Pump A", result.datasets)
        self.assertNotIn("Bad", result.datasets)


if __name__ == "__main__":
    unittest.main()
