from __future__ import annotations

from io import BytesIO

import pandas as pd


def build_template_workbook() -> bytes:
    example = pd.DataFrame(
        {
            "Pump A": [1200, 1450, 1680, 1710, 2100, 2380],
            "Motor B": [800, 950, 1010, 1250, 1360, 1510],
            "Bearing C": [300, 420, 510, 740, 860, 930],
        }
    )
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        example.to_excel(writer, index=False, sheet_name="failure_times")
    return buffer.getvalue()
