from __future__ import annotations

from io import BytesIO

import numpy as np
import pandas as pd

from core.validation import WorkbookParseResult, clean_life_data


def parse_excel(file_bytes: bytes) -> WorkbookParseResult:
    try:
        df = pd.read_excel(BytesIO(file_bytes))
    except Exception as exc:
        return WorkbookParseResult(datasets={}, warnings=(), error=f"Could not read the workbook: {exc}")

    if df.empty:
        return WorkbookParseResult(
            datasets={},
            warnings=(),
            error="The uploaded workbook is empty. Add at least one component column with positive numeric values.",
        )

    if len(df.columns) == 0:
        return WorkbookParseResult(
            datasets={},
            warnings=(),
            error="The uploaded workbook does not contain any columns.",
        )

    datasets: dict[str, np.ndarray] = {}
    warnings: list[str] = []

    for idx, col in enumerate(df.columns, start=1):
        column_name = str(col).strip()
        if not column_name or column_name.lower().startswith("unnamed"):
            warnings.append(f"Column {idx} has an empty or unnamed header and was skipped.")
            continue

        cleaned = clean_life_data(df[col])
        if len(cleaned) < 2:
            warnings.append(f"{column_name} was skipped because it has fewer than two positive numeric values.")
            continue

        datasets[column_name] = cleaned

    if not datasets:
        return WorkbookParseResult(
            datasets={},
            warnings=tuple(warnings),
            error="No usable component data was found. Each component column needs at least two positive numeric values.",
        )

    return WorkbookParseResult(datasets=datasets, warnings=tuple(warnings), error=None)


def serialize_datasets(datasets: dict[str, np.ndarray]) -> tuple[tuple[str, tuple[float, ...]], ...]:
    return tuple((name, tuple(map(float, data))) for name, data in sorted(datasets.items()))


def deserialize_datasets(payload: tuple[tuple[str, tuple[float, ...]], ...]) -> dict[str, np.ndarray]:
    return {name: np.asarray(values, dtype=float) for name, values in payload}
