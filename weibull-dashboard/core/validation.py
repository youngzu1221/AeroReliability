from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class WorkbookParseResult:
    datasets: dict[str, np.ndarray]
    warnings: tuple[str, ...]
    error: str | None = None


def clean_life_data(values: pd.Series | np.ndarray) -> np.ndarray:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    arr = arr[arr > 0]
    return np.sort(arr)
