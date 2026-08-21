from __future__ import annotations

import math


def calculate_mtbur(
    total_flight_exposure: float,
    removals: int,
    quantity_per_aircraft: float,
) -> float:
    """Return MTBUR using total exposure * QPA / number of removals."""
    total = float(total_flight_exposure)
    qpa = float(quantity_per_aircraft)
    removal_count = int(removals)

    if not math.isfinite(total) or total < 0.0:
        raise ValueError("Total flight exposure must be a finite value of zero or more.")
    if not math.isfinite(qpa) or qpa <= 0.0:
        raise ValueError("Quantity per aircraft must be a finite value greater than zero.")
    if removal_count <= 0:
        raise ValueError("Number of removals must be greater than zero.")

    return total * qpa / removal_count
