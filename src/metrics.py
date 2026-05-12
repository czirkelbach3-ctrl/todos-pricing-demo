"""Forecast-accuracy metrics."""
from __future__ import annotations

import numpy as np


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    # MAPE: only on non-zero actuals (industry-standard treatment)
    mask = y_true > 0
    mape = float(np.mean(np.abs(err[mask]) / y_true[mask])) if mask.any() else float("nan")

    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2)) if len(y_true) > 1 else 1.0
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2}


def incremental(plan: float, baseline: float) -> dict[str, float]:
    delta = plan - baseline
    pct = (delta / baseline) if baseline else 0.0
    return {"delta": float(delta), "pct": float(pct)}
