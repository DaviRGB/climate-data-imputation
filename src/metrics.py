from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mse(y_true, y_pred)))


def mre(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    return float(np.sum(np.abs(y_true - y_pred)) / (np.sum(np.abs(y_true)) + eps))


def smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    denom = np.abs(y_true) + np.abs(y_pred) + eps
    return float(np.mean(2.0 * np.abs(y_true - y_pred) / denom) * 100.0)


def naive_scale_from_series(y_true: np.ndarray, eps: float = 1e-8) -> float:
    if len(y_true) < 2:
        return eps
    return float(np.mean(np.abs(np.diff(y_true))))


def mase(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    return float(mae(y_true, y_pred) / (naive_scale_from_series(y_true, eps) + eps))


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        'MAE': mae(y_true, y_pred),
        'MSE': mse(y_true, y_pred),
        'RMSE': rmse(y_true, y_pred),
        'MRE': mre(y_true, y_pred),
        'SMAPE': smape(y_true, y_pred),
        'MASE': mase(y_true, y_pred),
    }


def evaluate_global(x_true_targets: np.ndarray, x_pred_targets: np.ndarray, artificial_mask: np.ndarray) -> dict:
    y_true = x_true_targets[artificial_mask]
    y_pred = x_pred_targets[artificial_mask]
    return compute_all_metrics(y_true, y_pred)


def evaluate_per_variable(
    x_true_targets: np.ndarray,
    x_pred_targets: np.ndarray,
    artificial_mask: np.ndarray,
    target_names: Sequence[str],
) -> pd.DataFrame:
    rows = []
    for i, target_name in enumerate(target_names):
        mask_i = artificial_mask[:, :, i]
        y_true = x_true_targets[:, :, i][mask_i]
        y_pred = x_pred_targets[:, :, i][mask_i]
        if len(y_true) == 0:
            continue
        row = compute_all_metrics(y_true, y_pred)
        row['variable'] = target_name
        row['n_masked_points'] = int(mask_i.sum())
        rows.append(row)
    return pd.DataFrame(rows)


def build_result_tables(
    model_name: str,
    split_name: str,
    seed: int,
    scenario_name: str,
    x_true_targets: np.ndarray,
    x_pred_targets: np.ndarray,
    artificial_mask: np.ndarray,
    target_names: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    global_metrics = evaluate_global(x_true_targets, x_pred_targets, artificial_mask)
    global_df = pd.DataFrame([
        {
            'model': model_name,
            'split': split_name,
            'seed': seed,
            'scenario': scenario_name,
            **global_metrics,
            'n_masked_points': int(artificial_mask.sum()),
        }
    ])

    per_var_df = evaluate_per_variable(x_true_targets, x_pred_targets, artificial_mask, target_names)
    if not per_var_df.empty:
        per_var_df.insert(0, 'model', model_name)
        per_var_df.insert(1, 'split', split_name)
        per_var_df.insert(2, 'seed', seed)
        per_var_df.insert(3, 'scenario', scenario_name)
    return global_df, per_var_df
