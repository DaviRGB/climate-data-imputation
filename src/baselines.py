from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer
from tqdm import tqdm

from typing import Dict

try:
    import cupy as cp
except ImportError:
    cp = None

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None

@dataclass
class BaselineResult:
    name: str
    X_imputed: np.ndarray


def _check_3d(x: np.ndarray) -> None:
    if x.ndim != 3:
        raise ValueError('X deve ter shape [N, T, F].')


def _flatten_3d(x: np.ndarray) -> np.ndarray:
    _check_3d(x)
    n_samples, n_steps, n_features = x.shape
    return x.reshape(n_samples * n_steps, n_features)


def _safe_row_subsample(flat: np.ndarray, max_rows: int | None, seed: int = 42) -> np.ndarray:
    if max_rows is None or max_rows <= 0 or flat.shape[0] <= max_rows:
        return flat
    rng = np.random.default_rng(seed)
    idx = rng.choice(flat.shape[0], size=max_rows, replace=False)
    idx.sort()
    return flat[idx]


def compute_feature_medians_from_train(x_train: np.ndarray) -> np.ndarray:
    _check_3d(x_train)
    flat = x_train.reshape(-1, x_train.shape[-1])
    medians = np.nanmedian(flat, axis=0)
    return np.where(np.isnan(medians), 0.0, medians)


def median_imputation(x_masked: np.ndarray, train_feature_medians: np.ndarray) -> BaselineResult:
    _check_3d(x_masked)
    x_out = x_masked.copy()
    for f in range(x_out.shape[-1]):
        mask = np.isnan(x_out[:, :, f])
        x_out[:, :, f][mask] = train_feature_medians[f]
    return BaselineResult(name='Median', X_imputed=x_out)


def locf_imputation(x_masked: np.ndarray) -> BaselineResult:
    _check_3d(x_masked)
    n_samples, n_steps, n_features = x_masked.shape
    x_out = x_masked.copy()
    for i in tqdm(range(n_samples), desc='LOCF', unit='janela'):
        for f in range(n_features):
            series = x_out[i, :, f].copy()
            last_val = np.nan
            for t in range(n_steps):
                if np.isnan(series[t]) and not np.isnan(last_val):
                    series[t] = last_val
                elif not np.isnan(series[t]):
                    last_val = series[t]
            next_val = np.nan
            for t in range(n_steps - 1, -1, -1):
                if np.isnan(series[t]) and not np.isnan(next_val):
                    series[t] = next_val
                elif not np.isnan(series[t]):
                    next_val = series[t]
            x_out[i, :, f] = series
    return BaselineResult(name='LOCF', X_imputed=x_out)


def locf_then_median_imputation(x_masked: np.ndarray, train_feature_medians: np.ndarray) -> BaselineResult:
    x_out = locf_imputation(x_masked).X_imputed
    for f in range(x_out.shape[-1]):
        mask = np.isnan(x_out[:, :, f])
        x_out[:, :, f][mask] = train_feature_medians[f]
    return BaselineResult(name='LOCFThenMedian', X_imputed=x_out)


def fit_knn_imputer_from_train(
    x_train: np.ndarray,
    n_neighbors: int = 5,
    weights: str = 'uniform',
    max_rows_fit: int | None = 100_000,
    seed: int = 42,
) -> KNNImputer:
    flat_train = _safe_row_subsample(_flatten_3d(x_train), max_rows_fit, seed)
    print(f'[KNN][FIT] linhas usadas: {flat_train.shape[0]:,}')
    imputer = KNNImputer(n_neighbors=n_neighbors, weights=weights)
    imputer.fit(flat_train)
    return imputer


def transform_with_knn_imputer(
    x_masked: np.ndarray,
    imputer: KNNImputer,
    n_neighbors: int = 5,
    batch_size: int = 2_000,
) -> BaselineResult:
    _check_3d(x_masked)
    n_samples, n_steps, n_features = x_masked.shape
    flat = _flatten_3d(x_masked)
    chunks = []
    t0 = time.time()
    ranges = [(i, min(i + batch_size, flat.shape[0])) for i in range(0, flat.shape[0], batch_size)]
    for batch_idx, (i0, i1) in enumerate(ranges, start=1):
        t_batch = time.time()
        print(f'[KNN] lote {batch_idx}/{len(ranges)} | linhas {i0:,}:{i1:,}')
        chunks.append(imputer.transform(flat[i0:i1]))
        print(f'[KNN] lote {batch_idx} concluído em {time.time() - t_batch:.2f}s')
    flat_imputed = np.concatenate(chunks, axis=0)
    print(f'[KNN] transform concluído em {time.time() - t0:.2f}s')
    return BaselineResult(name=f'KNNImputer_k{n_neighbors}', X_imputed=flat_imputed.reshape(n_samples, n_steps, n_features))


def fit_missforest_like_imputer_from_train(
    x_train: np.ndarray,
    random_state: int = 42,
    n_estimators: int = 100,
    max_iter: int = 6,
    max_depth: int = 5,
    max_features: str = 'sqrt',
    min_samples_split: int = 2,
    bootstrap: bool = False,
    n_jobs: int = 1,
    tol: float = 1e-2,
    n_nearest_features: int | None = 10,
    max_rows_fit: int | None = 50_000,
) -> IterativeImputer:
    flat_train = _safe_row_subsample(_flatten_3d(x_train), max_rows_fit, random_state)
    print(f'[MissForestLike][FIT] linhas usadas: {flat_train.shape[0]:,}')
    estimator = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=n_jobs, max_depth=max_depth, max_features=max_features, min_samples_split=min_samples_split, bootstrap=bootstrap)
    imputer = IterativeImputer(
        estimator=estimator,
        max_iter=max_iter,
        tol=tol,
        n_nearest_features=n_nearest_features,
        random_state=random_state,
        initial_strategy='median',
        sample_posterior=False,
        skip_complete=True,
        imputation_order='ascending',
        keep_empty_features=True,
        verbose=2,
    )
    imputer.fit(flat_train)
    return imputer


def transform_with_missforest_like_imputer(
    x_masked: np.ndarray,
    imputer: IterativeImputer,
    batch_size: int = 2_000,
) -> BaselineResult:
    _check_3d(x_masked)
    n_samples, n_steps, n_features = x_masked.shape
    flat = _flatten_3d(x_masked)
    chunks = []
    t0 = time.time()
    ranges = [(i, min(i + batch_size, flat.shape[0])) for i in range(0, flat.shape[0], batch_size)]
    for batch_idx, (i0, i1) in enumerate(ranges, start=1):
        t_batch = time.time()
        print(f'[MissForestLike] lote {batch_idx}/{len(ranges)} | linhas {i0:,}:{i1:,}')
        chunks.append(imputer.transform(flat[i0:i1]))
        print(f'[MissForestLike] lote {batch_idx} concluído em {time.time() - t_batch:.2f}s')
    flat_imputed = np.concatenate(chunks, axis=0)
    print(f'[MissForestLike] transform concluído em {time.time() - t0:.2f}s')
    return BaselineResult(name='MissForestLike', X_imputed=flat_imputed.reshape(n_samples, n_steps, n_features))


def fit_xgboost_per_target_from_train(
    X_train: np.ndarray,
    X_val_masked: np.ndarray,
    X_true_val_targets_scaled: np.ndarray,
    val_artificial_mask: np.ndarray,
    n_targets: int,
    random_state: int = 42,
    device: str = "cuda",
    n_estimators: int = 100,
    max_depth: int = 4,
    learning_rate: float = 0.05,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    reg_alpha: float = 0.0,
    reg_lambda: float = 1.0,
    n_jobs: int = 1,
    early_stopping_rounds: int = 20,
    eval_metric: str = "rmse",
) -> Dict[int, object]:
    _check_3d(X_train)
    _check_3d(X_val_masked)
    _check_3d(X_true_val_targets_scaled)
    
    if XGBRegressor is None:
        raise ImportError("xgboost não está instalado.")
    
    usar_gpu = isinstance(device, str) and device.startswith("cuda")
    if usar_gpu and cp is None:
        raise ImportError("cupy não está instalado. Para usar XGBoost na GPU, instale o cupy.")

    flat_train = _flatten_3d(X_train)
    flat_val = _flatten_3d(X_val_masked)

    y_val_targets = X_true_val_targets_scaled.reshape(-1, n_targets)
    val_mask_flat = val_artificial_mask.reshape(-1, n_targets).astype(bool)

    models = {}

    print(f"[XGB] Ajustando {n_targets} modelos com early stopping...")

    for target_idx in range(n_targets):

        y_train = flat_train[:, target_idx]
        obs_mask_train = ~np.isnan(y_train)

        if obs_mask_train.sum() == 0:
            print(f"[XGB][WARN] alvo {target_idx} sem observações no treino. Pulando.")
            continue

        X_train_target = np.delete(flat_train[obs_mask_train], target_idx, axis=1)
        y_train_target = y_train[obs_mask_train]

        eval_rows = val_mask_flat[:, target_idx]

        if eval_rows.sum() == 0:
            X_val_target = None
            y_val_target = None
            use_early_stopping = False
            print(f"[XGB][WARN] alvo {target_idx} sem pontos artificiais na validação. Treino sem early stopping.")
        else:
            X_val_target = np.delete(flat_val[eval_rows], target_idx, axis=1)
            y_val_target = y_val_targets[eval_rows, target_idx]
            
            valid_eval = ~np.isnan(y_val_target)
            X_val_target = X_val_target[valid_eval]
            y_val_target = y_val_target[valid_eval]

            use_early_stopping = len(y_val_target) > 0

        t0 = time.time()

        model = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            tree_method="hist",
            device=device,
            random_state=random_state,
            n_jobs=n_jobs,
            eval_metric=eval_metric,
            early_stopping_rounds=early_stopping_rounds if use_early_stopping else None,
        )

        if use_early_stopping:
            model.fit(
                X_train_target,
                y_train_target,
                eval_set=[(X_val_target, y_val_target)],
                verbose=False,
            )
            best_iter = getattr(model, "best_iteration", None)
            print(
                f"[XGB] alvo={target_idx} | treino={len(y_train_target):,} | "
                f"eval={len(y_val_target):,} | best_iteration={best_iter} | "
                f"fit em {time.time() - t0:.2f}s"
            )
        else:
            model.fit(X_train_target, y_train_target, verbose=False)
            print(
                f"[XGB] alvo={target_idx} | treino={len(y_train_target):,} | "
                f"sem early stopping | fit em {time.time() - t0:.2f}s"
            )

        models[target_idx] = model

    return models

def transform_with_xgboost_per_target(
    X_masked: np.ndarray,
    models: Dict[int, object],
    n_targets: int,
    predict_device: str = "cuda",
) -> BaselineResult:
    _check_3d(X_masked)

    n_samples, n_steps, n_features = X_masked.shape
    flat = X_masked.reshape(n_samples * n_steps, n_features).copy()

    start_time = time.time()
    print(f"[XGB] Iniciando imputação por alvo em {flat.shape[0]:,} linhas...")

    usar_gpu_pred = (
        isinstance(predict_device, str)
        and predict_device.startswith("cuda")
        and cp is not None
    )

    for target_idx in range(n_targets):
        if target_idx not in models:
            print(f"[XGB][WARN] Sem modelo para alvo {target_idx}. Pulando.")
            continue

        miss_mask = np.isnan(flat[:, target_idx])
        n_missing = int(miss_mask.sum())

        if n_missing == 0:
            print(f"[XGB] alvo={target_idx} sem faltantes neste split.")
            continue

        model = models[target_idx]
        X_pred_cpu = np.delete(flat[miss_mask], target_idx, axis=1).astype(np.float32, copy=False)
        best_iter = getattr(model, "best_iteration", None)

        if usar_gpu_pred:
            booster = model.get_booster()
            booster.set_param({"device": predict_device})
            X_pred_gpu = cp.asarray(X_pred_cpu)

            if best_iter is not None:
                preds_gpu = booster.inplace_predict(
                    X_pred_gpu,
                    iteration_range=(0, best_iter + 1),
                )
            else:
                preds_gpu = booster.inplace_predict(X_pred_gpu)

            preds = cp.asnumpy(preds_gpu)
        else:
            if best_iter is not None:
                preds = model.predict(X_pred_cpu, iteration_range=(0, best_iter + 1))
            else:
                preds = model.predict(X_pred_cpu)

        flat[miss_mask, target_idx] = preds
        print(f"[XGB] alvo={target_idx} | imputados={n_missing:,}")

    X_out = flat.reshape(n_samples, n_steps, n_features)
    print(f"[XGB] imputação concluída em {time.time() - start_time:.2f}s.")
    return BaselineResult(name="XGBoostPerTarget", X_imputed=X_out)

# def fit_xgboost_multioutput_from_train(
#     X_train: np.ndarray,
#     X_val_masked: np.ndarray,
#     X_true_val_targets: np.ndarray,
#     val_artificial_mask: np.ndarray,
#     n_targets: int,
#     random_state: int = 42,
#     device: str = "cuda",
#     n_estimators: int = 100,
#     max_depth: int = 4,
#     learning_rate: float = 0.05,
#     subsample: float = 0.8,
#     colsample_bytree: float = 0.8,
#     reg_alpha: float = 0.0,
#     reg_lambda: float = 1.0,
#     n_jobs: int = 1,
#     early_stopping_rounds: int = 20,
#     eval_metric: str = "rmse",
#     multi_strategy: str = "multi_output_tree",
# ):
#     _check_3d(X_train)
#     _check_3d(X_val_masked)

#     if XGBRegressor is None:
#         raise ImportError("xgboost não está instalado.")

#     flat_train = _flatten_3d(X_train)
#     flat_val = _flatten_3d(X_val_masked)

#     # y do treino: só as colunas alvo (INMET)
#     y_train = flat_train[:, :n_targets]
#     y_val_true = X_true_val_targets.reshape(-1, n_targets)
#     val_mask_flat = val_artificial_mask.reshape(-1, n_targets).astype(bool)

#     # Linhas sem NaN nos alvos
#     train_rows = ~np.isnan(y_train).any(axis=1)
    
#     X_train_multi = flat_train[train_rows, n_targets:]
#     y_train_multi = y_train[train_rows]

#     eval_rows = val_mask_flat.any(axis=1)
#     valid_y_rows = ~np.isnan(y_val_true).any(axis=1)
#     eval_rows = eval_rows & valid_y_rows

#     X_val_multi = flat_val[eval_rows, n_targets:]
#     y_val_multi = y_val_true[eval_rows]

#     use_early_stopping = len(y_val_multi) > 0

#     print(
#         f"[XGB-MO] treino={len(y_train_multi):,} | "
#         f"eval={len(y_val_multi):,} | "
#         f"multi_strategy={multi_strategy}"
#     )

#     model = XGBRegressor(
#         objective="reg:squarederror",
#         n_estimators=n_estimators,
#         max_depth=max_depth,
#         learning_rate=learning_rate,
#         subsample=subsample,
#         colsample_bytree=colsample_bytree,
#         reg_alpha=reg_alpha,
#         reg_lambda=reg_lambda,
#         tree_method="hist",
#         device=device,
#         random_state=random_state,
#         n_jobs=n_jobs,
#         eval_metric=eval_metric,
#         early_stopping_rounds=early_stopping_rounds if use_early_stopping else None,
#         multi_strategy=multi_strategy,
#     )

#     t0 = time.time()
#     if use_early_stopping:
#         model.fit(
#             X_train_multi,
#             y_train_multi,
#             eval_set=[(X_val_multi, y_val_multi)],
#             verbose=False,
#         )
#     else:
#         model.fit(X_train_multi, y_train_multi, verbose=False)

#     best_iter = getattr(model, "best_iteration", None)
#     print(
#         f"[XGB-MO] fit concluído em {time.time() - t0:.2f}s | "
#         f"best_iteration={best_iter}"
#     )
#     return model


# def transform_with_xgboost_multioutput(
#     X_masked: np.ndarray,
#     model,
#     n_targets: int,
#     predict_device: str = "cuda",
# ) -> BaselineResult:
#     _check_3d(X_masked)

#     n_samples, n_steps, n_features = X_masked.shape
#     flat = X_masked.reshape(n_samples * n_steps, n_features).copy()

#     booster = model.get_booster()
#     booster.set_param({"device": predict_device})

#     X_pred = flat[:, n_targets:]
#     preds_all = model.predict(X_pred)

#     if preds_all.ndim == 1:
#         preds_all = preds_all.reshape(-1, 1)

#     for target_idx in range(n_targets):
#         miss_mask = np.isnan(flat[:, target_idx])
#         n_missing = int(miss_mask.sum())
#         if n_missing == 0:
#             print(f"[XGB-MO] alvo={target_idx} sem faltantes neste split.")
#             continue

#         flat[miss_mask, target_idx] = preds_all[miss_mask, target_idx]
#         print(f"[XGB-MO] alvo={target_idx} | imputados={n_missing:,}")

#     X_out = flat.reshape(n_samples, n_steps, n_features)
#     return BaselineResult(name="XGBoostMultiOutput", X_imputed=X_out)


# # def transform_with_xgboost_iterative(
#     X_masked: np.ndarray,
#     models: Dict[int, object],
#     n_targets: int,
#     predict_device: str = "cuda",
#     max_iters: int = 5,
#     tol: float = 1e-4
# ) -> BaselineResult:
#     _check_3d(X_masked)

#     n_samples, n_steps, n_features = X_masked.shape
#     flat = X_masked.reshape(n_samples * n_steps, n_features).copy()

#     start_time = time.time()

#     missing_mask = np.isnan(flat)
    
#     if not missing_mask.any():
#         print("[XGB-Iterativo] Nenhum NaN encontrado. Retornando array original.")
#         return BaselineResult(name="XGBoostIterative", X_imputed=X_masked)

#     print(f"[XGB-Iterativo] Iniciando imputação em {flat.shape[0]:,} linhas. Max iterações: {max_iters}")

#     col_means = np.nanmean(flat, axis=0)
#     for target_idx in range(n_targets):
#         miss_idx = missing_mask[:, target_idx]
#         if miss_idx.any():
#             fill_value = col_means[target_idx] if not np.isnan(col_means[target_idx]) else 0.0
#             flat[miss_idx, target_idx] = fill_value

#     for it in range(max_iters):
#         flat_old = flat.copy()
#         print(f"  --- Iteração {it + 1}/{max_iters} ---")
        
#         for target_idx in range(n_targets):
#             if target_idx not in models:
#                 continue

#             miss_idx = missing_mask[:, target_idx]
#             n_missing = int(miss_idx.sum())

#             if n_missing == 0:
#                 continue

#             model = models[target_idx]
#             booster = model.get_booster()
#             booster.set_param({"device": predict_device})

#             X_pred_cpu = np.delete(flat[miss_idx], target_idx, axis=1).astype(np.float32, copy=False)
#             X_pred_gpu = cp.asarray(X_pred_cpu)

#             preds_gpu = booster.inplace_predict(X_pred_gpu)
#             preds = cp.asnumpy(preds_gpu)
#             flat[miss_idx, target_idx] = preds

#         diff = np.linalg.norm(flat[missing_mask] - flat_old[missing_mask]) / (np.linalg.norm(flat_old[missing_mask]) + 1e-9)
#         print(f"  [XGB-Iterativo] Diferença para iteração anterior: {diff:.6f}")
        
#         if diff < tol:
#             print(f"  [XGB-Iterativo] Convergiu na iteração {it + 1}!")
#             break

#     X_out = flat.reshape(n_samples, n_steps, n_features)
#     print(f"[XGB-Iterativo] Imputação concluída em {time.time() - start_time:.2f}s.")
    
#     return BaselineResult(name="XGBoostIterative", X_imputed=X_out)