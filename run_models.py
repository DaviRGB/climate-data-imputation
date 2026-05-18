from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import (
    ensure_dir,
    load_json,
    save_dataframe,
    save_json,
    parse_bool,
    prepare_data,
    maybe_apply_random_mask_on_targets,
    apply_physical_limits_to_targets,
    build_scenarios_from_config,
    select_scenarios,
    save_sample_window_debug,
)
from src.metrics import build_result_tables
from src.baselines import (
    compute_feature_medians_from_train,
    median_imputation,
    locf_imputation,
    locf_then_median_imputation,
    fit_knn_imputer_from_train,
    transform_with_knn_imputer,
    fit_missforest_like_imputer_from_train,
    transform_with_missforest_like_imputer,
    fit_xgboost_per_target_from_train,
    transform_with_xgboost_per_target,
    # fit_xgboost_multioutput_from_train,
    # transform_with_xgboost_multioutput,
)
from src.models.SAITS import run_saits
from src.models.ImputerFormer import run_imputeformer
from src.tune_parametrs import tune_hyperparameters
from src.models.Transformer import run_transformer

from src.tune_parametrs import collect_tuned_params_from_journal

def apply_runtime_model_flags(config: dict, model_group: str, run_saits=None, run_imputeformer=None, run_baselines=None, run_transformer=None) -> dict:
    cfg = dict(config)
    if model_group == 'deep':
        cfg['run_saits'] = True
        cfg['run_imputeformer'] = True
        cfg['run_baselines'] = False
        cfg['run_transformer'] = True
    elif model_group == 'baselines':
        cfg['run_saits'] = False
        cfg['run_imputeformer'] = False
        cfg['run_baselines'] = True
        cfg['run_transformer'] = False
    elif model_group != 'all':
        raise ValueError(f'model_group inválido: {model_group}')

    explicit = {
        'run_saits': parse_bool(run_saits),
        'run_imputeformer': parse_bool(run_imputeformer),
        'run_baselines': parse_bool(run_baselines),
        'run_transformer': parse_bool(run_transformer),
    }
    for key, value in explicit.items():
        if value is not None:
            cfg[key] = value
    return cfg


def inverse_targets_and_clip(x_pred: np.ndarray, scaler, config: dict) -> np.ndarray:
    n_targets = len(config['target_cols'])
    x_targets = scaler.inverse_transform_targets(x_pred[:, :, :n_targets], config['target_cols'])
    return apply_physical_limits_to_targets(x_targets, config['target_cols'], config['physical_limits'])


def load_or_tune_params(config: dict, out_dir: Path) -> dict:
    best_params_path = out_dir / 'best_params_tuning.json'
    if config.get('run_tuning', False):
        tuned_params = tune_hyperparameters(
            base_config=config,
            tune_saits=bool(config.get('run_saits', True)),
            tune_imputeformer=bool(config.get('run_imputeformer', True)),
            tune_transformer=bool(config.get('run_transformer', True)),
            tune_xgboost=bool(config.get('run_xgb_per_target_baseline', True)),
            output_dir=out_dir,
        )
        if tuned_params:
            config.update(tuned_params)
            save_json(tuned_params, best_params_path)
        return config

    if best_params_path.exists():
        best_params = load_json(best_params_path)
        config.update(best_params)
        print(f'[PARAMS] Reaproveitando best_params_tuning.json com {len(best_params)} parâmetros.')
    else:
        print('[PARAMS] Sem tuning. Usando apenas o JSON base.')
    return config


def save_prediction_samples(scenario_dir: Path, model_name: str, split_name: str, scenario_name: str, seed: int, x_true_targets, x_pred_targets, artificial_mask, metadata, target_names, n_samples: int) -> None:
    path = scenario_dir / 'predicoes' / f'sample_predictions_{model_name}_{split_name}_{scenario_name}_seed{seed}.npz'
    save_sample_window_debug(path, x_true_targets, x_pred_targets, artificial_mask, metadata, target_names, n_samples=n_samples, seed=seed)


def evaluate_and_store(
    scenario_dir: Path,
    model_name: str,
    seed: int,
    scenario_name: str,
    config: dict,
    scaler,
    x_pred_val: np.ndarray,
    x_pred_test: np.ndarray,
    x_true_val_targets: np.ndarray,
    x_true_test_targets: np.ndarray,
    val_mask,
    test_mask,
    val_metadata,
    test_metadata,
) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
    global_frames: list[pd.DataFrame] = []
    per_var_frames: list[pd.DataFrame] = []

    x_pred_val_targets = inverse_targets_and_clip(x_pred_val, scaler, config)
    x_pred_test_targets = inverse_targets_and_clip(x_pred_test, scaler, config)

    gdf, pdf = build_result_tables(model_name, 'validation', seed, scenario_name, x_true_val_targets, x_pred_val_targets, val_mask.artificial_mask, config['target_cols'])
    global_frames.append(gdf)
    per_var_frames.append(pdf)
    gdf, pdf = build_result_tables(model_name, 'test', seed, scenario_name, x_true_test_targets, x_pred_test_targets, test_mask.artificial_mask, config['target_cols'])
    global_frames.append(gdf)
    per_var_frames.append(pdf)

    if bool(config.get('save_prediction_samples', True)):
        n_samples = int(config.get('save_prediction_samples_n', 100))
        save_prediction_samples(scenario_dir, model_name, 'validation', scenario_name, seed, x_true_val_targets, x_pred_val_targets, val_mask.artificial_mask, val_metadata, config['target_cols'], n_samples)
        save_prediction_samples(scenario_dir, model_name, 'test', scenario_name, seed, x_true_test_targets, x_pred_test_targets, test_mask.artificial_mask, test_metadata, config['target_cols'], n_samples)

    return global_frames, per_var_frames


def run_baselines_for_scenario(
    config: dict, 
    seed: int, scenario_name: str, scenario_dir: Path, 
    scaler, x_train, x_val_clean, x_val_masked, 
    x_test_masked, x_true_val_targets, x_true_test_targets, 
    val_mask, test_mask, val_metadata, test_metadata) -> tuple[list[pd.DataFrame], list[pd.DataFrame], list[dict]]:
    global_frames: list[pd.DataFrame] = []
    per_var_frames: list[pd.DataFrame] = []
    time_rows: list[dict] = []

    train_feature_medians = compute_feature_medians_from_train(x_train)
    baseline_specs = []
    
    if bool(config.get('run_median_baseline', False)):
        baseline_specs.append(('Median', None, lambda x, _: median_imputation(x, train_feature_medians)))
    if bool(config.get('run_locf_baseline', False)):
        baseline_specs.append(('LOCF', None, lambda x, _: locf_imputation(x)))
    if bool(config.get('run_locf_then_median_baseline', False)):
        baseline_specs.append(('LOCFThenMedian', None, lambda x, _: locf_then_median_imputation(x, train_feature_medians)))
        
    if bool(config.get('run_knn_baseline', False)):
        baseline_specs.append(
            ('KNN',
             lambda xt: fit_knn_imputer_from_train(xt, n_neighbors=int(config.get('knn_neighbors', 5)), weights=str(config.get('knn_weights', 'uniform')), max_rows_fit=config.get('knn_max_rows_fit', 100_000), seed=seed),
             lambda x, fitted: transform_with_knn_imputer(x, fitted, n_neighbors=int(config.get('knn_neighbors', 5)), batch_size=int(config.get('knn_transform_batch_size', 2000))))
        )
    if bool(config.get('run_missforest_baseline', False)):
        baseline_specs.append(
            ('MissForestLike',
             lambda xt: fit_missforest_like_imputer_from_train(xt, random_state=seed, n_estimators=int(config.get('mf_n_estimators', 10)), max_iter=int(config.get('mf_max_iter', 2)), n_jobs=int(config.get('mf_n_jobs', 1)), tol=float(config.get('mf_tol', 1e-2)), n_nearest_features=config.get('mf_n_nearest_features', 10), max_rows_fit=config.get('mf_max_rows_fit', 50_000)),
             lambda x, fitted: transform_with_missforest_like_imputer(x, fitted, batch_size=int(config.get('mf_transform_batch_size', 2000))))
        )

    if bool(config.get('run_xgb_per_target_baseline', False)):

        baseline_specs.append(
            ('XGBoostPerTarget',
             lambda xt: fit_xgboost_per_target_from_train(
                    X_train=xt, X_val_masked=x_val_masked, X_true_val_targets_scaled=x_val_clean[:, :, :len(config['target_cols'])], val_artificial_mask=val_mask.artificial_mask,
                    n_targets=len(config['target_cols']), random_state=seed, device=config.get('xgb_device', 'cuda'), n_estimators=int(config.get('xgb_n_estimators', 100)),
                    max_depth=int(config.get('xgb_max_depth', 4)), learning_rate=float(config.get('xgb_learning_rate', 0.05)), subsample=float(config.get('xgb_subsample', 0.8)),
                    colsample_bytree=float(config.get('xgb_colsample_bytree', 0.8)), reg_alpha=float(config.get('xgb_reg_alpha', 0.0)), reg_lambda=float(config.get('xgb_reg_lambda', 1.0)), n_jobs=int(config.get('xgb_n_jobs', 1)), early_stopping_rounds=int(config.get('xgb_early_stopping_rounds', 20)), eval_metric=str(config.get('xgb_eval_metric', 'rmse')),
                ),
            #  lambda xt: fit_xgboost_per_target_from_train(xt, n_targets=len(config['target_cols']), random_state = seed, device = config.get('xgb_device', 'cuda'), n_estimators=int(config.get('xgb_n_estimators', 100)), max_depth=int(config.get('xgb_max_depth', 4)), learning_rate=float(config.get('xgb_learning_rate', 0.05)), subsample=float(config.get('xgb_subsample', 0.8)), colsample_bytree=float(config.get('xgb_colsample_bytree', 0.8)), reg_alpha=float(config.get('xgb_reg_alpha', 0.0)), reg_lambda=float(config.get('xgb_reg_lambda', 1.0)), n_jobs=int(config.get('xgb_n_jobs', 1)), eval_metric=str(config.get('xgb_eval_metric', 'rmse'))),
             lambda x, fitted: transform_with_xgboost_per_target(x, fitted, n_targets=len(config['target_cols']), predict_device=config.get('xgb_device', 'cuda'),)) # , batch_size=int(config.get('xgb_transform_batch_size', 2000))
        )

    # if bool(config.get('run_xgb_multioutput_baseline', False)):
    #     baseline_specs.append(
    #         ('XGBoostMultiOutput',
    #          lambda xt: fit_xgboost_multioutput_from_train(X_train=xt, X_val_masked=x_val_masked, X_val_targets_scaled=x_val_clean[:, :, :len(config['target_cols'])], X_true_val_targets=x_true_val_targets, val_artificial_mask=val_mask.artificial_mask, n_targets=len(config['target_cols']), random_state=seed, device=config.get('xgb_device', 'cuda'), n_estimators=int(config.get('xgb_n_estimators', 100)), max_depth=int(config.get('xgb_max_depth', 4)), learning_rate=float(config.get('xgb_learning_rate', 0.05)), subsample=float(config.get('xgb_subsample', 0.8)), colsample_bytree=float(config.get('xgb_colsample_bytree', 0.8)), reg_alpha=float(config.get('xgb_reg_alpha', 0.0)), reg_lambda=float(config.get('xgb_reg_lambda', 1.0)), n_jobs=int(config.get('xgb_n_jobs', 1)), early_stopping_rounds=int(config.get('xgb_early_stopping_rounds', 20)), eval_metric=str(config.get('xgb_eval_metric', 'rmse')), multi_strategy=str(config.get('xgb_multi_strategy', 'multi_output_tree'))),
    #          lambda x, fitted: transform_with_xgboost_multioutput(x, fitted, n_targets=len(config['target_cols']), predict_device=config.get('xgb_device', 'cuda'),)) # , batch_size=int(config.get('xgb_transform_batch_size', 2000))
    #     )
    for model_name, fit_fn, transform_fn in baseline_specs:
        try:
            fitted = None
            if fit_fn is not None:
                t0 = time.time()
                fitted = fit_fn(x_train)
                time_rows.append({'scenario': scenario_name, 'model': model_name, 'stage': 'fit_train', 'seconds': time.time() - t0})

            t0 = time.time()
            pred_val = transform_fn(x_val_masked, fitted).X_imputed
            time_rows.append({'scenario': scenario_name, 'model': model_name, 'stage': 'validation_predict', 'seconds': time.time() - t0})
            t0 = time.time()
            pred_test = transform_fn(x_test_masked, fitted).X_imputed
            time_rows.append({'scenario': scenario_name, 'model': model_name, 'stage': 'test_predict', 'seconds': time.time() - t0})

            gframes, pframes = evaluate_and_store(scenario_dir, model_name, seed, scenario_name, config, scaler, pred_val, pred_test, x_true_val_targets, x_true_test_targets, val_mask, test_mask, val_metadata, test_metadata)
            global_frames.extend(gframes)
            per_var_frames.extend(pframes)
        except Exception as exc:
            save_json({'scenario': scenario_name, 'model': model_name, 'error': repr(exc)}, scenario_dir / f'erro_{model_name}.json')
            print(f'[ERRO] {model_name} falhou: {exc}')
    return global_frames, per_var_frames, time_rows


def run_deep_models_for_scenario(config: dict, seed: int, scenario_name: str, scenario_dir: Path, scaler, x_train, x_val_masked, x_val_original, x_test_masked, x_true_val_targets, x_true_test_targets, val_mask, test_mask, val_metadata, test_metadata) -> tuple[list[pd.DataFrame], list[pd.DataFrame], list[dict]]:
    global_frames: list[pd.DataFrame] = []
    per_var_frames: list[pd.DataFrame] = []
    time_rows: list[dict] = []
    config = dict(config)
    config['n_features'] = len(config['target_cols']) + len(config['feature_cols']) + len(config.get('feature_n_normalizar', [])) #como adicionar aqui + config["feature_n_normalizar"] sem quebrar tudo? --- IGNORE --- e sem ela ir para a função de normalização?

    if bool(config.get('run_saits', True)):
        t0 = time.time()
        pred_val, pred_test = run_saits(x_train, x_val_masked, x_val_original, x_test_masked, config, checkpoint_dir=scenario_dir / 'checkpoints' / 'saits')
        time_rows.append({'scenario': scenario_name, 'model': 'SAITS', 'stage': 'full_run', 'seconds': time.time() - t0})
        gframes, pframes = evaluate_and_store(scenario_dir, 'SAITS', seed, scenario_name, config, scaler, pred_val, pred_test, x_true_val_targets, x_true_test_targets, val_mask, test_mask, val_metadata, test_metadata)
        global_frames.extend(gframes)
        per_var_frames.extend(pframes)

    if bool(config.get('run_imputeformer', True)):
        t0 = time.time()
        pred_val, pred_test = run_imputeformer(x_train, x_val_masked, x_val_original, x_test_masked, config, checkpoint_dir=scenario_dir / 'checkpoints' / 'imputeformer')
        time_rows.append({'scenario': scenario_name, 'model': 'ImputeFormer', 'stage': 'full_run', 'seconds': time.time() - t0})
        gframes, pframes = evaluate_and_store(scenario_dir, 'ImputeFormer', seed, scenario_name, config, scaler, pred_val, pred_test, x_true_val_targets, x_true_test_targets, val_mask, test_mask, val_metadata, test_metadata)
        global_frames.extend(gframes)
        per_var_frames.extend(pframes)

    if bool(config.get('run_transformer', True)):
        t0 = time.time()
        pred_val, pred_test = run_transformer(x_train, x_val_masked, x_val_original, x_test_masked, config, checkpoint_dir=scenario_dir / 'checkpoints' / 'transformer')
        time_rows.append({'scenario': scenario_name, 'model': 'Transformer', 'stage': 'full_run', 'seconds': time.time() - t0})
        gframes, pframes = evaluate_and_store(scenario_dir, 'Transformer', seed, scenario_name, config, scaler, pred_val, pred_test, x_true_val_targets, x_true_test_targets, val_mask, test_mask, val_metadata, test_metadata)
        global_frames.extend(gframes)
        per_var_frames.extend(pframes)
        
    return global_frames, per_var_frames, time_rows

def _upsert_results(new_df: pd.DataFrame, path: Path, keys: list[str]) -> None:
    if not path.exists():
        save_dataframe(new_df, path)
        return
    if path.suffix == '.csv':
        old_df = pd.read_csv(path)
    else:
        old_df = pd.read_parquet(path)
        
    combined = pd.concat([old_df, new_df], ignore_index=True)
    valid_keys = [k for k in keys if k in combined.columns]
    
    if valid_keys:
        combined = combined.drop_duplicates(subset=valid_keys, keep='last')

    save_dataframe(combined, path)

def save_scenario_outputs(scenario_dir: Path, global_frames: list[pd.DataFrame], per_var_frames: list[pd.DataFrame], time_rows: list[dict]) -> None:
    if global_frames:
        new_global = pd.concat(global_frames, ignore_index=True)
        _upsert_results(new_global, scenario_dir / 'metricas_globais.csv', keys=['model', 'split', 'seed', 'scenario'])
        
    if per_var_frames:
        new_per_var = pd.concat(per_var_frames, ignore_index=True)
        _upsert_results(new_per_var, scenario_dir / 'metricas_por_variavel.csv', keys=['model', 'split', 'seed', 'scenario', 'variable'])
        
    if time_rows:
        new_times = pd.DataFrame(time_rows)
        _upsert_results(new_times, scenario_dir / 'tempos.csv', keys=['model', 'scenario', 'stage'])

def run_one_scenario(config: dict, prepared, scenario: dict, out_dir: Path) -> None:
    seed = int(config.get('seed', 42))
    n_targets = len(config['target_cols'])
    scenario_name = scenario['name']
    scenario_dir = ensure_dir(out_dir / scenario_name)

    x_train = prepared.train_windows.X
    x_val = prepared.val_windows.X
    x_test = prepared.test_windows.X

    train_mask = maybe_apply_random_mask_on_targets(x_train, n_targets, float(scenario['train_mask_fraction']), seed)
    val_mask = maybe_apply_random_mask_on_targets(x_val, n_targets, float(scenario['val_mask_fraction']), seed)
    test_mask = maybe_apply_random_mask_on_targets(x_test, n_targets, float(scenario['test_mask_fraction']), seed)

    print("\n" + "=" * 60)
    print(f"RESUMO DOS DADOS DO CENÁRIO Executando cenário: {scenario['name']}")
    print("=" * 60)
    print(f"x_train       : {x_train.shape}")
    print(f"x_val_clean   : {x_val.shape}")
    print(f"x_test_clean  : {x_test.shape}")
    print(f"x_val_masked  : {val_mask.X_masked.shape} | missing rate: {val_mask.artificial_mask.mean() * 100:.2f}%")
    print(f"x_test_masked : {test_mask.X_masked.shape} | missing rate: {test_mask.artificial_mask.mean() * 100:.2f}%")
    print(f"Máscara treino: {int(train_mask.artificial_mask.sum()):,} pontos")
    print(f"Máscara val   : {int(val_mask.artificial_mask.sum()):,} pontos")
    print(f"Máscara teste : {int(test_mask.artificial_mask.sum()):,} pontos")
    print("=" * 60 + "\n")
    
    if bool(config.get('save_mask_arrays', True)):
        mask_dir = ensure_dir(scenario_dir / 'masks')
        np.save(mask_dir / 'train_mask.npy', train_mask.artificial_mask)
        np.save(mask_dir / 'val_mask.npy', val_mask.artificial_mask)
        np.save(mask_dir / 'test_mask.npy', test_mask.artificial_mask)

    x_true_val_targets = inverse_targets_and_clip(x_val, prepared.scaler, config)
    x_true_test_targets = inverse_targets_and_clip(x_test, prepared.scaler, config)

    global_frames: list[pd.DataFrame] = []
    per_var_frames: list[pd.DataFrame] = []
    time_rows: list[dict] = []

    if bool(config.get('run_baselines', False)):
        gframes, pframes, times = run_baselines_for_scenario(
            config, seed, scenario_name, scenario_dir, prepared.scaler,
            x_train, x_val, val_mask.X_masked, test_mask.X_masked,
            x_true_val_targets, x_true_test_targets,
            val_mask, test_mask,
            prepared.val_windows.metadata, prepared.test_windows.metadata,
        )
        global_frames.extend(gframes)
        per_var_frames.extend(pframes)
        time_rows.extend(times)

    if bool(config.get('run_saits', True)) or bool(config.get('run_imputeformer', True) or bool(config.get('run_transformer', True))):
        gframes, pframes, times = run_deep_models_for_scenario(
            config, seed, scenario_name, scenario_dir, prepared.scaler,
            train_mask.X_masked, val_mask.X_masked, x_val, test_mask.X_masked,
            x_true_val_targets, x_true_test_targets,
            val_mask, test_mask,
            prepared.val_windows.metadata, prepared.test_windows.metadata,
        )
        global_frames.extend(gframes)
        per_var_frames.extend(pframes)
        time_rows.extend(times)

    save_scenario_outputs(scenario_dir, global_frames, per_var_frames, time_rows)
    save_json({'scenario': scenario_name, 'seed': seed, 'config_snapshot': config}, scenario_dir / 'run_info.json')


def consolidate_results(out_dir: Path) -> None:
    global_frames = []
    per_var_frames = []
    time_frames = []
    for scenario_dir in sorted([p for p in out_dir.iterdir() if p.is_dir()]):
        g = scenario_dir / 'metricas_globais.csv'
        p = scenario_dir / 'metricas_por_variavel.csv'
        t = scenario_dir / 'tempos.csv'
        if g.exists():
            global_frames.append(pd.read_csv(g))
        if p.exists():
            per_var_frames.append(pd.read_csv(p))
        if t.exists():
            time_frames.append(pd.read_csv(t))
    if global_frames:
        save_dataframe(pd.concat(global_frames, ignore_index=True), out_dir / 'metricas_globais_todas.csv')
    if per_var_frames:
        save_dataframe(pd.concat(per_var_frames, ignore_index=True), out_dir / 'metricas_por_variavel_todas.csv')
    if time_frames:
        save_dataframe(pd.concat(time_frames, ignore_index=True), out_dir / 'tempos_todos.csv')

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', choices=['tune', 'tune_worker', 'collect_tuning', 'scenario', 'run_all', 'consolidate'], default='run_all')
    parser.add_argument('--scenario', type=str, default=None)
    parser.add_argument('--config', type=str, default='experimento_base.json')
    parser.add_argument('--out_dir', type=str, default=None)
    parser.add_argument('--model_group', type=str, choices=['all', 'deep', 'baselines'], default='all')
    parser.add_argument('--run_saits', type=str, default=None)
    parser.add_argument('--run_imputeformer', type=str, default=None)
    parser.add_argument('--run_transformer', type=str, default=None)
    parser.add_argument('--run_baselines', type=str, default=None)
    args = parser.parse_args()

    config = load_json(args.config)
    out_dir = Path(args.out_dir) if args.out_dir else ensure_dir(config['pasta_saida'])
    ensure_dir(out_dir)
    config = apply_runtime_model_flags(config, args.model_group, args.run_saits, args.run_imputeformer, args.run_baselines, args.run_transformer)

    if args.action == 'consolidate':
        consolidate_results(out_dir)
        return

    if args.action == 'tune_worker':
        config['run_tuning'] = True
        tune_hyperparameters(
            base_config=config,
            tune_saits=bool(config.get('run_saits', True)),
            tune_imputeformer=bool(config.get('run_imputeformer', True)),
            tune_transformer=bool(config.get('run_transformer', True)),
            tune_xgboost=bool(config.get('run_xgb_per_target_baseline', True)),
            output_dir=out_dir,
        )
        return

    if args.action == 'collect_tuning':
        tuned_params = collect_tuned_params_from_journal(
            base_config=config,
            tune_saits=bool(config.get('run_saits', True)),
            tune_imputeformer=bool(config.get('run_imputeformer', True)),
            tune_transformer=bool(config.get('run_transformer', True)),
            tune_xgboost=bool(config.get('run_xgb_per_target_baseline', True)),
            output_dir=out_dir,
        )
        if tuned_params:
            config.update(tuned_params)
            save_json(tuned_params, out_dir / 'best_params_tuning.json')
        save_json(config, out_dir / 'config_usada.json')
        return

    if args.action == 'tune':
        config['run_tuning'] = True
        config = load_or_tune_params(config, out_dir)
        save_json(config, out_dir / 'config_usada.json')
        return

    config['run_tuning'] = False if args.action in {'scenario', 'run_all'} else bool(config.get('run_tuning', False))
    config = load_or_tune_params(config, out_dir)

    config_usada_path = out_dir / 'config_usada.json'
    save_json(config, config_usada_path)

    prepared = prepare_data(config)

    scenarios = select_scenarios(config, args.scenario if args.action == 'scenario' else None)
    for scenario in scenarios:
        run_one_scenario(config, prepared, scenario, out_dir)

    consolidate_results(out_dir)
    save_json({'finished_at': datetime.now().isoformat(), 'config': config}, out_dir / 'run_summary.json')

if __name__ == '__main__':
    main()
