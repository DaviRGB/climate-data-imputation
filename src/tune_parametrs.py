from __future__ import annotations

import copy
from datetime import datetime
from pathlib import Path

import optuna

from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

from src.utils import load_json, save_dataframe, save_json, prepare_data, maybe_apply_random_mask_on_targets, apply_physical_limits_to_targets
from src.metrics import evaluate_global

from src.baselines import fit_xgboost_per_target_from_train, transform_with_xgboost_per_target
from src.metrics import build_result_tables

from src.models.SAITS import run_saits
from src.models.ImputerFormer import run_imputeformer
from src.models.Transformer import run_transformer

import gc
import optuna


def collect_tuned_params_from_journal(
    base_config: dict,
    tune_saits: bool = True,
    tune_imputeformer: bool = True,
    tune_transformer: bool = True,
    tune_xgboost: bool = True,
    output_dir: str | Path | None = None,
) -> dict:
    best_params: dict = {}
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = Path(output_dir) if output_dir is not None else None

    def _collect_one(model_name: str):
        storage, study_name, _ = _build_optuna_storage(base_config, model_name, output_dir)
        study = optuna.load_study(study_name=study_name, storage=storage)
        best_params.update(study.best_trial.params)
        if out_dir is not None:
            _save_study_logs(study, model_name, out_dir, timestamp)

    if tune_saits:
        _collect_one('saits')
    if tune_imputeformer:
        _collect_one('imputeformer')
    if tune_transformer:
        _collect_one('transformer')
    if tune_xgboost:
        _collect_one('xgboost')

    return best_params

def run_model_oom_protection(model_name: str, run_fn, *args, **kwargs):
    try:
        return run_fn(*args, **kwargs)
    except RuntimeError as e:
        msg = str(e).lower()

        if 'out of memory' in msg or 'cuda error: out of memory' in msg:
            print(f'[OOM] Trial podado em {model_name}: {e}')

            try:
                import torch
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

            raise optuna.TrialPruned(f'{model_name} pruned por OOM de GPU')

        raise

def _build_optuna_storage(base_config: dict, model_name: str, output_dir: str | Path | None):
    if output_dir is None:
        out_dir = Path(base_config.get('pasta_saida', '.'))
    else:
        out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    journal_path = out_dir / 'optuna_journal.log'
    study_name = f'{model_name}_tuning'

    storage = JournalStorage(JournalFileBackend(str(journal_path)))
    return storage, study_name, journal_path

def inverse_targets_and_clip(x_pred, scaler, config: dict):
    n_targets = len(config['target_cols'])
    x_targets = scaler.inverse_transform_targets(x_pred[:, :, :n_targets], config['target_cols'])
    return apply_physical_limits_to_targets(x_targets, config['target_cols'], config['physical_limits'])


def _save_study_logs(study, model_name: str, out_dir: Path, timestamp: str) -> None:
    trials_df = study.trials_dataframe()
    save_dataframe(trials_df, out_dir / f'tuning_trials_{model_name}.csv')
    save_dataframe(trials_df, out_dir / f'tuning_trials_{model_name}_{timestamp}.csv')
    if study.best_trial is not None:
        summary = {
            'model': model_name,
            'study_name': study.study_name,
            'best_trial': int(study.best_trial.number),
            'best_value_rmse': float(study.best_trial.value),
            'best_params': study.best_trial.params,
        }
        save_json(summary, out_dir / f'tuning_summary_{model_name}.json')


def prepare_dev_data(config: dict):
    prepared = prepare_data(config)
    return prepared.scaler, prepared.train_windows.X, prepared.val_windows.X


def objective_saits(trial, base_config: dict, scaler, x_train, x_val):
    config = copy.deepcopy(base_config)
    config['n_features'] = len(config['target_cols']) + len(config['feature_cols']) + len(config.get('feature_n_normalizar', []))
    config['saits_optimizer'] = trial.suggest_categorical('saits_optimizer', ['adam', 'adamw'])
    config['saits_lr'] = trial.suggest_float('saits_lr', 1e-5, 3e-3, log=True)
    config['saits_weight_decay'] = trial.suggest_float('saits_weight_decay', 1e-8, 1e-2, log=True)
    config['saits_n_layers'] = trial.suggest_int('saits_n_layers', 1, 3)
    config['saits_d_model'] = trial.suggest_categorical('saits_d_model', [64, 128, 256])
    config['saits_n_heads'] = trial.suggest_categorical('saits_n_heads', [2, 4, 8])
    config['saits_d_ffn'] = trial.suggest_categorical('saits_d_ffn', [128, 256, 512])
    config['saits_dropout'] = trial.suggest_float('saits_dropout', 0.05, 0.2)
    config['saits_attn_dropout'] = trial.suggest_float('saits_attn_dropout', 0.05, 0.2)
    config['saits_batch_size'] = trial.suggest_categorical('saits_batch_size', [32, 64, 128, 256])
    config['saits_epochs'] = int(base_config.get('tune_epochs', 30))
    config['saits_patience'] = int(base_config.get('tune_patience', 5))

    mask = maybe_apply_random_mask_on_targets(x_val, len(config['target_cols']), float(config.get('mask_fraction_eval', 0.2)), int(config.get('seed', 42)))
    val_pred, _ = run_model_oom_protection('SAITS', run_saits, x_train, mask.X_masked, x_val, mask.X_masked, config)
    x_true_targets = inverse_targets_and_clip(x_val, scaler, config)
    x_pred_targets = inverse_targets_and_clip(val_pred, scaler, config)
    return evaluate_global(x_true_targets, x_pred_targets, mask.artificial_mask)['RMSE']


def objective_imputeformer(trial, base_config: dict, scaler, x_train, x_val):
    config = copy.deepcopy(base_config)
    config['n_features'] = len(config['target_cols']) + len(config['feature_cols']) + len(config.get('feature_n_normalizar', []))
    config['imp_optimizer'] = trial.suggest_categorical('imp_optimizer', ['adam', 'adamw'])
    config['imp_lr'] = trial.suggest_float('imp_lr', 1e-5, 3e-3, log=True)
    config['imp_weight_decay'] = trial.suggest_float('imp_weight_decay', 1e-8, 1e-2, log=True)
    config['imp_n_layers'] = trial.suggest_int('imp_n_layers', 1, 3)
    config['imp_d_model'] = trial.suggest_categorical('imp_d_model', [64, 128, 256])
    config['imp_n_heads'] = trial.suggest_categorical('imp_n_heads', [2, 4, 8])
    config['imp_d_ffn'] = trial.suggest_categorical('imp_d_ffn', [128, 256, 512])
    config['imp_dropout'] = trial.suggest_float('imp_dropout', 0.05, 0.2)
    config['imp_batch_size'] = trial.suggest_categorical('imp_batch_size', [16, 32, 64, 128])
    config['imp_epochs'] = trial.suggest_int('imp_epochs', 20, int(base_config.get('tune_epochs', 30)))
    config['imp_patience'] = int(base_config.get('tune_patience', 5))

    mask = maybe_apply_random_mask_on_targets(x_val, len(config['target_cols']), float(config.get('mask_fraction_eval', 0.2)), int(config.get('seed', 42)))
    val_pred, _ = run_model_oom_protection('ImputeFormer', run_imputeformer, x_train, mask.X_masked, x_val, mask.X_masked, config)
    x_true_targets = inverse_targets_and_clip(x_val, scaler, config)
    x_pred_targets = inverse_targets_and_clip(val_pred, scaler, config)
    return evaluate_global(x_true_targets, x_pred_targets, mask.artificial_mask)['RMSE']


def objective_transformer(trial, base_config: dict, scaler, x_train, x_val):
    config = copy.deepcopy(base_config)
    config['n_features'] = len(config['target_cols']) + len(config['feature_cols']) + len(config.get('feature_n_normalizar', []))
    config['transf_optimizer'] = trial.suggest_categorical('transf_optimizer', ['adam', 'adamw'])
    config['transf_lr'] = trial.suggest_float('transf_lr', 1e-5, 3e-3, log=True)
    config['transf_weight_decay'] = trial.suggest_float('transf_weight_decay', 1e-8, 1e-2, log=True)
    config['transf_n_layers'] = trial.suggest_int('transf_n_layers', 1, 3)
    config['transf_d_model'] = trial.suggest_categorical('transf_d_model', [64, 128, 256])
    config['transf_n_heads'] = trial.suggest_categorical('transf_n_heads', [2, 4, 8])
    config['transf_d_ffn'] = trial.suggest_categorical('transf_d_ffn', [128, 256, 512])
    config['transf_dropout'] = trial.suggest_float('transf_dropout', 0.05, 0.2)
    config['transf_attn_dropout'] = trial.suggest_float('transf_attn_dropout', 0.05, 0.2)
    config['transf_batch_size'] = trial.suggest_categorical('transf_batch_size', [16, 32, 64, 128])
    config['transf_epochs'] = trial.suggest_int('transf_epochs', 20, int(base_config.get('tune_epochs', 30)))
    config['transf_patience'] = int(base_config.get('tune_patience', 5))

    mask = maybe_apply_random_mask_on_targets(x_val, len(config['target_cols']), float(config.get('mask_fraction_eval', 0.2)), int(config.get('seed', 42)))
    val_pred, _ = run_model_oom_protection('Transformer', run_transformer, x_train, mask.X_masked, x_val, mask.X_masked, config)
    x_true_targets = inverse_targets_and_clip(x_val, scaler, config)
    x_pred_targets = inverse_targets_and_clip(val_pred, scaler, config)
    return evaluate_global(x_true_targets, x_pred_targets, mask.artificial_mask)['RMSE']


def objective_xgboost(trial, base_config, scaler, x_train, x_val):
    config = copy.deepcopy(base_config)

    config['xgb_n_estimators'] = trial.suggest_int('xgb_n_estimators', 80, 250, step=20)
    config['xgb_max_depth'] = trial.suggest_int('xgb_max_depth', 3, 8)
    config['xgb_learning_rate'] = trial.suggest_float('xgb_learning_rate', 0.01, 0.15, log=True)
    config['xgb_subsample'] = trial.suggest_float('xgb_subsample', 0.6, 1.0)
    config['xgb_colsample_bytree'] = trial.suggest_float('xgb_colsample_bytree', 0.6, 1.0)
    config['xgb_reg_alpha'] = trial.suggest_float('xgb_reg_alpha', 0.0, 1.0)
    config['xgb_reg_lambda'] = trial.suggest_float('xgb_reg_lambda', 0.1, 5.0, log=True)
    config['xgb_early_stopping_rounds'] = trial.suggest_int('xgb_early_stopping_rounds', 10, 30, step=5)

    n_targets = len(config['target_cols'])

    mask = maybe_apply_random_mask_on_targets(
        x_val,
        n_targets,
        float(config.get('mask_fraction_eval', 0.2)),
        int(config.get('seed', 42))
    )

    models = fit_xgboost_per_target_from_train(
        X_train=x_train,
        X_val_masked=mask.X_masked,
        X_true_val_targets_scaled=x_val[:, :, :n_targets],
        val_artificial_mask=mask.artificial_mask,
        n_targets=n_targets,
        random_state=int(config.get('seed', 42)),
        device=config.get('xgb_device', 'cuda'),
        n_estimators=config['xgb_n_estimators'],
        max_depth=config['xgb_max_depth'],
        learning_rate=config['xgb_learning_rate'],
        subsample=config['xgb_subsample'],
        colsample_bytree=config['xgb_colsample_bytree'],
        reg_alpha=config['xgb_reg_alpha'],
        reg_lambda=config['xgb_reg_lambda'],
        n_jobs=int(config.get('xgb_n_jobs', 1)),
        early_stopping_rounds=config['xgb_early_stopping_rounds'],
        eval_metric=str(config.get('xgb_eval_metric', 'rmse')),
    )

    pred_val = transform_with_xgboost_per_target(
        mask.X_masked,
        models,
        n_targets=n_targets,
        predict_device=config.get('xgb_device', 'cuda'),
    ).X_imputed

    x_true_targets = inverse_targets_and_clip(x_val, scaler, config)
    x_pred_targets = inverse_targets_and_clip(pred_val, scaler, config)

    return evaluate_global(x_true_targets, x_pred_targets, mask.artificial_mask)['RMSE']


def tune_hyperparameters(base_config: dict, tune_saits: bool = True, tune_imputeformer: bool = True, tune_transformer: bool = True, tune_xgboost: bool = True, output_dir: str | Path | None = None) -> dict:
    scaler, x_train, x_val = prepare_dev_data(base_config)
    best_params: dict = {}
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = Path(output_dir) if output_dir is not None else None

    if tune_saits:
        storage, study_name, journal_path = _build_optuna_storage(base_config, 'saits', output_dir)
        study = optuna.create_study(
            direction='minimize',
            study_name=study_name,
            storage=storage,
            load_if_exists=True,
        )
        study.optimize(lambda trial: objective_saits(trial, base_config, scaler, x_train, x_val), n_trials=int(base_config.get('saits_tuning_trials', 10)))
        best_params.update(study.best_trial.params)
        if out_dir is not None:
            _save_study_logs(study, 'saits', out_dir, timestamp)

    if tune_imputeformer:
        storage, study_name, journal_path = _build_optuna_storage(base_config, 'imputeformer', output_dir)
        study = optuna.create_study(
            direction='minimize',
            study_name=study_name,
            storage=storage,
            load_if_exists=True,
        )
        study.optimize(lambda trial: objective_imputeformer(trial, base_config, scaler, x_train, x_val), n_trials=int(base_config.get('imp_tuning_trials', 5)))
        best_params.update(study.best_trial.params)
        if out_dir is not None:
            _save_study_logs(study, 'imputeformer', out_dir, timestamp)

    if tune_transformer:
        storage, study_name, journal_path = _build_optuna_storage(base_config, 'transformer', output_dir)
        study = optuna.create_study(
            direction='minimize',
            study_name=study_name,
            storage=storage,
            load_if_exists=True,
        )
        print(f'[OPTUNA] Transformer usando journal em: {journal_path}')
        study.optimize(lambda trial: objective_transformer(trial, base_config, scaler, x_train, x_val), n_trials=int(base_config.get('transf_tuning_trials', 5)))
        best_params.update(study.best_trial.params)
        if out_dir is not None:
            _save_study_logs(study, 'transformer', out_dir, timestamp)
    
    if tune_xgboost:
        storage, study_name, journal_path = _build_optuna_storage(base_config, 'xgboost', output_dir)
        study = optuna.create_study(
            direction='minimize',
            study_name=study_name,
            storage=storage,
            load_if_exists=True,
        )
        study.optimize(
            lambda trial: objective_xgboost(trial, base_config, scaler, x_train, x_val),
            n_trials=int(base_config.get('xgb_tuning_trials', 30))
        )
        best_params.update(study.best_trial.params)
        if out_dir is not None:
            _save_study_logs(study, 'xgboost', out_dir, timestamp)

    return best_params

if __name__ == '__main__':
    base_config = load_json('experimento_base.json')
    tune_hyperparameters(base_config, base_config.get('run_saits', True), base_config.get('run_imputeformer', True), base_config.get('run_transformer', True), base_config.get('run_xgboost', True), Path(base_config['pasta_saida']))
