from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


# =========================
# IO
# =========================

def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: str | Path) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: dict, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_dataframe(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == '.parquet':
        return pd.read_parquet(path)
    if path.suffix == '.csv':
        return pd.read_csv(path)
    raise ValueError(f'Formato não suportado: {path.suffix}')


def save_dataframe(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    if path.suffix == '.parquet':
        df.to_parquet(path, index=False)
    elif path.suffix == '.csv':
        df.to_csv(path, index=False)
    else:
        raise ValueError(f'Formato não suportado: {path.suffix}')


# =========================
# CONFIG / CENÁRIOS
# =========================

def parse_bool(value: str | bool | None) -> bool | None:
    if value is None or isinstance(value, bool):
        return value
    txt = str(value).strip().lower()
    if txt in {'1', 'true', 't', 'yes', 'y', 'on'}:
        return True
    if txt in {'0', 'false', 'f', 'no', 'n', 'off'}:
        return False
    raise ValueError(f'Valor booleano inválido: {value}')


def build_scenarios_from_config(config: dict) -> list[dict]:
    scenarios = [
        {
            'name': 'base',
            'train_mask_fraction': 0.0,
            'val_mask_fraction': float(config.get('mask_fraction_eval', 0.2)),
            'test_mask_fraction': float(config.get('mask_fraction_eval', 0.2)),
        }
    ]
    for frac in config.get('MASKARAMENTO_ALEATORIO_DOS_CENARIOS', []):
        frac = float(frac)
        scenarios.append(
            {
                'name': f'mcar_{int(frac * 100)}pct',
                'train_mask_fraction': frac,
                'val_mask_fraction': frac,
                'test_mask_fraction': frac,
            }
        )
    return scenarios


def select_scenarios(config: dict, requested_name: str | None = None) -> list[dict]:
    scenarios = build_scenarios_from_config(config)
    if requested_name:
        selected = [s for s in scenarios if s['name'] == requested_name]
        if not selected:
            raise ValueError(f'Cenário não encontrado: {requested_name}')
        return selected

    run_base = bool(config.get('run_base_scenario', True))
    run_mcar = bool(config.get('run_mcar_scenarios', True))
    selected = []
    for scenario in scenarios:
        if scenario['name'] == 'base' and run_base:
            selected.append(scenario)
        elif scenario['name'] != 'base' and run_mcar:
            selected.append(scenario)
    if not selected:
        raise ValueError('Nenhum cenário selecionado.')
    return selected


# =========================
# SPLIT / SCALER
# =========================

@dataclass
class SplitResult:
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    info: dict


@dataclass
class NaNStandardScaler:
    feature_names: list[str] | None = None
    means_: np.ndarray | None = None
    stds_: np.ndarray | None = None

    def fit(self, df: pd.DataFrame, feature_names: Sequence[str]) -> 'NaNStandardScaler':
        self.feature_names = list(feature_names)
        values = df[self.feature_names].to_numpy(dtype=float)
        self.means_ = np.nanmean(values, axis=0)
        self.stds_ = np.nanstd(values, axis=0)
        self.stds_ = np.where((self.stds_ == 0) | np.isnan(self.stds_), 1.0, self.stds_)
        return self

    def transform_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        self._check()
        out = df.copy()
        out[self.feature_names] = (out[self.feature_names].to_numpy(dtype=float) - self.means_) / self.stds_
        return out

    def inverse_transform_targets(self, x_targets: np.ndarray, target_names: Sequence[str]) -> np.ndarray:
        self._check()
        idx = [self.feature_names.index(name) for name in target_names]
        means = self.means_[idx].reshape(1, 1, -1)
        stds = self.stds_[idx].reshape(1, 1, -1)
        return x_targets * stds + means

    def to_dict(self) -> dict:
        self._check()
        return {
            'feature_names': self.feature_names,
            'means': self.means_.tolist(),
            'stds': self.stds_.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'NaNStandardScaler':
        obj = cls()
        obj.feature_names = list(data['feature_names'])
        obj.means_ = np.array(data['means'], dtype=float)
        obj.stds_ = np.array(data['stds'], dtype=float)
        return obj

    def _check(self) -> None:
        if self.feature_names is None or self.means_ is None or self.stds_ is None:
            raise ValueError('Scaler ainda não ajustado.')


def temporal_split(
    df: pd.DataFrame,
    time_col: str,
    dev_frac: float = 0.8,
    val_frac_within_dev: float = 0.2,
    gap_steps: int = 0,
) -> SplitResult:
    data = df.copy()
    data[time_col] = pd.to_datetime(data[time_col])
    data = data.sort_values(time_col).reset_index(drop=True)

    unique_times = pd.Index(sorted(data[time_col].dropna().unique()))
    n_times = len(unique_times)
    if n_times < 10:
        raise ValueError('Poucos timestamps únicos para o split temporal.')

    dev_end = int(n_times * dev_frac)
    dev_times = unique_times[:dev_end]
    test_times = unique_times[dev_end:]

    train_end_inside_dev = int(len(dev_times) * (1 - val_frac_within_dev))
    train_times = dev_times[:train_end_inside_dev]
    val_times = dev_times[train_end_inside_dev:]

    if gap_steps > 0:
        train_times = train_times[:-gap_steps] if len(train_times) > gap_steps else train_times[:0]
        val_times = val_times[gap_steps:] if len(val_times) > gap_steps else val_times[:0]
        val_times = val_times[:-gap_steps] if len(val_times) > gap_steps else val_times[:0]
        test_times = test_times[gap_steps:] if len(test_times) > gap_steps else test_times[:0]

    train_df = data[data[time_col].isin(train_times)].copy()
    val_df = data[data[time_col].isin(val_times)].copy()
    test_df = data[data[time_col].isin(test_times)].copy()

    info = {
        'n_unique_times': n_times,
        'train_start': str(train_times.min()) if len(train_times) else None,
        'train_end': str(train_times.max()) if len(train_times) else None,
        'val_start': str(val_times.min()) if len(val_times) else None,
        'val_end': str(val_times.max()) if len(val_times) else None,
        'test_start': str(test_times.min()) if len(test_times) else None,
        'test_end': str(test_times.max()) if len(test_times) else None,
        'n_rows_train': int(len(train_df)),
        'n_rows_val': int(len(val_df)),
        'n_rows_test': int(len(test_df)),
    }
    return SplitResult(train_df=train_df, val_df=val_df, test_df=test_df, info=info)


# =========================
# JANELAS
# =========================

@dataclass
class WindowResult:
    X: np.ndarray
    metadata: pd.DataFrame
    feature_names: list[str]


@dataclass
class PreparedData:
    split_info: dict
    scaler: NaNStandardScaler
    train_windows: WindowResult
    val_windows: WindowResult
    test_windows: WindowResult


def create_windows(
    df: pd.DataFrame,
    station_col: str,
    time_col: str,
    feature_names: Sequence[str],
    window_size: int,
    step_size: int,
    max_windows: int | None = None,
) -> WindowResult:
    data = df.copy()
    data[time_col] = pd.to_datetime(data[time_col])
    data = data.sort_values([station_col, time_col]).reset_index(drop=True)

    windows: list[np.ndarray] = []
    rows: list[dict] = []
    feature_names = list(feature_names)

    for station_id, group in data.groupby(station_col, sort=False):
        group = group.sort_values(time_col).reset_index(drop=True)
        values = group[feature_names].to_numpy(dtype=float)
        times = group[time_col].to_numpy()

        if len(group) < window_size:
            continue

        for start in range(0, len(group) - window_size + 1, step_size):
            end = start + window_size
            windows.append(values[start:end])
            rows.append(
                {
                    'station_id': station_id,
                    'window_start_idx_local': start,
                    'window_end_idx_local_exclusive': end,
                    'window_start_time': pd.Timestamp(times[start]),
                    'window_end_time': pd.Timestamp(times[end - 1]),
                }
            )
            if max_windows is not None and len(windows) >= max_windows:
                return WindowResult(np.stack(windows, axis=0), pd.DataFrame(rows), feature_names)

    if not windows:
        raise ValueError('Nenhuma janela foi criada.')
    return WindowResult(np.stack(windows, axis=0), pd.DataFrame(rows), feature_names)


def save_windows(result: WindowResult, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    md = result.metadata.copy()
    np.savez_compressed(
        path,
        X=result.X,
        station_id=md['station_id'].astype(str).to_numpy(),
        window_start_idx_local=md['window_start_idx_local'].to_numpy(),
        window_end_idx_local_exclusive=md['window_end_idx_local_exclusive'].to_numpy(),
        window_start_time=md['window_start_time'].astype(str).to_numpy(),
        window_end_time=md['window_end_time'].astype(str).to_numpy(),
        feature_names=np.array(result.feature_names, dtype=str),
    )


def load_windows(path: str | Path) -> WindowResult:
    data = np.load(path, allow_pickle=True)
    metadata = pd.DataFrame(
        {
            'station_id': data['station_id'].astype(str),
            'window_start_idx_local': data['window_start_idx_local'],
            'window_end_idx_local_exclusive': data['window_end_idx_local_exclusive'],
            'window_start_time': pd.to_datetime(data['window_start_time']),
            'window_end_time': pd.to_datetime(data['window_end_time']),
        }
    )
    return WindowResult(X=data['X'], metadata=metadata, feature_names=[str(x) for x in data['feature_names']])


def build_cache_key(config: dict) -> str:
    return f"ws{config['window_size']}_ss{config['step_size']}_gap{config.get('gap_steps', 0)}"


def prepare_data(config: dict) -> PreparedData:
    cache_root = Path(config.get('cache_dir', Path(config['pasta_saida']) / 'cache'))
    cache_dir = ensure_dir(cache_root / build_cache_key(config))

    scaler_path = cache_dir / 'scaler.json'
    split_path = cache_dir / 'split_info.json'
    train_path = cache_dir / 'train_windows.npz'
    val_path = cache_dir / 'val_windows.npz'
    test_path = cache_dir / 'test_windows.npz'

    if all(p.exists() for p in [scaler_path, split_path, train_path, val_path, test_path]):
        print('[CACHE] Carregando janelas prontas...')
        return PreparedData(
            split_info=load_json(split_path),
            scaler=NaNStandardScaler.from_dict(load_json(scaler_path)),
            train_windows=load_windows(train_path),
            val_windows=load_windows(val_path),
            test_windows=load_windows(test_path),
        )

    print('[DATA] Carregando dataframe...')
    df = load_dataframe(config['arquivo_entrada'])
    split = temporal_split(
        df=df,
        time_col=config['coluna_tempo'],
        dev_frac=config['train_frac_dev'],
        val_frac_within_dev=config['val_frac_within_dev'],
        gap_steps=config.get('gap_steps', 0),
    )

    feature_normaliza = config["target_cols"] + config["feature_cols"]
    
    features_n_normalizar = config.get("feature_n_normalizar", [])
    all_features = feature_normaliza + features_n_normalizar
    
    # all_features = list(config['target_cols']) + list(config['feature_cols'])
    scaler = NaNStandardScaler().fit(split.train_df, feature_normaliza)

    print('[DATA] Escalando splits...')
    train_scaled = scaler.transform_dataframe(split.train_df)
    val_scaled = scaler.transform_dataframe(split.val_df)
    test_scaled = scaler.transform_dataframe(split.test_df)

    print('[DATA] Criando janelas...')
    train_windows = create_windows(
        train_scaled,
        station_col=config['coluna_estacao'],
        time_col=config['coluna_tempo'],
        feature_names=all_features,
        window_size=config['window_size'],
        step_size=config['step_size'],
        max_windows=config.get('max_windows_train'),
    )
    val_windows = create_windows(
        val_scaled,
        station_col=config['coluna_estacao'],
        time_col=config['coluna_tempo'],
        feature_names=all_features,
        window_size=config['window_size'],
        step_size=config['step_size'],
        max_windows=config.get('max_windows_val'),
    )
    test_windows = create_windows(
        test_scaled,
        station_col=config['coluna_estacao'],
        time_col=config['coluna_tempo'],
        feature_names=all_features,
        window_size=config['window_size'],
        step_size=config['step_size'],
        max_windows=config.get('max_windows_test'),
    )

    print('[CACHE] Salvando janelas prontas...')
    save_json(split.info, split_path)
    save_json(scaler.to_dict(), scaler_path)
    save_windows(train_windows, train_path)
    save_windows(val_windows, val_path)
    save_windows(test_windows, test_path)

    return PreparedData(split.info, scaler, train_windows, val_windows, test_windows)


# =========================
# MÁSCARAS E LIMITES
# =========================

@dataclass
class MaskResult:
    X_masked: np.ndarray
    artificial_mask: np.ndarray
    ground_truth: np.ndarray


def apply_random_mask_on_targets(X: np.ndarray, n_targets: int, mask_fraction: float, seed: int = 42) -> MaskResult:
    if X.ndim != 3:
        raise ValueError('X deve ter shape [N, T, F].')
    rng = np.random.default_rng(seed)
    X_masked = X.copy()
    target_block = X[:, :, :n_targets]
    observed = ~np.isnan(target_block)
    candidates = np.argwhere(observed)
    n_to_mask = int(round(len(candidates) * mask_fraction))
    artificial_mask = np.zeros_like(target_block, dtype=bool)
    if n_to_mask > 0 and len(candidates) > 0:
        chosen = rng.choice(len(candidates), size=n_to_mask, replace=False)
        selected = candidates[chosen]
        for i, t, f in selected:
            artificial_mask[i, t, f] = True
            X_masked[i, t, f] = np.nan
    return MaskResult(X_masked=X_masked, artificial_mask=artificial_mask, ground_truth=target_block.copy())


def maybe_apply_random_mask_on_targets(X: np.ndarray, n_targets: int, mask_fraction: float, seed: int = 42) -> MaskResult:
    if mask_fraction <= 0:
        return MaskResult(
            X_masked=X.copy(),
            artificial_mask=np.zeros_like(X[:, :, :n_targets], dtype=bool),
            ground_truth=X[:, :, :n_targets].copy(),
        )
    return apply_random_mask_on_targets(X, n_targets, mask_fraction, seed)


def apply_physical_limits_to_targets(
    x_targets: np.ndarray,
    target_names: Sequence[str],
    physical_limits: dict[str, list | tuple],
) -> np.ndarray:
    out = x_targets.copy()
    for i, name in enumerate(target_names):
        if name not in physical_limits:
            continue
        vmin, vmax = physical_limits[name]
        if vmin is not None:
            out[:, :, i] = np.maximum(out[:, :, i], vmin)
        if vmax is not None:
            out[:, :, i] = np.minimum(out[:, :, i], vmax)
    return out


# =========================
# SALVAR AMOSTRAS
# =========================

def save_sample_window_debug(
    path: str | Path,
    x_true_targets: np.ndarray,
    x_pred_targets: np.ndarray,
    artificial_mask: np.ndarray,
    metadata: pd.DataFrame,
    target_names: Sequence[str],
    n_samples: int = 50,
    seed: int = 42,
) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    n = x_true_targets.shape[0]
    if n == 0:
        return
    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    if n_samples < n:
        indices = rng.choice(indices, size=n_samples, replace=False)
    np.savez_compressed(
        path,
        indices=indices,
        X_true_targets=x_true_targets[indices],
        X_pred_targets=x_pred_targets[indices],
        artificial_mask=artificial_mask[indices],
        station_id=metadata['station_id'].iloc[indices].astype(str).to_numpy(),
        window_start_time=metadata['window_start_time'].iloc[indices].astype(str).to_numpy(),
        window_end_time=metadata['window_end_time'].iloc[indices].astype(str).to_numpy(),
        target_names=np.array(list(target_names), dtype=str),
    )

# =========================
# CARREGAMENTO DE CHECKPOINTS (PYPOTS / DEEP LEARNING)
# =========================

def _find_latest_checkpoint(checkpoint_dir: Path) -> Path | None:
    candidates = []
    for pattern in ('*.pypots', '*.pt', '*.pth', '*.ckpt'):
        candidates.extend(checkpoint_dir.rglob(pattern))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)

def try_load_checkpoint(model_obj, checkpoint_dir: Path, model_name: str) -> bool:
    latest = _find_latest_checkpoint(checkpoint_dir)
    targets = [str(latest)] if latest is not None else [str(checkpoint_dir)]
    for method_name in ('load', 'load_model'):
        method = getattr(model_obj, method_name, None)
        if not callable(method):
            continue
        for target in targets:
            try:
                method(target)
                print(f'{model_name}: checkpoint carregado via {method_name}: {target}')
                return True
            except Exception:
                pass
    return False