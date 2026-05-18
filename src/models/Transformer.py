from __future__ import annotations
import gc
from pathlib import Path
import numpy as np

from src.utils import try_load_checkpoint

def run_transformer(
    X_train: np.ndarray,
    X_val_masked: np.ndarray,
    X_val_original: np.ndarray,
    X_test_masked: np.ndarray,
    config: dict,
    checkpoint_dir: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    import torch
    from pypots.imputation import Transformer
    from pypots.optim import Adam, AdamW

    def cleanup() -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def predict_in_chunks(model_obj, x_input: np.ndarray, split_name: str) -> np.ndarray:
        chunk_size = int(config.get('transf_predict_chunk_size', 2000))
        if x_input.shape[0] <= chunk_size:
            return model_obj.predict({'X': x_input})['imputation']
        print(f'Transformer: predição em chunks para {split_name} ({x_input.shape[0]} amostras).')
        preds = []
        for start in range(0, x_input.shape[0], chunk_size):
            end = min(start + chunk_size, x_input.shape[0])
            preds.append(model_obj.predict({'X': x_input[start:end]})['imputation'])
            cleanup()
        return np.concatenate(preds, axis=0)

    d_model = int(config.get('transf_d_model', 128))
    n_heads = int(config.get('transf_n_heads', 4))
    if d_model % n_heads != 0:
        raise ValueError(f'Transformer: d_model={d_model} precisa ser divisível por n_heads={n_heads}.')

    optimizer_name = str(config.get('transf_optimizer', 'adamw')).lower()
    lr = float(config.get('transf_lr', 1e-3))
    weight_decay = float(config.get('transf_weight_decay', 0.0))
    optimizer = AdamW(lr=lr, weight_decay=weight_decay) if optimizer_name == 'adamw' else Adam(lr=lr, weight_decay=weight_decay)

    model = Transformer(
        n_steps=config['window_size'],
        n_features=config['n_features'],
        n_layers=int(config.get('transf_n_layers', 2)),
        d_model=d_model,
        n_heads=n_heads,
        d_k=d_model // n_heads,
        d_v=d_model // n_heads,
        d_ffn=int(config.get('transf_d_ffn', 256)),
        dropout=float(config.get('transf_dropout', 0.1)),
        attn_dropout=float(config.get('transf_attn_dropout', 0.1)),
        batch_size=int(config.get('transf_batch_size', 64)),
        epochs=int(config.get('transf_epochs', 100)),
        patience=int(config.get('transf_patience', 10)),
        optimizer=optimizer,
        num_workers=int(config.get('num_workers', 0)),
        device=config.get('device', 'cuda'),
        saving_path=str(checkpoint_dir) if checkpoint_dir else None,
        model_saving_strategy=config.get('transf_model_saving_strategy', 'best') if checkpoint_dir else None,
    )

    checkpoint_loaded = False
    if checkpoint_dir and bool(config.get('resume_training_from_checkpoint', True)):
        checkpoint_loaded = try_load_checkpoint(model, Path(checkpoint_dir), model_name='Transformer')

    skip_fit = checkpoint_loaded and bool(config.get('skip_fit_if_checkpoint_loaded', False))
    if not skip_fit:
        model.fit({'X': X_train}, {'X': X_val_masked, 'X_ori': X_val_original})

    val_pred = predict_in_chunks(model, X_val_masked, 'validation')
    test_pred = predict_in_chunks(model, X_test_masked, 'test')
    cleanup()
    return val_pred, test_pred