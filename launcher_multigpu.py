from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def detect_available_gpus() -> list[str]:
    try:
        import torch
        if torch.cuda.is_available():
            return [str(i) for i in range(torch.cuda.device_count())]
    except Exception:
        pass
    try:
        proc = subprocess.run(['nvidia-smi', '--query-gpu=index', '--format=csv,noheader'], check=True, text=True, capture_output=True)
        return [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    except Exception:
        return []


def spawn_on_gpu(cmd_args: list[str], gpu_id: str, working_dir: Path):
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    return subprocess.Popen(cmd_args, env=env, cwd=str(working_dir))


def run_commands_on_gpus(commands_list: list[list[str]], available_gpus: list[str], working_dir: Path):
    running = {gpu: None for gpu in available_gpus}
    running_cmd = {gpu: None for gpu in available_gpus}
    queue = commands_list.copy()
    while queue or any(proc is not None for proc in running.values()):
        for gpu in available_gpus:
            proc = running[gpu]
            if proc is not None and proc.poll() is not None:
                if proc.returncode != 0:
                    raise RuntimeError(f"[GPU {gpu}] Falhou: {' '.join(running_cmd[gpu] or [])}")
                print(f'[GPU {gpu}] Tarefa concluída.')
                running[gpu] = None
                running_cmd[gpu] = None
        for gpu in available_gpus:
            if running[gpu] is None and queue:
                cmd = queue.pop(0)
                print(f"[GPU {gpu}] Iniciando: {' '.join(cmd)}")
                running[gpu] = spawn_on_gpu(cmd, gpu, working_dir)
                running_cmd[gpu] = cmd
        time.sleep(3)


def build_scenario_names() -> list[str]:
    return ['base', 'mcar_10pct', 'mcar_20pct', 'mcar_30pct', 'mcar_40pct', 'mcar_50pct', 'mcar_60pct', 'mcar_70pct', 'mcar_80pct']

def build_tuning_commands(config_path: Path, out_dir: Path, only_model: str, run_baselines: bool, n_workers: int) -> list[list[str]]:
    run_saits = 'true' if only_model in {'saits', 'both'} else 'false'
    run_imputeformer = 'true' if only_model in {'imputeformer', 'both'} else 'false'
    run_transformer = 'true' if only_model in {'transformer', 'both'} else 'false'
    run_baselines_flag = 'true' if (run_baselines or only_model == 'baselines') else 'false'

    base_cmd = [
        sys.executable,
        'run_models.py',
        '--action', 'tune_worker',
        '--config', str(config_path),
        '--out_dir', str(out_dir),
        '--model_group', 'all',
        '--run_saits', run_saits,
        '--run_imputeformer', run_imputeformer,
        '--run_baselines', run_baselines_flag,
        '--run_transformer', run_transformer,
    ]
    return [base_cmd.copy() for _ in range(n_workers)]

def build_commands(config_path: Path, out_dir: Path, only_model: str, run_baselines: bool) -> list[list[str]]:
    commands: list[list[str]] = []
    for scenario_name in build_scenario_names():
        run_saits = 'true' if only_model in {'saits', 'both'} else 'false'
        run_imputeformer = 'true' if only_model in {'imputeformer', 'both'} else 'false'
        run_transformer = 'true' if only_model in {'transformer', 'both'} else 'false'
        run_baselines_flag = 'true' if (run_baselines or only_model == 'baselines') else 'false'
        commands.append([
            sys.executable,
            'run_models.py',
            '--action', 'scenario',
            '--scenario', scenario_name,
            '--config', str(config_path),
            '--out_dir', str(out_dir),
            '--model_group', 'all',
            '--run_saits', run_saits,
            '--run_imputeformer', run_imputeformer,
            '--run_transformer', run_transformer,
            '--run_baselines', run_baselines_flag,
        ])
    return commands


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', type=str, default='auto')
    parser.add_argument('--root_dir', type=str, default='.')
    parser.add_argument('--config', type=str, default='experimento_base.json')
    parser.add_argument('--results_root', type=str, default='resultados')
    parser.add_argument('--only_model', choices=['saits', 'imputeformer', 'transformer', 'baselines', 'both'], default='both')
    parser.add_argument('--run_baselines', action='store_true')
    parser.add_argument('--run_tuning_phase', action='store_true')
    args = parser.parse_args()

    project_root = Path(args.root_dir).expanduser().resolve()
    available_gpus = detect_available_gpus() if args.gpus.strip().lower() == 'auto' else [g.strip() for g in args.gpus.split(',') if g.strip()]
    if not available_gpus:
        raise RuntimeError('Nenhuma GPU detectada/informada.')

    config_path = (project_root / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config)
    results_root = (project_root / args.results_root).resolve() if not Path(args.results_root).is_absolute() else Path(args.results_root)
    out_dir = results_root / f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.run_tuning_phase:
        tune_commands = build_tuning_commands(
            config_path=config_path,
            out_dir=out_dir,
            only_model=args.only_model,
            run_baselines=args.run_baselines,
            n_workers=len(available_gpus),
        )
        run_commands_on_gpus(tune_commands, available_gpus, project_root)

        subprocess.run([
            sys.executable,
            'run_models.py',
            '--action', 'collect_tuning',
            '--config', str(config_path),
            '--out_dir', str(out_dir),
            '--model_group', 'all',
            '--run_saits', 'true' if args.only_model in {'saits', 'both'} else 'false',
            '--run_imputeformer', 'true' if args.only_model in {'imputeformer', 'both'} else 'false',
            '--run_baselines', 'true' if (args.run_baselines or args.only_model == 'baselines') else 'false',
            '--run_transformer', 'true' if args.only_model in {'transformer', 'both'} else 'false',
        ], cwd=str(project_root), check=True)

    commands = build_commands(config_path, out_dir, args.only_model, args.run_baselines)
    run_commands_on_gpus(commands, available_gpus, project_root)
    subprocess.run([sys.executable, 'run_models.py', '--action', 'consolidate', '--config', str(config_path), '--out_dir', str(out_dir)], cwd=str(project_root), check=True)
    print(f'Finalizado. Resultados em: {out_dir}')


if __name__ == '__main__':
    main()
