#!/usr/bin/env python3
"""
Phase 1 Hyperparameter Search for Dreamer V3 JEPA

This script performs a coarse, low-fidelity hyperparameter search for JEPA parameters
using Optuna with Hyperband/ASHA pruning. It runs 16-24 short trials and ranks them
by evaluation return for Phase 2 selection.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import hydra
import numpy as np
import optuna
import torch
import wandb
from omegaconf import DictConfig, OmegaConf

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

from dreamer_v3_jepa_search import train_with_eval_search


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Phase 1 JEPA Hyperparameter Search")
    
    # Required arguments
    parser.add_argument("--env", type=str, required=True, help="Environment name (e.g., 'dmc', 'atari')")
    parser.add_argument("--full-steps", type=int, required=True, help="Full training steps for Phase 2")
    
    # Search parameters
    parser.add_argument("--fidelity-frac", type=float, default=0.15, 
                       help="Fraction of full steps for Phase 1 (0.1-0.2)")
    parser.add_argument("--n-trials", type=int, default=20, help="Number of trials to run")
    parser.add_argument("--eval-every", type=int, default=10000, help="Evaluate every N steps")
    parser.add_argument("--eval-episodes", type=int, default=10, help="Number of evaluation episodes")
    
    # Output and logging
    parser.add_argument("--output-dir", type=str, default="./runs/phase1", 
                       help="Output directory for results")
    parser.add_argument("--wandb-project", type=str, help="Weights & Biases project name")
    parser.add_argument("--tensorboard", action="store_true", help="Enable TensorBoard logging")
    
    # Search strategy
    parser.add_argument("--sampler", type=str, default="random", choices=["random", "tpe"],
                       help="Optuna sampler")
    parser.add_argument("--pruner", type=str, default="hyperband", choices=["hyperband", "asha"],
                       help="Optuna pruner")
    
    # Reproducibility
    parser.add_argument("--seed0", type=int, default=0, help="Base seed for trials")
    
    # Dry run
    parser.add_argument("--dry-run", action="store_true", help="Run a quick dry run test")
    
    return parser.parse_args()


def validate_args(args):
    """Validate command line arguments."""
    if not 0.1 <= args.fidelity_frac <= 0.2:
        raise ValueError(f"fidelity_frac must be in [0.1, 0.2], got {args.fidelity_frac}")
    
    if args.n_trials < 1:
        raise ValueError(f"n_trials must be >= 1, got {args.n_trials}")
    
    if args.full_steps <= 0:
        raise ValueError(f"full_steps must be > 0, got {args.full_steps}")


def load_base_config(env_name: str) -> DictConfig:
    """Load base configuration for the environment."""
    # Load the base Dreamer V3 JEPA config
    config_path = Path(__file__).parent / "sheeprl" / "configs"
    
    # Create a minimal config for the search
    base_cfg = {
        "algo": {
            "name": "dreamer_v3_jepa",
            "jepa_coef": 1.0,
            "jepa_ema": 0.996,
            "jepa_proj_dim": 1024,
            "jepa_hidden": 1024,
            "jepa_mask": {
                "erase_frac": 0.6,
                "vec_dropout": 0.2
            }
        },
        "env": {
            "id": env_name,
            "num_envs": 1,
            "sync_env": True,
            "frame_stack": -1,
            "screen_size": 64,
            "grayscale": False,
            "clip_rewards": True,
            "action_repeat": 1,
            "max_episode_steps": 1000
        },
        "fabric": {
            "accelerator": "auto",
            "devices": 1,
            "num_nodes": 1,
            "strategy": "auto"
        },
        "buffer": {
            "size": 1000000,
            "memmap": False,
            "validate_args": False
        },
        "metric": {
            "log_level": 1,
            "log_every": 1000,
            "disable_timer": False
        },
        "checkpoint": {
            "every": 0,
            "save_last": False,
            "resume_from": None
        },
        "model_manager": {
            "disabled": True
        },
        "seed": 42,
        "num_threads": 1,
        "float32_matmul_precision": "high",
        "dry_run": False
    }
    
    return OmegaConf.create(base_cfg)


def create_objective_function(
    base_cfg: DictConfig,
    args,
    output_dir: Path,
    study_name: str
) -> callable:
    """Create the objective function for Optuna."""
    
    def objective(trial: optuna.Trial) -> float:
        """Objective function for a single trial."""
        trial_id = trial.number
        seed = args.seed0 + trial_id
        
        # Sample hyperparameters
        jepa_coef = trial.suggest_categorical("jepa_coef", [0.3, 1.0, 3.0])
        jepa_ema = trial.suggest_categorical("jepa_ema", [0.992, 0.996, 0.999])
        erase_frac = trial.suggest_categorical("jepa_erase_frac", [0.4, 0.6])
        
        # Fixed parameters
        vec_dropout = 0.2
        jepa_proj_dim = 1024
        jepa_hidden = 1024
        
        # Create trial configuration
        trial_cfg = OmegaConf.create(OmegaConf.to_yaml(base_cfg))
        trial_cfg.algo.jepa_coef = jepa_coef
        trial_cfg.algo.jepa_ema = jepa_ema
        trial_cfg.algo.jepa_mask.erase_frac = erase_frac
        trial_cfg.algo.jepa_mask.vec_dropout = vec_dropout
        trial_cfg.algo.jepa_proj_dim = jepa_proj_dim
        trial_cfg.algo.jepa_hidden = jepa_hidden
        trial_cfg.seed = seed
        
        # Calculate trial budget
        per_trial_steps = int(np.ceil(args.full_steps * args.fidelity_frac))
        
        # Set up trial directory
        trial_dir = output_dir / f"trial_{trial_id}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up W&B if requested
        if args.wandb_project:
            wandb.init(
                project=args.wandb_project,
                name=f"trial_{trial_id}",
                config={
                    "trial_id": trial_id,
                    "jepa_coef": jepa_coef,
                    "jepa_ema": jepa_ema,
                    "erase_frac": erase_frac,
                    "vec_dropout": vec_dropout,
                    "jepa_proj_dim": jepa_proj_dim,
                    "jepa_hidden": jepa_hidden,
                    "seed": seed,
                    "max_steps": per_trial_steps,
                    "eval_every": args.eval_every
                },
                reinit=True
            )
        
        try:
            # Run training
            start_time = time.time()
            
            # Use the training wrapper
            eval_history = train_with_eval_search(
                trial_cfg,
                max_env_steps=per_trial_steps,
                eval_every_steps=args.eval_every,
                seed=seed
            )
            
            wall_time = time.time() - start_time
            
            if not eval_history:
                print(f"Trial {trial_id} failed - no evaluation history")
                return -float('inf')
            
            # Compute smoothed returns and report to Optuna
            smoothed_returns = []
            for i, (step, ret) in enumerate(eval_history):
                if i < 2:
                    smoothed = np.mean([r for _, r in eval_history[:i+1]])
                else:
                    recent = [r for _, r in eval_history[max(0, i-2):i+1]]
                    smoothed = np.mean(recent)
                
                smoothed_returns.append(smoothed)
                
                # Report to Optuna for pruning
                trial.report(smoothed, step=step)
                
                # Check if trial should be pruned
                if trial.should_prune():
                    print(f"Trial {trial_id} pruned at step {step}")
                    raise optuna.TrialPruned()
            
            # Get best smoothed return
            best_return = max(smoothed_returns)
            best_step = eval_history[smoothed_returns.index(best_return)][0]
            
            # Save trial results
            trial_results = {
                "trial_id": trial_id,
                "state": "COMPLETE",
                "seed": seed,
                "env": args.env,
                "max_steps": per_trial_steps,
                "jepa_coef": jepa_coef,
                "jepa_ema": jepa_ema,
                "erase_frac": erase_frac,
                "vec_dropout": vec_dropout,
                "jepa_proj_dim": jepa_proj_dim,
                "jepa_hidden": jepa_hidden,
                "best_eval_return": best_return,
                "best_step": best_step,
                "wall_time_s": wall_time,
                "notes": ""
            }
            
            # Save trial results
            results_file = trial_dir / "results.json"
            with open(results_file, 'w') as f:
                json.dump(trial_results, f, indent=2)
            
            # Save evaluation history
            history_file = trial_dir / "history.csv"
            with open(history_file, 'w') as f:
                f.write("step,eval_return,smoothed_return\n")
                for i, ((step, ret), smoothed) in enumerate(zip(eval_history, smoothed_returns)):
                    f.write(f"{step},{ret:.4f},{smoothed:.4f}\n")
            
            # Log to W&B
            if args.wandb_project:
                wandb.log({
                    "best_eval_return": best_return,
                    "best_step": best_step,
                    "wall_time": wall_time,
                    "final_eval_return": eval_history[-1][1] if eval_history else 0
                })
                wandb.finish()
            
            print(f"Trial {trial_id} completed: best_return={best_return:.4f} at step {best_step}")
            return best_return
            
        except Exception as e:
            print(f"Trial {trial_id} failed with error: {e}")
            
            # Save failed trial results
            trial_results = {
                "trial_id": trial_id,
                "state": "FAILED",
                "seed": seed,
                "env": args.env,
                "max_steps": per_trial_steps,
                "jepa_coef": jepa_coef,
                "jepa_ema": jepa_ema,
                "erase_frac": erase_frac,
                "vec_dropout": vec_dropout,
                "jepa_proj_dim": jepa_proj_dim,
                "jepa_hidden": jepa_hidden,
                "best_eval_return": -float('inf'),
                "best_step": 0,
                "wall_time_s": time.time() - start_time,
                "notes": str(e)
            }
            
            results_file = trial_dir / "results.json"
            with open(results_file, 'w') as f:
                json.dump(trial_results, f, indent=2)
            
            if args.wandb_project:
                wandb.finish(exit_code=1)
            
            return -float('inf')
    
    return objective


def create_sampler(sampler_name: str) -> optuna.samplers.BaseSampler:
    """Create Optuna sampler."""
    if sampler_name == "random":
        return optuna.samplers.RandomSampler()
    elif sampler_name == "tpe":
        return optuna.samplers.TPESampler()
    else:
        raise ValueError(f"Unknown sampler: {sampler_name}")


def create_pruner(pruner_name: str, max_steps: int) -> optuna.pruners.BasePruner:
    """Create Optuna pruner."""
    grace_period = int(0.2 * max_steps)  # 20% grace period
    
    if pruner_name == "hyperband":
        return optuna.pruners.HyperbandPruner(
            min_resource=grace_period,
            max_resource=max_steps,
            reduction_factor=3
        )
    elif pruner_name == "asha":
        return optuna.pruners.SuccessiveHalvingPruner(
            min_resource=grace_period,
            reduction_factor=3
        )
    else:
        raise ValueError(f"Unknown pruner: {pruner_name}")


def save_results(study: optuna.Study, output_dir: Path, args):
    """Save search results and artifacts."""
    # Create results CSV
    results_data = []
    
    for trial in study.trials:
        trial_dir = output_dir / f"trial_{trial.number}"
        results_file = trial_dir / "results.json"
        
        if results_file.exists():
            with open(results_file, 'r') as f:
                trial_data = json.load(f)
            results_data.append(trial_data)
        else:
            # Create entry for failed/pruned trials
            results_data.append({
                "trial_id": trial.number,
                "state": trial.state.name,
                "seed": args.seed0 + trial.number,
                "env": args.env,
                "max_steps": int(np.ceil(args.full_steps * args.fidelity_frac)),
                "jepa_coef": trial.params.get("jepa_coef", 0),
                "jepa_ema": trial.params.get("jepa_ema", 0),
                "erase_frac": trial.params.get("jepa_erase_frac", 0),
                "vec_dropout": 0.2,
                "jepa_proj_dim": 1024,
                "jepa_hidden": 1024,
                "best_eval_return": trial.value if trial.value is not None else -float('inf'),
                "best_step": 0,
                "wall_time_s": 0,
                "notes": str(trial.state)
            })
    
    # Save results CSV
    import csv
    results_file = output_dir / "results.csv"
    with open(results_file, 'w', newline='') as f:
        if results_data:
            writer = csv.DictWriter(f, fieldnames=results_data[0].keys())
            writer.writeheader()
            writer.writerows(results_data)
    
    # Get top trials
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    completed_trials.sort(key=lambda t: t.value, reverse=True)
    
    # Save top K configs
    top_k = min(6, len(completed_trials))
    top_configs = []
    
    for i, trial in enumerate(completed_trials[:top_k]):
        config = {
            "rank": i + 1,
            "trial_id": trial.number,
            "best_eval_return": trial.value,
            "params": trial.params
        }
        top_configs.append(config)
    
    topk_file = output_dir / "topk.json"
    with open(topk_file, 'w') as f:
        json.dump(top_configs, f, indent=2)
    
    # Save best config
    if completed_trials:
        best_trial = completed_trials[0]
        best_config = OmegaConf.create({
            "algo": {
                "name": "dreamer_v3_jepa",
                "jepa_coef": best_trial.params["jepa_coef"],
                "jepa_ema": best_trial.params["jepa_ema"],
                "jepa_mask": {
                    "erase_frac": best_trial.params["jepa_erase_frac"],
                    "vec_dropout": 0.2
                },
                "jepa_proj_dim": 1024,
                "jepa_hidden": 1024
            },
            "env": {
                "id": args.env
            },
            "seed": args.seed0 + best_trial.number,
            "best_eval_return": best_trial.value
        })
        
        best_config_file = output_dir / "best_config.yaml"
        with open(best_config_file, 'w') as f:
            OmegaConf.save(best_config, f)
    
    # Create summary
    summary_file = output_dir / "SUMMARY.md"
    with open(summary_file, 'w') as f:
        f.write("# Phase 1 Hyperparameter Search Summary\n\n")
        f.write(f"**Environment**: {args.env}\n")
        f.write(f"**Total Trials**: {len(study.trials)}\n")
        f.write(f"**Completed Trials**: {len(completed_trials)}\n")
        f.write(f"**Search Budget**: {args.full_steps * args.fidelity_frac:.0f} steps per trial\n\n")
        
        f.write("## Top Configurations\n\n")
        f.write("| Rank | Trial ID | Best Return | JEPA Coef | JEPA EMA | Erase Frac |\n")
        f.write("|------|----------|-------------|-----------|----------|------------|\n")
        
        for config in top_configs:
            params = config["params"]
            f.write(f"| {config['rank']} | {config['trial_id']} | {config['best_eval_return']:.4f} | "
                   f"{params['jepa_coef']} | {params['jepa_ema']} | {params['jepa_erase_frac']} |\n")
        
        if completed_trials:
            f.write(f"\n## Best Command for Phase 2\n\n")
            f.write("```bash\n")
            f.write(f"sheeprl exp=dreamer_v3_jepa env={args.env} \\\n")
            f.write(f"  algo.jepa_coef={best_trial.params['jepa_coef']} \\\n")
            f.write(f"  algo.jepa_ema={best_trial.params['jepa_ema']} \\\n")
            f.write(f"  algo.jepa_mask.erase_frac={best_trial.params['jepa_erase_frac']} \\\n")
            f.write(f"  algo.total_steps={args.full_steps}\n")
            f.write("```\n")
    
    return results_data, top_configs


def print_summary(results_data: List[Dict], top_configs: List[Dict]):
    """Print a summary table to stdout."""
    print("\n" + "="*80)
    print("PHASE 1 HYPERPARAMETER SEARCH SUMMARY")
    print("="*80)
    
    if not results_data:
        print("No results to display.")
        return
    
    # Count by state
    states = {}
    for result in results_data:
        state = result["state"]
        states[state] = states.get(state, 0) + 1
    
    print(f"Total Trials: {len(results_data)}")
    for state, count in states.items():
        print(f"  {state}: {count}")
    
    if top_configs:
        print(f"\nTop {len(top_configs)} Configurations:")
        print("-" * 80)
        print(f"{'Rank':<4} {'Trial':<6} {'Return':<8} {'Coef':<6} {'EMA':<8} {'Erase':<6}")
        print("-" * 80)
        
        for config in top_configs:
            params = config["params"]
            print(f"{config['rank']:<4} {config['trial_id']:<6} "
                  f"{config['best_eval_return']:<8.4f} {params['jepa_coef']:<6} "
                  f"{params['jepa_ema']:<8} {params['jepa_erase_frac']:<6}")
    
    print("="*80)


def main():
    """Main function."""
    args = parse_args()
    validate_args(args)
    
    # Set up output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load base configuration
    base_cfg = load_base_config(args.env)
    
    # Calculate trial budget
    per_trial_steps = int(np.ceil(args.full_steps * args.fidelity_frac))
    
    print(f"Starting Phase 1 hyperparameter search...")
    print(f"Environment: {args.env}")
    print(f"Trials: {args.n_trials}")
    print(f"Steps per trial: {per_trial_steps}")
    print(f"Evaluation every: {args.eval_every} steps")
    print(f"Output directory: {output_dir}")
    
    # Create study
    study_name = f"jepa_phase1_{args.env}"
    storage_url = f"sqlite:///{output_dir}/study.db"
    
    sampler = create_sampler(args.sampler)
    pruner = create_pruner(args.pruner, per_trial_steps)
    
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        study_name=study_name,
        storage=storage_url,
        load_if_exists=True
    )
    
    # Create objective function
    objective = create_objective_function(base_cfg, args, output_dir, study_name)
    
    # Run optimization
    try:
        study.optimize(objective, n_trials=args.n_trials)
    except KeyboardInterrupt:
        print("\nSearch interrupted by user.")
    
    # Save results
    print("\nSaving results...")
    results_data, top_configs = save_results(study, output_dir, args)
    
    # Print summary
    print_summary(results_data, top_configs)
    
    print(f"\nResults saved to: {output_dir}")
    print("Files created:")
    print(f"  - results.csv: All trial results")
    print(f"  - topk.json: Top {len(top_configs)} configurations")
    print(f"  - best_config.yaml: Best configuration")
    print(f"  - SUMMARY.md: Human-readable summary")


if __name__ == "__main__":
    main()
