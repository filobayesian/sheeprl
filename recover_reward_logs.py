#!/usr/bin/env python3
"""
SheepRL Reward Log Recovery Script

This script helps recover reward logs from past SheepRL training runs.
It supports multiple log formats:
1. TensorBoard event files (.tfevents)
2. Memory-mapped buffer files (.memmap)  
3. Weights & Biases logs (if available)

Usage:
    python recover_reward_logs.py --run_path <path_to_run_directory>
    python recover_reward_logs.py --list_runs  # List available runs
    python recover_reward_logs.py --run_path <path> --format tensorboard
    python recover_reward_logs.py --run_path <path> --format memmap
    python recover_reward_logs.py --run_path <path> --format all
"""

import argparse
import os
import sys
import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

# TensorBoard imports
try:
    from tensorflow.python.summary.summary_iterator import summary_iterator
    TENSORBOARD_AVAILABLE = True
except ImportError:
    print("Warning: TensorFlow not available. TensorBoard log reading will be disabled.")
    print("Install with: pip install tensorflow")
    TENSORBOARD_AVAILABLE = False

# W&B imports
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    print("Warning: Weights & Biases not available. W&B log reading will be disabled.")
    print("Install with: pip install wandb")
    WANDB_AVAILABLE = False

# Plotting imports
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    print("Warning: Matplotlib not available. Plotting will be disabled.")
    print("Install with: pip install matplotlib")
    MATPLOTLIB_AVAILABLE = False


class RewardLogRecovery:
    """Main class for recovering reward logs from SheepRL runs."""
    
    def __init__(self, logs_dir: str = "logs/runs"):
        """Initialize with the logs directory path."""
        self.logs_dir = Path(logs_dir)
        if not self.logs_dir.exists():
            raise FileNotFoundError(f"Logs directory not found: {logs_dir}")
    
    def list_available_runs(self) -> List[Dict[str, str]]:
        """List all available training runs."""
        runs = []
        
        # Search for all run directories
        for algo_dir in self.logs_dir.iterdir():
            if not algo_dir.is_dir():
                continue
                
            for env_dir in algo_dir.iterdir():
                if not env_dir.is_dir():
                    continue
                    
                for run_dir in env_dir.iterdir():
                    if not run_dir.is_dir():
                        continue
                    
                    # Check what log formats are available
                    formats = []
                    
                    # Check for TensorBoard logs
                    version_dirs = list(run_dir.glob("version_*"))
                    if version_dirs:
                        for version_dir in version_dirs:
                            if list(version_dir.glob("events.out.tfevents.*")):
                                formats.append("tensorboard")
                                break
                    
                    # Check for memmap buffers
                    if (run_dir / "version_0" / "memmap_buffer").exists():
                        formats.append("memmap")
                    
                    # Check for W&B logs
                    if (run_dir / "wandb").exists():
                        formats.append("wandb")
                    
                    if formats:
                        runs.append({
                            "algorithm": algo_dir.name,
                            "environment": env_dir.name,
                            "run_name": run_dir.name,
                            "path": str(run_dir),
                            "formats": formats
                        })
        
        return runs
    
    def read_tensorboard_rewards(self, run_path: str) -> Optional[pd.DataFrame]:
        """Extract reward data from TensorBoard event files."""
        if not TENSORBOARD_AVAILABLE:
            print("TensorFlow not available. Cannot read TensorBoard logs.")
            return None
        
        run_dir = Path(run_path)
        rewards_data = []
        
        # Look for event files in version directories
        for version_dir in run_dir.glob("version_*"):
            event_files = list(version_dir.glob("events.out.tfevents.*"))
            
            for event_file in event_files:
                try:
                    for event in summary_iterator(str(event_file)):
                        if event.summary:
                            for value in event.summary.value:
                                # Look for reward-related metrics
                                if any(keyword in value.tag.lower() for keyword in ['reward', 'rew_avg', 'episode']):
                                    rewards_data.append({
                                        'step': event.step,
                                        'wall_time': event.wall_time,
                                        'metric': value.tag,
                                        'value': value.simple_value
                                    })
                except Exception as e:
                    print(f"Error reading {event_file}: {e}")
                    continue
        
        if rewards_data:
            df = pd.DataFrame(rewards_data)
            return df
        else:
            print("No reward data found in TensorBoard logs.")
            return None
    
    def read_memmap_rewards(self, run_path: str) -> Optional[pd.DataFrame]:
        """Extract reward data from memory-mapped buffer files."""
        run_dir = Path(run_path)
        memmap_dir = run_dir / "version_0" / "memmap_buffer"
        
        if not memmap_dir.exists():
            print("No memmap buffer directory found.")
            return None
        
        all_rewards = []
        
        # Look for reward files in rank/env subdirectories
        for rank_dir in memmap_dir.glob("rank_*"):
            for env_dir in rank_dir.glob("env_*"):
                reward_file = env_dir / "rewards.memmap"
                
                if reward_file.exists():
                    try:
                        # Try to read the memmap file
                        # Note: You may need to adjust dtype and shape based on your specific setup
                        rewards = np.memmap(reward_file, dtype=np.float32, mode='r')
                        
                        for i, reward in enumerate(rewards):
                            all_rewards.append({
                                'step': i,
                                'rank': rank_dir.name,
                                'env': env_dir.name,
                                'reward': reward
                            })
                    except Exception as e:
                        print(f"Error reading {reward_file}: {e}")
                        continue
        
        if all_rewards:
            df = pd.DataFrame(all_rewards)
            return df
        else:
            print("No reward data found in memmap files.")
            return None
    
    def read_wandb_rewards(self, run_path: str) -> Optional[pd.DataFrame]:
        """Extract reward data from W&B logs."""
        if not WANDB_AVAILABLE:
            print("Weights & Biases not available. Cannot read W&B logs.")
            return None
        
        run_dir = Path(run_path)
        wandb_dir = run_dir / "wandb"
        
        if not wandb_dir.exists():
            print("No W&B directory found.")
            return None
        
        # Look for W&B run directories
        run_dirs = list(wandb_dir.glob("run-*"))
        
        if not run_dirs:
            print("No W&B run directories found.")
            return None
        
        # For now, just provide instructions for manual recovery
        print("W&B logs found. To recover data:")
        print("1. Use wandb.restore() to download the run data")
        print("2. Or access the W&B web interface")
        print("3. Run directories found:", [str(d) for d in run_dirs])
        
        return None
    
    def recover_rewards(self, run_path: str, format_type: str = "all") -> Dict[str, pd.DataFrame]:
        """Recover reward data from the specified run."""
        results = {}
        
        if format_type in ["all", "tensorboard"]:
            print("Reading TensorBoard logs...")
            tb_data = self.read_tensorboard_rewards(run_path)
            if tb_data is not None:
                results["tensorboard"] = tb_data
        
        if format_type in ["all", "memmap"]:
            print("Reading memmap buffer logs...")
            memmap_data = self.read_memmap_rewards(run_path)
            if memmap_data is not None:
                results["memmap"] = memmap_data
        
        if format_type in ["all", "wandb"]:
            print("Checking W&B logs...")
            wandb_data = self.read_wandb_rewards(run_path)
            if wandb_data is not None:
                results["wandb"] = wandb_data
        
        return results
    
    def save_rewards_csv(self, rewards_data: Dict[str, pd.DataFrame], output_dir: str):
        """Save recovered reward data to CSV files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for format_name, df in rewards_data.items():
            csv_path = output_path / f"rewards_{format_name}.csv"
            df.to_csv(csv_path, index=False)
            print(f"Saved {format_name} rewards to: {csv_path}")
    
    def plot_rewards(self, rewards_data: Dict[str, pd.DataFrame], output_dir: str = None):
        """Plot recovered reward data."""
        if not MATPLOTLIB_AVAILABLE:
            print("Matplotlib not available. Cannot create plots.")
            return
        
        fig, axes = plt.subplots(len(rewards_data), 1, figsize=(12, 4 * len(rewards_data)))
        if len(rewards_data) == 1:
            axes = [axes]
        
        for i, (format_name, df) in enumerate(rewards_data.items()):
            ax = axes[i]
            
            if format_name == "tensorboard":
                # Plot reward metrics from TensorBoard
                reward_metrics = df[df['metric'].str.contains('reward|rew_avg', case=False, na=False)]
                for metric in reward_metrics['metric'].unique():
                    metric_data = reward_metrics[reward_metrics['metric'] == metric]
                    ax.plot(metric_data['step'], metric_data['value'], label=metric, marker='o')
                ax.set_title(f"Rewards from TensorBoard")
                
            elif format_name == "memmap":
                # Plot rewards from memmap files
                for env in df['env'].unique():
                    env_data = df[df['env'] == env]
                    ax.plot(env_data['step'], env_data['reward'], label=f"{env}", alpha=0.7)
                ax.set_title(f"Rewards from Memmap Buffers")
            
            ax.set_xlabel("Step")
            ax.set_ylabel("Reward")
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_dir:
            plot_path = Path(output_dir) / "rewards_plot.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            print(f"Saved plot to: {plot_path}")
        
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Recover reward logs from SheepRL training runs")
    parser.add_argument("--logs_dir", default="logs/runs", help="Path to logs directory")
    parser.add_argument("--list_runs", action="store_true", help="List available runs")
    parser.add_argument("--run_path", help="Path to specific run directory")
    parser.add_argument("--format", choices=["tensorboard", "memmap", "wandb", "all"], 
                       default="all", help="Log format to read")
    parser.add_argument("--output_dir", help="Directory to save recovered data")
    parser.add_argument("--plot", action="store_true", help="Create plots of reward data")
    
    args = parser.parse_args()
    
    # Initialize recovery tool
    try:
        recovery = RewardLogRecovery(args.logs_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    # List available runs
    if args.list_runs:
        runs = recovery.list_available_runs()
        if not runs:
            print("No training runs found.")
            return
        
        print(f"Found {len(runs)} training runs:")
        print("-" * 80)
        for run in runs:
            print(f"Algorithm: {run['algorithm']}")
            print(f"Environment: {run['environment']}")
            print(f"Run: {run['run_name']}")
            print(f"Path: {run['path']}")
            print(f"Available formats: {', '.join(run['formats'])}")
            print("-" * 80)
        return
    
    # Recover rewards from specific run
    if args.run_path:
        if not Path(args.run_path).exists():
            print(f"Error: Run path does not exist: {args.run_path}")
            sys.exit(1)
        
        print(f"Recovering rewards from: {args.run_path}")
        print(f"Format: {args.format}")
        
        rewards_data = recovery.recover_rewards(args.run_path, args.format)
        
        if not rewards_data:
            print("No reward data could be recovered.")
            return
        
        # Print summary
        print("\nRecovered data summary:")
        for format_name, df in rewards_data.items():
            print(f"  {format_name}: {len(df)} data points")
            if format_name == "tensorboard":
                metrics = df['metric'].unique()
                print(f"    Metrics: {', '.join(metrics)}")
            elif format_name == "memmap":
                envs = df['env'].unique() if 'env' in df.columns else []
                print(f"    Environments: {', '.join(envs)}")
        
        # Save data if output directory specified
        if args.output_dir:
            recovery.save_rewards_csv(rewards_data, args.output_dir)
        
        # Create plots if requested
        if args.plot:
            recovery.plot_rewards(rewards_data, args.output_dir)
    
    else:
        print("Please specify --run_path or use --list_runs to see available runs.")
        print("Use --help for more information.")


if __name__ == "__main__":
    main()
