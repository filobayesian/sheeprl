#!/usr/bin/env python3
"""
Simple reward analysis script for recovered SheepRL logs.
Works with just pandas and basic Python - no TensorFlow required.
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path

def analyze_rewards_csv(csv_path: str):
    """Analyze reward data from a CSV file."""
    df = pd.read_csv(csv_path)
    
    print(f"Reward Data Analysis: {csv_path}")
    print("=" * 60)
    
    # Basic statistics
    print(f"Total data points: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    
    if 'reward' in df.columns:
        rewards = df['reward']
        print(f"\nReward Statistics:")
        print(f"  Mean reward: {rewards.mean():.4f}")
        print(f"  Std reward: {rewards.std():.4f}")
        print(f"  Min reward: {rewards.min():.4f}")
        print(f"  Max reward: {rewards.max():.4f}")
        print(f"  Total reward: {rewards.sum():.4f}")
        
        # Non-zero rewards
        nonzero_rewards = rewards[rewards != 0]
        if len(nonzero_rewards) > 0:
            print(f"\nNon-zero rewards ({len(nonzero_rewards)} points):")
            print(f"  Mean: {nonzero_rewards.mean():.4f}")
            print(f"  Std: {nonzero_rewards.std():.4f}")
            print(f"  Min: {nonzero_rewards.min():.4f}")
            print(f"  Max: {nonzero_rewards.max():.4f}")
        
        # Per environment analysis (if applicable)
        if 'env' in df.columns:
            print(f"\nPer-environment analysis:")
            for env in df['env'].unique():
                env_rewards = df[df['env'] == env]['reward']
                env_nonzero = env_rewards[env_rewards != 0]
                print(f"  {env}: {len(env_rewards)} steps, {len(env_nonzero)} non-zero rewards")
                if len(env_nonzero) > 0:
                    print(f"    Non-zero mean: {env_nonzero.mean():.4f}, sum: {env_nonzero.sum():.4f}")
    
    # Look for episode boundaries (reward spikes or patterns)
    if 'reward' in df.columns:
        rewards = df['reward']
        episode_ends = []
        
        # Simple heuristic: look for reward changes or non-zero rewards
        for i in range(1, len(rewards)):
            if rewards.iloc[i] != 0 or (i > 0 and rewards.iloc[i-1] != 0 and rewards.iloc[i] == 0):
                episode_ends.append(i)
        
        if episode_ends:
            print(f"\nPossible episode boundaries found: {len(episode_ends)}")
            if len(episode_ends) <= 20:  # Show first 20
                print("  Steps with rewards:", episode_ends[:20])
            else:
                print("  First 10 steps with rewards:", episode_ends[:10])
                print("  Last 10 steps with rewards:", episode_ends[-10:])

def main():
    parser = argparse.ArgumentParser(description="Analyze recovered reward CSV files")
    parser.add_argument("csv_file", help="Path to the CSV file with reward data")
    
    args = parser.parse_args()
    
    if not Path(args.csv_file).exists():
        print(f"Error: File not found: {args.csv_file}")
        return
    
    analyze_rewards_csv(args.csv_file)

if __name__ == "__main__":
    main()
