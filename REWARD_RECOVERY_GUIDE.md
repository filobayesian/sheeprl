# SheepRL Reward Log Recovery Guide

This guide helps you recover reward logs from past SheepRL training runs. You now have tools to extract reward data from multiple log formats.

## Quick Start

1. **List available runs:**
   ```bash
   python recover_reward_logs.py --list_runs
   ```

2. **Recover rewards from a specific run:**
   ```bash
   python recover_reward_logs.py --run_path "logs/runs/dreamer_v3/PongNoFrameskip-v4/dv3_pong" --format all --output_dir recovered_data
   ```

3. **Analyze recovered data:**
   ```bash
   python analyze_rewards.py recovered_data/rewards_memmap.csv
   ```

## Available Log Formats

Your SheepRL runs store rewards in multiple formats:

### 1. TensorBoard Logs (.tfevents files)
- **Location:** `logs/runs/*/version_*/events.out.tfevents.*`
- **Contains:** Aggregated metrics like `Rewards/rew_avg`, episode length, losses
- **Best for:** Training curves, performance metrics over time
- **Requires:** `pip install tensorflow`

### 2. Memory-Mapped Buffer Files (.memmap)
- **Location:** `logs/runs/*/version_*/memmap_buffer/rank_*/env_*/rewards.memmap`
- **Contains:** Raw step-by-step reward values
- **Best for:** Detailed reward analysis, episode reconstruction
- **Requires:** Only numpy/pandas (built-in)

### 3. Weights & Biases Logs
- **Location:** `logs/runs/*/wandb/`
- **Contains:** Rich metrics, plots, system info
- **Best for:** Interactive analysis, sharing results
- **Requires:** `pip install wandb`

## Example Analysis Results

From your Pong training run (`jepa_pong_light`):
- **Total steps:** 2,000,000
- **Episodes with rewards:** 40 episodes
- **Average reward per episode:** -1.78 (mostly losses)
- **Best episode:** +1.88 reward
- **Total cumulative reward:** -71.25

This shows the agent was learning but still struggling with Pong (negative rewards indicate losing).

## Installation

```bash
# Basic functionality (memmap + analysis)
pip install pandas numpy

# Full functionality (all log formats)
pip install -r requirements_recovery.txt
```

## Usage Examples

### 1. Explore Your Runs
```bash
# See all available runs
python recover_reward_logs.py --list_runs

# Focus on specific algorithm
python recover_reward_logs.py --list_runs | grep dreamer_v3_jepa
```

### 2. Recover Different Data Types

```bash
# Get TensorBoard metrics (training curves)
python recover_reward_logs.py --run_path "logs/runs/dreamer_v3/PongNoFrameskip-v4/dv3_pong" --format tensorboard

# Get raw step data (detailed episode analysis)  
python recover_reward_logs.py --run_path "logs/runs/dreamer_v3/PongNoFrameskip-v4/dv3_pong" --format memmap

# Get everything
python recover_reward_logs.py --run_path "logs/runs/dreamer_v3/PongNoFrameskip-v4/dv3_pong" --format all
```

### 3. Analysis and Visualization

```bash
# Basic statistics
python analyze_rewards.py recovered_data/rewards_memmap.csv

# With plotting (requires matplotlib)
python recover_reward_logs.py --run_path "logs/runs/dreamer_v3/PongNoFrameskip-v4/dv3_pong" --plot --output_dir plots
```

## What Each Format Gives You

| Format | Pros | Cons | Use Case |
|--------|------|------|----------|
| **TensorBoard** | Aggregated metrics, easy to plot | Requires TensorFlow, less detail | Training curves, performance trends |
| **Memmap** | Raw data, no dependencies | Large files, needs processing | Episode analysis, reward distribution |
| **W&B** | Rich interface, easy sharing | Requires account/setup | Interactive exploration, collaboration |

## Troubleshooting

### "No reward data found"
- Check if the run completed successfully
- Try different format types (`--format all`)
- Verify the run path exists

### "TensorFlow not available"
- Install: `pip install tensorflow`
- Or use `--format memmap` to skip TensorBoard logs

### Large files taking too long
- Use `--format tensorboard` for aggregated data only
- Process memmap files in chunks for very long runs

## Your Available Runs

From the scan of your logs directory, you have **29 training runs** including:

**Dreamer V3 runs:**
- 4 discrete_dummy environment runs
- 5 PongNoFrameskip-v4 runs (including `dv3_pong`, `dv3_pong_light`)

**Dreamer V3 + JEPA runs:**
- 6 discrete_dummy environment runs  
- 13 PongNoFrameskip-v4 runs (including `jepa_pong`, `jepa_pong_light`, etc.)

Most runs have both TensorBoard and memmap data available, giving you multiple recovery options.

## Next Steps

1. **Choose your run:** Pick from the 29 available runs using `--list_runs`
2. **Recover data:** Use the appropriate format for your analysis needs
3. **Analyze:** Use the analysis script or load CSV data into your preferred tool
4. **Visualize:** Create plots to understand training progress and reward patterns

The recovery tools are now ready to use! You can recover rewards from any of your past training runs.
