# JEPA Hyperparameter Search for Dreamer V3

This directory contains a comprehensive hyperparameter search system for the Joint Embedding Predictive Architecture (JEPA) module in Dreamer V3. The system performs a coarse, low-fidelity search to identify promising configurations for Phase 2 training.

## Overview

The hyperparameter search system implements:
- **Optuna-based optimization** with Hyperband/ASHA pruning
- **Early stopping** based on evaluation returns (not JEPA loss)
- **Comprehensive logging** with W&B integration
- **Artifact generation** for easy Phase 2 reproduction
- **Resumable studies** with SQLite storage

## Files

- `search_phase1.py` - Main hyperparameter search script
- `dreamer_v3_jepa_search.py` - Modified training loop with evaluation hooks
- `train_wrapper.py` - Training wrapper interface (legacy)
- `test_search.py` - Test suite for validation
- `requirements_search.txt` - Additional dependencies
- `README_HYPERPARAMETER_SEARCH.md` - This documentation

## Installation

1. Install additional dependencies:
```bash
pip install -r requirements_search.txt
```

2. Ensure you have the main SheepRL dependencies installed.

## Usage

### Basic Usage

```bash
python search_phase1.py \
  --env dmc \
  --full-steps 1000000 \
  --fidelity-frac 0.15 \
  --n-trials 20 \
  --eval-every 10000 \
  --eval-episodes 10 \
  --output-dir ./runs/phase1
```

### With W&B Logging

```bash
python search_phase1.py \
  --env dmc \
  --full-steps 1000000 \
  --fidelity-frac 0.15 \
  --n-trials 20 \
  --eval-every 10000 \
  --eval-episodes 10 \
  --output-dir ./runs/phase1 \
  --wandb-project jepa-search
```

### Dry Run Test

```bash
python search_phase1.py \
  --env dmc \
  --full-steps 20000 \
  --fidelity-frac 0.1 \
  --n-trials 4 \
  --eval-every 2000 \
  --eval-episodes 3 \
  --output-dir ./test_runs \
  --dry-run
```

## Command Line Arguments

### Required Arguments
- `--env`: Environment name (e.g., 'dmc', 'atari')
- `--full-steps`: Full training steps for Phase 2

### Search Parameters
- `--fidelity-frac`: Fraction of full steps for Phase 1 (0.1-0.2, default: 0.15)
- `--n-trials`: Number of trials to run (default: 20)
- `--eval-every`: Evaluate every N steps (default: 10000)
- `--eval-episodes`: Number of evaluation episodes (default: 10)

### Output and Logging
- `--output-dir`: Output directory for results (default: ./runs/phase1)
- `--wandb-project`: Weights & Biases project name (optional)
- `--tensorboard`: Enable TensorBoard logging (flag)

### Search Strategy
- `--sampler`: Optuna sampler ('random' or 'tpe', default: 'random')
- `--pruner`: Optuna pruner ('hyperband' or 'asha', default: 'hyperband')

### Reproducibility
- `--seed0`: Base seed for trials (default: 0)

## Search Space

The search explores the following JEPA hyperparameters:

### Variable Parameters
- `jepa_coef`: Categorical {0.3, 1.0, 3.0}
- `jepa_ema`: Categorical {0.992, 0.996, 0.999}
- `jepa_mask.erase_frac`: Categorical {0.4, 0.6}

### Fixed Parameters
- `jepa_mask.vec_dropout`: 0.2
- `jepa_proj_dim`: 1024
- `jepa_hidden`: 1024

All other Dreamer/world-model/actor-critic parameters remain unchanged from the baseline.

## Output Artifacts

The search generates the following files in the output directory:

### Core Results
- `results.csv`: All trial results with hyperparameters and metrics
- `topk.json`: Top 4-6 configurations sorted by best evaluation return
- `best_config.yaml`: Best configuration for Phase 2 reproduction
- `SUMMARY.md`: Human-readable summary with reproduction commands

### Per-Trial Data
- `trial_{id}/results.json`: Individual trial results
- `trial_{id}/history.csv`: Evaluation history (step, eval_return, smoothed_return)
- `trial_{id}/training.log`: Training logs (if enabled)

### Study Data
- `study.db`: SQLite database for resumable studies

## Evaluation and Pruning

### Evaluation Strategy
- Evaluates every `--eval-every` environment steps
- Uses `--eval-episodes` episodes per evaluation
- Computes mean evaluation return across episodes

### Smoothing
- Maintains moving average over last 3 evaluations
- Uses mean of available evaluations if < 3 available
- Reports smoothed values to Optuna for pruning decisions

### Pruning
- **Hyperband**: Multi-armed bandit with successive halving
- **ASHA**: Asynchronous successive halving algorithm
- Grace period: 20-30% of trial budget before pruning becomes active
- Prunes based on smoothed evaluation returns

## Integration with Existing Code

The search system integrates with the existing SheepRL codebase:

1. **Uses existing Dreamer V3 JEPA implementation** with minimal modifications
2. **Leverages existing configuration system** with Hydra
3. **Compatible with existing environments** and wrappers
4. **Maintains existing logging and checkpointing** infrastructure

## Testing

Run the test suite to validate the implementation:

```bash
python test_search.py
```

The test suite validates:
- Configuration loading
- Optuna setup
- Dry-run execution
- Output artifact generation

## Example Workflow

### Phase 1: Hyperparameter Search
```bash
# Run search
python search_phase1.py \
  --env dmc \
  --full-steps 1000000 \
  --fidelity-frac 0.15 \
  --n-trials 20 \
  --eval-every 10000 \
  --output-dir ./runs/phase1 \
  --wandb-project jepa-search

# Check results
cat ./runs/phase1/SUMMARY.md
```

### Phase 2: Full Training
```bash
# Use best configuration from Phase 1
sheeprl exp=dreamer_v3_jepa env=dmc \
  algo.jepa_coef=1.0 \
  algo.jepa_ema=0.996 \
  algo.jepa_mask.erase_frac=0.6 \
  algo.total_steps=1000000
```

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch size or use fewer environments
2. **Evaluation Never Runs**: Check environment setup and evaluation frequency
3. **All Trials Pruned**: Increase grace period or reduce pruning aggressiveness
4. **Study Not Resuming**: Check SQLite database permissions

### Debug Mode

Enable verbose logging:
```bash
python search_phase1.py --env dmc --full-steps 100000 --n-trials 2 --eval-every 5000
```

### Performance Tips

1. **Use appropriate fidelity**: 0.15 is usually sufficient for initial screening
2. **Optimize evaluation frequency**: Balance between pruning accuracy and overhead
3. **Use W&B for monitoring**: Helps identify issues early
4. **Run on appropriate hardware**: GPU recommended for faster training

## Advanced Usage

### Custom Search Spaces

Modify the search space in `search_phase1.py`:
```python
# In create_objective_function
jepa_coef = trial.suggest_categorical("jepa_coef", [0.1, 0.5, 1.0, 2.0, 5.0])
jepa_ema = trial.suggest_categorical("jepa_ema", [0.99, 0.995, 0.999, 0.9995])
```

### Multi-Objective Optimization

Extend the objective function to optimize multiple metrics:
```python
def objective(trial):
    # ... training code ...
    return {
        "eval_return": best_return,
        "training_time": wall_time,
        "memory_usage": peak_memory
    }
```

### Custom Pruning Strategies

Implement custom pruning logic:
```python
def custom_pruner(trial, step, value):
    # Custom pruning logic
    if step > 10000 and value < threshold:
        raise optuna.TrialPruned()
```

## Contributing

When extending the hyperparameter search system:

1. **Maintain backward compatibility** with existing configurations
2. **Add comprehensive tests** for new functionality
3. **Update documentation** for new features
4. **Follow existing code style** and patterns
5. **Validate with dry-run tests** before deployment

## License

This hyperparameter search system follows the same license as the main SheepRL project.
