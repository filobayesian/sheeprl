# JEPA Hyperparameter Search Implementation 


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

