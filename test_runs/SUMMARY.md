# Phase 1 Hyperparameter Search Summary

**Environment**: dmc
**Total Trials**: 2
**Completed Trials**: 2
**Search Budget**: 2000 steps per trial

## Top Configurations

| Rank | Trial ID | Best Return | JEPA Coef | JEPA EMA | Erase Frac |
|------|----------|-------------|-----------|----------|------------|
| 1 | 0 | -inf | 1.0 | 0.992 | 0.6 |
| 2 | 1 | -inf | 1.0 | 0.996 | 0.4 |

## Best Command for Phase 2

```bash
sheeprl exp=dreamer_v3_jepa env=dmc \
  algo.jepa_coef=1.0 \
  algo.jepa_ema=0.992 \
  algo.jepa_mask.erase_frac=0.6 \
  algo.total_steps=20000
```
