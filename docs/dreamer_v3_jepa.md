### DreamerV3-JEPA (reconstruction-free encoder training)

What changed: replace pixel reconstruction with a JEPA objective on encoder features; dynamics, actor, critic unchanged.

Commands:

```bash
python sheeprl.py exp=dreamer_v3_jepa env=atari
python sheeprl.py exp=dreamer_v3_jepa env=dmc
python sheeprl_eval.py checkpoint_path=/path/to/ckpt.pt env=atari
```

Ablations: `algo.jepa_coef`, `algo.jepa_ema`, masking on/off; compare to DreamerV3 with identical seeds/steps.

Repro: list seeds, env versions, Fabric precision, DDP world size.


