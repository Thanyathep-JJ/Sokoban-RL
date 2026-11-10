# TODO

## 1. Retrain stage 2 seed 2 with Double DQN (paused)

Seed 2 reached 77 % best validation success with vanilla DQN, below the ≥ 80 % stage 2
criterion (seeds 0 and 1 reached 82 %). The decision to retrain was made on validation only.

A Double DQN run was started on 2026-11-10 and stopped at 30k / 500k steps
(validation 13 % at that point, which is normal this early). Training cannot resume from a
checkpoint, so start over:

```bash
sokoban-rl train --stage 2 --seed 2 --double --steps 500000 --eps-decay 150000 \
    --threads 8 --out-dir models_double/s2_seed2
```

- Takes about 2–2.5 h on CPU, so keep the PC awake and plugged in.
- When it is done, evaluate it once on val and test:
  ```bash
  sokoban-rl evaluate --stage 2 --splits val test --agents dqn --seeds 2 --models-dir models_double
  ```
- Add it to the README as an **extra row** next to the original seed 2 (do not replace the
  original numbers), and say it is Double DQN with 500k steps.

## 2. Optional follow-ups

- **Stuck-move fix:** 50 of 59 validation failures are the agent repeating one move into a
  wall. Try skipping actions that leave the board unchanged at play time; measure on
  validation only and report it as a post-hoc change.
- **Harder levels:** generated levels are mostly 3–7 steps (almost none above 9).
- **GPU:** PyTorch here is the CPU build; the RTX 3050 needs the CUDA build plus a
  `--device` option in `dqn.py`. Time it before relying on it.
- **Stage 3** (7×7, 2 boxes): `make-levels --stage 3`, then train with
  `--init models/s2_seed0/best.pt --eps-start 0.5 --steps 1000000`.
