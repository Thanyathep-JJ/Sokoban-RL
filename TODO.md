# TODO

## 1. Retrain stage 2 seed 2 with Double DQN (done 2026-11-11)

Double DQN, 500k steps: best validation 82 %, test 81.0 % (original seed 2: 77 % / 71.5 %).
Added to the README as an extra row; the original table is unchanged.

## 2. Optional follow-ups

- **Stuck-move fix:** 50 of 59 validation failures are the agent repeating one move into a
  wall. Try skipping actions that leave the board unchanged at play time; measure on
  validation only and report it as a post-hoc change.
- **Harder levels:** generated levels are mostly 3–7 steps (almost none above 9).
- **GPU:** PyTorch here is the CPU build; the RTX 3050 needs the CUDA build plus a
  `--device` option in `dqn.py`. Time it before relying on it.
- **Stage 3** (7×7, 2 boxes): `make-levels --stage 3`, then train with
  `--init models/s2_seed0/best.pt --eps-start 0.5 --steps 1000000`.
