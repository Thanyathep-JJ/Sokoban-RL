# TODO

## 1. Retrain stage 2 seed 2 with Double DQN (done 2026-11-11)

Double DQN, 500k steps: best validation 82 %, test 81.0 % (original seed 2: 77 % / 71.5 %).
Added to the README as an extra row; the original table is unchanged.

## 2. Game modes with 2 and 3 boxes (kept for later)

`two_boxes` and `expert` exist in `campaign.DIFFICULTIES` but are hidden from the window
(`sokoban-rl gui --difficulty two_boxes` still opens them). There is no working AI for them yet:
a stage 3 (2-box) DQN run reached only 1-4 % validation after 350k steps (random: 14 %) and was
stopped. Ideas before trying again: a curriculum from very short 2-box levels, a smaller board
(4x4 floor), or reward shaping for moving a box closer to a free target.

## 3. Better AI for all board sizes: cause found (2026-11-11)

Diagnosis (val sets, greedy play): in 62-87 % of states the network rates a move that changes
nothing (walking into a wall or a blocked box) at least as high as the best real move. The true
gap is tiny (step penalty 0.1 with gamma 0.99, about 0.1-0.2 points), smaller than the network's
error, so it often picks the wasted move, the board stays the same, and it repeats it until the
50-step limit. On bigger boards the error is larger, loops dominate the replay buffer, values sink
(stage 6: median max-Q fell from ~8 to ~1 between 50k and 300k steps) and success drops.

Evidence that this is the main cause: skipping wasted moves at play time, with no retraining,
raises validation success for every model, e.g. stage 2 model 82 -> 93 % (4x4 floor) and
29 -> 39 % (5x5 floor); stage 5 model 41 -> 67 % (4x4) and 35 -> 45 % (5x5).

Done: `--mask` (play time) lifts the 3 stage 2 runs from 75.8 to 89.2 % test. Training with `--mask` is running (models_mask/s5_seed0, 2026-11-11 evening).
Original proposal: treat wasted moves as unavailable (invalid-action masking) both when acting and in
the Q target (max over real moves only), as an opt-in training flag so earlier results stay
reproducible. Then retrain stage 5 or 6 and compare on the same test sets.

## 4. Optional follow-ups

- **Stuck-move fix:** 50 of 59 validation failures are the agent repeating one move into a
  wall. Try skipping actions that leave the board unchanged at play time; measure on
  validation only and report it as a post-hoc change.
- **Harder levels:** generated levels are mostly 3–7 steps (almost none above 9).
- **GPU:** PyTorch here is the CPU build; the RTX 3050 needs the CUDA build plus a
  `--device` option in `dqn.py`. Time it before relying on it.
- **Stage 3** (7×7, 2 boxes): `make-levels --stage 3`, then train with
  `--init models/s2_seed0/best.pt --eps-start 0.5 --steps 1000000`.
