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

Done (2026-11-12): `--mask` is an opt-in flag for play time and training; earlier results stay
reproducible. At play time it lifts the 3 stage 2 runs from 75.8 to 89.2 % test. Training stage 5
with `--mask` (400k steps from the seed 2 Double DQN model) gives 85.5 % test, and 83 % on the 5x5
floor where stage 2 gets 27-42 %. The game now plays 5x5-floor levels with it (`--big-model`).
Masking only at play time was not enough for the bigger boards; masking in training was.

## 4. Optional follow-ups

- **AI for the designed levels (2026-11-12): tried, did not work.** Stage 7 (`make-levels --stage 7`,
  masked DQN warm-started from `models_mask/s5_seed0`, 500k steps) made the AI worse: 42.0 % vs
  53.0 % on the s7 test with `--mask` + `--avoid-repeats`; validation fell from 15 % to 3 % during
  training (see README). The game keeps the stage 5 model. Ideas that were not tried: reward
  shaping (distance of the box to the target), a curriculum from 1 to 4 box lines, a lower learning
  rate for fine-tuning, or learning from the BFS solutions (imitation first, DQN after).
- **GPU:** PyTorch here is the CPU build; the RTX 3050 needs the CUDA build plus a
  `--device` option in `dqn.py`. Time it before relying on it.
- **Stage 3** (7×7, 2 boxes): `make-levels --stage 3`, then train with
  `--init models/s2_seed0/best.pt --eps-start 0.5 --steps 1000000`.
