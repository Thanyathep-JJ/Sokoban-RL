# Sokoban-RL

An AI that teaches itself to play Sokoban with reinforcement learning. The game, level
generator and BFS solver are written from scratch (NumPy); the agents are a tabular
**Q-Learning** agent (stage 1) and a small-CNN **DQN** in PyTorch (stages 2–3).
BFS provides the optimal answer that the AI is compared against.

| Stage | Levels | Algorithm | Pass criterion |
|---|---|---|---|
| 1 | one fixed 6×6 level, 1 box | Q-Learning (table) | solves it with the BFS-optimal number of steps |
| 2 | random 6×6 levels, 1 box | DQN | ≥ 80 % solved on the validation set |
| 3 (optional) | random 7×7 levels, 2 boxes | DQN, warm-started from stage 2 | ≥ 50 % on validation |

Full design notes and the step-by-step build plan are in [PLAN.md](PLAN.md).

## Install

```bash
python -m venv .venv && .venv\Scripts\activate      # Windows (use source .venv/bin/activate elsewhere)
pip install -e ".[dev]"
pytest
```

## Run

All commands are run from the repository root.

```bash
# Stage 1: Q-Learning on the example level (expect: URD, 3 steps, matches BFS)
sokoban-rl q-learn --level data/example.txt --seeds 0 1 2

# Stage 2: generate 1,000 / 100 / 200 train / val / test levels, then train 3 seeds
sokoban-rl make-levels --stage 2
sokoban-rl train --stage 2 --seed 0 --steps 20000      # quick smoke test first
sokoban-rl train --stage 2 --seed 0                    # full run (also --seed 1, --seed 2)
sokoban-rl plot models/s2_seed0/log.csv models/s2_seed1/log.csv models/s2_seed2/log.csv

# Final numbers: Random vs DQN (best.pt of each seed) vs BFS on val and test
sokoban-rl evaluate --stage 2 --splits val test --agents random bfs dqn --seeds 0 1 2

# Demo: AI solves one level (terminal animation + GIF)
sokoban-rl play --file data/example.txt --agent qtable --gif results/example.gif
sokoban-rl play --split test --idx 5 --agent dqn --gif results/test5.gif
sokoban-rl play --file my_level.txt --agent dqn        # draw your own 6x6 level
```

`python -m sokoban_rl ...` works too. Training flags: `--double` (Double DQN),
`--init <weights>` (warm start), `--threads N`, `--eval-every N`; see `sokoban-rl train -h`.

Level symbols: `#` wall · `@` player · `+` player on target · `$` box · `.` target · `*` box on target.
Actions: `U D L R`; the example level is solved by `URD`.

## Layout

```
src/sokoban_rl/
  game.py        rules, level parsing, observation tensor, SokobanEnv
  solver.py      BFS oracle (shares move() with the game)
  levelgen.py    reverse-play level generator, rotation-aware de-duplication
  qlearning.py   stage 1
  dqn.py         stages 2-3: network, replay buffer, training loop
  policies.py    random / BFS / Q-table policies
  evaluation.py  success %, steps vs BFS, deadlock %
  render.py      terminal animation + GIF
  plotting.py    training curves
  cli.py         the `sokoban-rl` command
tests/           pytest suite
data/            example.txt (tracked); generated s{2,3}_{train,val,test}.json (git-ignored)
models/ results/ generated artifacts (git-ignored)
```

## Design notes

- **Rewards:** −0.1 per step, +1 box onto a target, −1 box off a target, +10 win,
  −5 and end on a corner deadlock.
- **Terminal vs time limit:** win/deadlock are terminal; the 50-step limit is a
  *truncation*, so the Q target still bootstraps there.
- **No leakage:** train/val/test use different seeds and are de-duplicated up to
  rotation and reflection (8 symmetries). The test split is for the final report only.
- **Observation:** 4 planes (wall, player, box, target) padded to 7×7, so the same
  network works for 6×6 and 7×7 levels.
- **Known limits:** only corner deadlocks are detected (not frozen boxes along walls or
  box pairs); small maps only; DQN training takes about 10 s per 1,000 steps on this CPU,
  i.e. ~50 min for 300k steps.

## Results

Not recorded yet. After training, paste the table from `results/s2_results.md` here.
