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

# In a second terminal while training runs: progress bar, percent, time left (Ctrl+C stops watching only)
sokoban-rl watch                                       # every unfinished run
sokoban-rl watch models_double/s2_seed2                # one specific run

# Final numbers: Random vs DQN (best.pt of each seed) vs BFS on val and test
sokoban-rl evaluate --stage 2 --splits val test --agents random bfs dqn --seeds 0 1 2

# Demo: AI solves one level (terminal animation + GIF)
sokoban-rl play --file data/example.txt --agent qtable --gif results/example.gif
sokoban-rl play --split test --idx 5 --agent dqn --gif results/test5.gif
sokoban-rl play --file my_level.txt --agent dqn        # draw your own 6x6 level

# Challenge: you play first, then the AI plays the same level and both are compared with BFS
sokoban-rl challenge                                   # random test level
sokoban-rl challenge --split test --idx 11             # a specific level (replay one you lost)
sokoban-rl challenge --file my_level.txt --agent qtable
```

In `challenge`, move with **W A S D** or the arrow keys, **Z** undoes, **R** restarts, **G** gives
up (asks to confirm, then shows the AI's answer), **Q** quits. When you win or give up it shows
your moves, the AI's moves and the shortest BFS answer side by side. Bumping into a wall is
not counted as a move for you, but it is counted for the AI, which is how its stuck-on-a-wall
failures show up.

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
  human.py       play a level yourself and compare with the AI (`challenge`)
  monitor.py     live progress bar for a running training (`watch`)
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
  box pairs); small maps only; the greedy DQN can get stuck repeating a move (see Results).

## Results

### Stage 1 — Q-Learning on the example level
All 3 seeds learn `URD` (3 steps, identical to BFS) in about 0.5 s each, from ~140 visited states.

### Stage 2 — DQN on random 6×6 levels (1 box)
`best.pt` of each seed (selected on validation), greedy play (ε = 0), mean ± std over seeds 0, 1, 2.
Random is also averaged over 3 seeds.

| Agent | Train (1,000) | Validation (100) | **Test (200)** | Steps ÷ BFS | Deadlock |
|---|---|---|---|---|---|
| Random | 22.8 ± 1.3 % | 23.3 ± 4.2 % | 27.2 ± 3.4 % | 5.85 | 26.2 % |
| **DQN** | 96.6 ± 0.0 % | 80.3 ± 2.4 % | **75.8 ± 3.2 %** | 1.01 | 0.0 % |
| BFS (optimal) | 100 % | 100 % | 100 % | 1.00 | 0 % |

Per-seed best validation: 82 %, 82 %, 77 %, so 2 of 3 seeds pass the ≥ 80 % stage 2 criterion.
Training time: ~93 min per seed for 300k steps, 3 seeds in parallel on CPU (4 threads each);
the plan's 15–30 min estimate was too optimistic.

![training curves](docs/s2_training.png)

**What the numbers say**

- **It learned to play, far above random:** 76 % on unseen test levels vs 27 % for random.
  When it solves a level it is almost always optimal (1.01 × BFS) and it never pushes a box
  into a corner.
- **It partly memorises the training levels:** 97 % on train vs 80 % validation / 76 % test.
  Validation is slightly optimistic because `best.pt` was chosen on it; test is the honest number.
- **Harder levels fail more:** on the test set (seed 0), success is 82 % for 3-step levels but
  only 25 % for 7-step levels.
- **Main failure mode is a loop, not a bad push:** of 59 validation failures across the
  3 seeds, 50 are the agent repeating one move into a wall and 9 are a two-move
  back-and-forth. Because a wall bump leaves the board unchanged, a deterministic greedy
  policy repeats the same choice until the 50-step limit.

| Optimal win (test #12, 6 steps) | Win with detour (test #14, 11 vs 7) | Loss: stuck on a wall (test #11) |
|---|---|---|
| ![](docs/test12.gif) | ![](docs/test14.gif) | ![](docs/test11.gif) |
