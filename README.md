# QWOP Lab

A small restart of the QWOP experiment: reproducible evaluation, verified video replay,
and a controlled PPO pilot. The aim is both better running and evidence we can trust.

This first milestone compares two equal-budget fine-tuning runs from the same historical
PPO checkpoint. The treatment changes **only the per-step time penalty from 10 to 30**.
The control continues training with the original reward. This is a single-seed development
pilot, not yet a test of whether an AI researcher beats ordinary search.

## Setup

Windows reference setup: Python 3.12, Chrome, FFmpeg on PATH. An NVIDIA GPU is optional;
the reference runtime uses CPU Torch and one Torch thread for the small MLPs.

```powershell
powershell -ExecutionPolicy Bypass -File scripts/bootstrap.ps1
.venv/Scripts/python.exe -m pytest -q
```

`requirements.lock.txt` pins the tested runtime and development dependencies. Bootstrap
downloads the game from its author's website, applies qwop-gym's patch, resolves a matching
ChromeDriver, and downloads the two historical checkpoints from an immutable Git commit.
The proprietary game source stays in ignored local runtime files; it is not redistributed.

`game-source.lock.json` and `checkpoints.lock.json` identify the source artifacts by SHA-256.
`.runtime/runtime.json` identifies the local browser, driver, and game files. Runs fail if
these files change. A Chrome update requires an explicit re-bootstrap and creates a different
runtime contract; old replays are not silently accepted under the new contract.

If Chrome discovery needs help:

```powershell
.venv/Scripts/python.exe -m qwop_lab.cli bootstrap --browser 'C:/Program Files/Google/Chrome/Application/chrome.exe'
```

## Run the bounded pilot

```powershell
.venv/Scripts/python.exe -m qwop_lab.cli pilot --out runs/my-pilot
```

Defaults: 131,072 training steps per arm, training seed 42, six development cases per
arm, and two verified videos. The shared cap is **332,144 calls to env.step()**, including
training, validation, and video replay. Existing output directories are never reused.
No paid model API is used. The command stops after this one pilot.

Outputs include:

- `control/` and `treatment/`: effective configuration, source/runtime provenance,
  initial/final SB3 checkpoints, and episode-level training JSONL.
- `control-eval/` and `treatment-eval/`: per-step observations represented by hashes,
  actions, distances, clocks, termination flags, and summary statistics.
- `comparison.json`: paired cases, completion/speed summaries, limitations, budget.
- `comparison.mp4`: both arms on the same predetermined seed/reset case. The shorter
  run freezes at its final frame while the longer run finishes.
- `budget.sqlite`: a shared ledger, safe across independent worker processes.

To test the plumbing with less training, use `--steps 8192`. This is not enough to make
a useful performance claim. Step counts must be positive multiples of 512.

## Inspect the historical models

```powershell
.venv/Scripts/python.exe -m qwop_lab.cli evaluate --checkpoint .runtime/checkpoints/original.pt --out runs/history-original --label original --ledger runs/history-budget.sqlite --cap 50000
.venv/Scripts/python.exe -m qwop_lab.cli evaluate --checkpoint .runtime/checkpoints/speed.pt --out runs/history-speed --label speed --ledger runs/history-budget.sqlite --cap 50000
.venv/Scripts/python.exe -m qwop_lab.cli replay --input runs/history-original/seed-101-soft-0.json --out runs/history-original.mp4 --label 'Historical original PPO' --ledger runs/history-budget.sqlite --cap 50000
```

For standalone controlled training, see `qwop-lab train --help`. The adapter checks that
legacy weights are present; it never substitutes a random untrained policy. Tests verify
that importing legacy actor/critic weights into SB3 preserves logits and value predictions.

## What the benchmark means

- Physics: original patched QWOP through `qwop-gym==1.0.1`, four frames per action,
  the full 16-action enumeration, 60 normalized float32 observations, 5,000-step timeout.
- Completion: the native environment success flag at termination, including its existing
  fallback after 105 metres. First observed crossing of 100 m is recorded separately.
- Time: both raw `info.time` and `score_time = info.time * 10` are stored. The latter
  is the old repository's comparison convention. The game's displayed timer uses a
  different scale. Video duration follows converted score-time increments at 30 fps,
  plus one second of final hold. Wall-clock training time is a separate measurement.
- Reset: hard reset with an explicit seed, then zero or one explicit soft resets.
  Merely labeling episodes with different seeds does not imply different physical states.
- Evaluation: deterministic actions for every model. Reports count unique initial
  observation hashes, action sequences, and raw traces. Clock jitter can make raw traces
  unique even when physical behavior repeats; action-sequence counts expose that issue.
- Replay: every body-state hash, distance, and termination flag must match. The native
  reset introduces a small wall-clock-derived timer offset even after a hard reset. We
  subtract that single offset for clock verification and allow 5e-6 raw-time units for
float32 rounding; all 60 normalized observation floats must match exactly. Video hashes and the
  offset are recorded in adjacent JSON. Any divergence aborts and removes the partial MP4.
- Budget: workers reserve blocks of 256 steps transactionally before sending actions.
  Clean exits release unused reservations. A crashed worker can overcharge a block but
  cannot make spent steps disappear. Ambiguous WebSocket responses abort rather than
  resending an action and silently advancing physics twice.

## What this does not establish

The first pilot has one training seed and development cases with very limited physical
diversity. It has no held-out final test and does not measure an AI research process.
It optimizes speed/completion; convincing gait quality still requires reviewing videos.
The historical scores must be remeasured under the pinned runtime rather than copied from
old JSON files. Do not interpret faster training or one better checkpoint as a general result.

Next decisions should follow the pilot: repeated training seeds, a genuinely varied and
specified robustness suite, QR-DQN as another baseline, and only then AI-guided versus
ordinary search under matched total budgets. Exact mid-episode training resume is not
implemented; interrupted SB3 checkpoints preserve optimizer/model state but not the live
game state. Use a new run and label any continuation explicitly.

## Layout and checks

`qwop_lab/` contains the browser/environment adapter, ledger, model adapters, evaluator,
video exporter, maintained PPO trainer, and pilot runner. `tests/` exercises accounting,
reset history, model conversion, summary/replay selection, divergence detection, and timing.

```powershell
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m ruff check qwop_lab tests
.venv/Scripts/python.exe -m ruff format --check qwop_lab tests
.venv/Scripts/python.exe -m pip check
```

The original review is retained in `REPO_REVIEW.md`. Local review clones are ignored and
are not required by the new package. Bootstrap recovers its reference checkpoints itself.
