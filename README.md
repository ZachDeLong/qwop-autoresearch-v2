# QWOP Lab

A small restart of the QWOP experiment: reproducible evaluation, verified video replay,
and a controlled PPO pilot. The aim is both better running and evidence we can trust.

The latest [gait-reward screen](results/gait-001/report.md) trains a native-reward
baseline and a posture-dependent reward for 2,097,152 steps each on paired seed 83.
Both finish both reset phases. The gait reward reduces knee proximity from 98.9%
to 63.2% and increases upright forward displacement from 0.165% to 1.717%, but takes
22.38% longer to finish. Both reviewed sample sets still show knee-scooting. This
reward remains an exploratory result, not an upright-running solution. The report
includes a [verified comparison video](results/gait-001/comparison.mp4), raw-pose
metrics, frozen source/models, and [all engineering costs](results/gait-engineering-001/report.md).

The first architecture research campaign compared fresh PPO training, a fixed
baseline, predefined architecture search, and two auditable AI proposals. See
[RESEARCH_PLAN.md](RESEARCH_PLAN.md) for budgets, methodology, commands, and limitations.

The first architecture campaign is recorded in
[results/architecture-001/report.md](results/architecture-001/report.md), with a frozen
protocol, an explicit transport-failure amendment, learning curves, and a hash-indexed
archive of checkpoints and raw trajectories.

The completed [fresh-seed replication](results/replication-002/report.md) trains the
selected kinematic architecture and original baseline from scratch on seeds 17, 29,
and 61 with 1,048,576 interactions per run. The custom design uses 48% fewer parameters,
wins one of three seed comparisons, and averages 1.46% slower. All final models finish
both reset phases, but all reviewed replays show knee-scooting. Baseline seed 29 required
an explicitly recorded continuation; the two uninterrupted pairs also give mixed results.
This tests the selected design, not a replicated AI research process.

The report's hash-indexed archive preserves the exact study source and failure evidence.
Subsequent maintenance adds bounded Windows JSON-write retries, browser shutdown cleanup,
and failure exit status for batch jobs. These fixes are outside the frozen study; its
source guards intentionally reject reruns with changed live code. Use the archived
source and pinned runtime to reproduce that study, or initialize a new protocol.

The earlier pilot compares two equal-budget fine-tuning runs from the same historical
PPO checkpoint. The treatment changes **only the per-step time penalty from 10 to 30**.
The control continues training with the original reward. This is a single-seed development
pilot, not yet a test of whether an AI researcher beats ordinary search.

## Setup

### macOS

Use Python 3.12 and an installed Chrome. The Windows lock includes `torch+cpu`,
which is not a macOS wheel; install the pinned project dependencies instead.
Each new gait study records its resolved local dependencies and runtime hashes.

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -e '.[dev,video]'
.venv/bin/python -m qwop_lab.cli bootstrap --browser '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'
.venv/bin/python -m pytest -q
```

Bootstrap reproduces the locked game's exact line endings on either platform.
The game is served from a temporary loopback HTTP server bound to `127.0.0.1`;
only the installed game directory is served, with directory listings disabled.
The server and browser close with the environment. The optional video extra
provides FFmpeg when it is not already on PATH.

### Gait reward experiment

```sh
.venv/bin/python -m qwop_lab.gait_study init --out runs/gait-001
.venv/bin/python -m qwop_lab.gait_study batch --out runs/gait-001
.venv/bin/python -m qwop_lab.gait_study status --out runs/gait-001
.venv/bin/python scripts/summarize_gait_training.py --out runs/gait-001
.venv/bin/python -m qwop_lab.gait_study archive --out runs/gait-001 --destination results/gait-001
```

Defaults: two concurrent fresh 128x128 PPO policies, paired training seed 83,
2,097,152 steps per arm, checkpoints every 524,288 steps, and two deterministic
reset phases per checkpoint. Including evaluation and final verified replay,
the total cap is **4,284,304 environment steps**. This is one exploratory seed
pair, not a replicated result. See [GAIT_EXPERIMENT.md](GAIT_EXPERIMENT.md).

The baseline keeps the native reward exactly. The treatment changes only its
positive forward-speed component using measured pelvis height, torso tilt, and
knee clearance. Both arms record identical gait diagnostics; evaluation always
uses native reward. Raw unclipped body poses are saved, so the audit can recompute
the measurements and replay can verify them as well as policy observations.

For an engineering smoke test, initialize a separate output with `--steps 8192
--interval 8192 --seed 84`. This is insufficient training for a behavioral claim.
Existing output directories are never reused and failed training is not restarted.

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
  subtract that single offset for clock verification and allow the larger of 5e-6 raw-time
units or the two float32 clocks' combined rounding bound (plus 1e-8 for reset rounding).
All 60 normalized observation floats must match exactly. Video hashes and the
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
