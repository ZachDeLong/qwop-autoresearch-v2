QWOP pilot 001 — September 15, 2026

**Result: the higher time penalty lost to the matched control in this pilot.**

| Model | Training in this pilot | Mean finish time | Finishes |
|---|---:|---:|---:|
| Historical original, remeasured | 0 | 167.74 s | 6/6 |
| Control: original time penalty 10 | 131,072 steps | 154.75 s | 6/6 |
| Treatment: time penalty 30 | 131,072 steps | 161.50 s | 6/6 |
| Historical speed model, remeasured | 0 | 133.25 s | 6/6 |

The treatment took 6.75 seconds longer than the control, approximately 4.36% more time.
Both arms started from the exact same historical checkpoint with seed 42, the same
128x128 Tanh architecture, optimizer settings, and training allowance. Only the time
penalty changed. The control improved without a new reward, demonstrating why extra
training needs its own comparison arm. The historical speed model remains fastest;
its earlier training budget and recipe are different, so it is a reference rather
than a matched control for this experiment.

**What is now working**

- A separate, pinned Python 3.12 runtime and headless Chrome integration.
- Explicit hard/soft reset cases and checkpoint/runtime/code hashes on every run.
- A shared transactional budget that counts training, validation, and replay calls.
- Maintained Stable-Baselines3 PPO with verified import of the historical weights.
- Per-step trajectory recording and MP4 export with simulation-derived timestamps.
- A complete pilot command for future independent replications.

**Verification and costs**

The two pilot videos matched all 2,361 recorded transitions: all 60 normalized
observation values, distances, and termination flags. Reset introduces a small native
clock offset, so elapsed-time verification subtracts that one offset and allows
5e-6 raw-time units for float32 rounding. No body-state tolerance was used.

All 11 tests passed, including separate-process budget accounting, budget exhaustion
before an action, abandoned reservation behavior, checkpoint conversion, reset history,
replay divergence detection, and clock-drift rejection. Lint, formatting, dependency
checks, real training, evaluation, and video export passed.

- Pilot training: 262,144 environment steps total.
- Pilot total including evaluation and replay: 278,740 steps.
- Historical diagnostics and verified replays: 15,801 steps.
- All environment calls in this session: 294,541 steps.
- Training throughput: approximately 820–830 steps/second in these runs.
- Paid model API usage: none.

The first pilot was executed as individual CLI stages while the runner was being built.
Its cap was 350,000 steps, with a separate 50,000-step diagnostic cap. The packaged
one-command runner reserves a worst-case cap of 332,144 steps at these defaults.
The relevant training/environment implementation hashes match across both arms.

**Limits on the result**

There is only one training seed. Six validation case labels collapse to one initial
observation and two physical trajectories per model. These are development cases,
not independent robustness trials or a held-out test. This result does not establish
that a higher time penalty is generally worse, that AI research beats parameter search,
or that a visually convincing gait has been achieved.

Historical scores changed under the current runtime: the archived 158.75/130.40 s
became 167.74/133.25 s. Their original complete runtime was not recorded, so the cause
has not been isolated. The new experiment uses its own measured scores consistently.

**Artifacts**

- [Side-by-side pilot video](C:/Users/zachd/Documents/ChatGPT/QWOP/runs/pilot-001/comparison.mp4)
- [Historical original versus speed model](C:/Users/zachd/Documents/ChatGPT/QWOP/runs/diagnostics/history-comparison.mp4)
- [Structured result](C:/Users/zachd/Documents/ChatGPT/QWOP/results/pilot-001/result.json)
- [Checkpoints, raw traces, manifests, and exact source snapshot](C:/Users/zachd/Documents/ChatGPT/QWOP/results/pilot-001/artifacts.tar.gz)
- [Archive file hashes](C:/Users/zachd/Documents/ChatGPT/QWOP/results/pilot-001/artifact-index.json)

The videos use the same predetermined seed-101/soft-0 case for both arms; they were
not selected from different best-case episodes. The faster controller holds its final
frame while the slower controller finishes. The archive contains no proprietary game
source, browser binary, API credentials, or video; bootstrap acquires locked references.

**Recommended next decision**

Keep the time-penalty change unpromoted. Repeat the controlled comparison across more
training seeds before drawing a general conclusion. The next algorithm comparison can
then introduce QR-DQN under a separately specified budget. Improving gait quality and
testing an AI-guided research loop remain explicit later milestones.
