# Architecture research: first bounded campaign

The first question is whether a researcher-designed policy architecture is useful in QWOP.
This is an initial screen, not a claim that the researcher improves RL in general or invents
an architecture. All training begins from fresh weights, using the same PPO implementation,
reward, 60 observations, 16 actions, and four-frame action interval.

| Approach | Trials | Training interactions |
| --- | --- | ---: |
| Fixed baseline | 128 x 128 Tanh MLP | 1,048,576 |
| Predefined search | 64 x 64 and 256 x 256 Tanh MLPs | 524,288 each |
| AI-directed design | Two proposals; second written after observing the first | 524,288 each |

Each approach has another 75,000 interactions reserved for evaluation and verified replay.
The maximum campaign expenditure is 3,370,728 environment steps. Parameter count is capped
at 250,000 and training wall time at two hours per approach (one hour per search trial).
This is equal training data and equal caps, not equal realized compute or total project cost.
CPU time, model size and inference/training overhead should be reported alongside performance.

The initial custom design shares a small encoder between the left and right leg. It makes
torso-relative limb positions explicit and combines the leg encodings with a full-observation
branch. The hypothesis is that reusable leg features improve coordination and learning speed.
The raw branch preserves the original observation information. This combines familiar ideas;
it is not presented as a new scientific architecture.

The ordinary search is fully specified before campaign training. The second research proposal
must cite the hash of the completed first result and explain its hypothesis and expected signal.
Codex is the researcher in this task. The Python runner executes proposals and records results;
it contains no autonomous API-backed researcher, makes no paid model calls, and cannot
independently attest the exact Codex model snapshot. Prompts/proposals and evidence are saved.

## Measurement and selection

Save trained checkpoints after each 262,144 steps, following PPO gradient updates. Evaluate
each on the same two explicit reset phases. Only each trial's final checkpoint is eligible
for selection. Select the fastest valid observed 100m finish, break ties by completion rate,
and use mean terminal distance only when none of the candidates finish. Report reliability
separately. The finish requires both a recorded 100m crossing and native terminal success.

The benchmark stores raw info.time as game-scale seconds. Older reports used info.time * 10
at native termination, which includes distance traveled after crossing 100m. Those old
numbers are retained and must not be compared numerically without rescoring. Videos run on
the simulation clock and label their overlay as game time.

The six final cases repeat two reset phases across three seed labels. They are descriptive
development cases, not six independent physical conditions or a held-out robustness suite.
Training uses one seed per trial. Inspect verified videos for gait: a faster finish alone
does not prove upright running. A full scientific comparison requires repeated campaigns,
fresh training seeds, a specified robustness suite, and a stronger ordinary search baseline.

The current campaign isolates architecture within PPO. QR-DQN and longer 10–20M PPO training
remain follow-up baselines; the current million-step allocation cannot settle their potential.

## Running the campaign

```powershell
.venv/Scripts/python.exe -m qwop_lab.cli campaign init --out runs/architecture-001
.venv/Scripts/python.exe -m qwop_lab.cli campaign trial --out runs/architecture-001 --proposal runs/architecture-001/proposals/baseline-fixed.json
.venv/Scripts/python.exe -m qwop_lab.cli campaign trial --out runs/architecture-001 --proposal runs/architecture-001/proposals/search-small.json
.venv/Scripts/python.exe -m qwop_lab.cli campaign trial --out runs/architecture-001 --proposal runs/architecture-001/proposals/search-large.json
.venv/Scripts/python.exe -m qwop_lab.cli campaign trial --out runs/architecture-001 --proposal runs/architecture-001/proposals/research-1.json
# Write research-2.json after reading the completed research-1 result.
.venv/Scripts/python.exe -m qwop_lab.cli campaign trial --out runs/architecture-001 --proposal runs/architecture-001/proposals/research-2.json
.venv/Scripts/python.exe -m qwop_lab.cli campaign status --out runs/architecture-001
.venv/Scripts/python.exe -m qwop_lab.cli campaign finish --out runs/architecture-001
```

Separate arms may run concurrently in independent processes. Run a search arm's trials in
the listed order. Existing trial directories cannot be reused. Interrupted trials remain
charged and are marked failed; an interrupted campaign does not acquire additional budget.
The protocol and protected trainer/evaluator hashes are frozen before training. Candidate
modules have versioned filenames, source copies, and hashes checked when weights are loaded.

Proposal format:

```json
{
  "arm": "research",
  "slot": "proposal-1",
  "architecture": {
    "layers": [64, 64],
    "activation": "Tanh",
    "extractor": "qwop_lab.candidates.body_grouped_v1:BodyGroupedExtractor"
  },
  "hypothesis": "Sharing leg features helps coordinate the gait.",
  "expected_signal": "Better completion or 100m time at the same trial budget.",
  "evidence": []
}
```

The second proposal's evidence must include the exact SHA-256 of
`research/proposal-1/result.json`, using that path relative to the campaign directory.

## Prior engineering work

The custom architecture was exercised before protocol freeze with 8,192 training steps,
30,000 evaluation steps and up to 5,000 replay steps. That smoke test checks serialization,
browser operation and trace replay. Its weights are discarded; its expense and observed
behavior are disclosed separately from the campaign. This is not a preregistered blind study.

## Recorded amendment for architecture-001

The first custom trial lost a browser response at charged step 46,253. Its result and
training manifest remain immutable. `runs/architecture-001/amendment-001.json` authorizes
one fresh retry of the identical architecture using 477,184 of that slot's remaining
steps. The remaining 851 interactions cannot fill a PPO rollout and are left unused.
No ledger cap increases. The retry is evaluated at its final checkpoint. The second
proposal must cite both the original failure and the retry result. This amendment is
recorded before retry training and makes the experiment an amended exploratory campaign.

The explicitly recorded recovery path is:

```powershell
.venv/Scripts/python.exe -m qwop_lab.campaign_recovery amend --out runs/architecture-001
.venv/Scripts/python.exe -m qwop_lab.campaign_recovery retry --out runs/architecture-001
.venv/Scripts/python.exe -m qwop_lab.campaign_recovery adaptive --out runs/architecture-001 --proposal runs/architecture-001/proposals/research-2.json
.venv/Scripts/python.exe -m qwop_lab.campaign_recovery finish --out runs/architecture-001
```

The exact transport failure cause is unknown. A repeat failure stops the amended campaign
for diagnosis; it does not grant additional training attempts or reset the budget.

## Fresh-seed architecture replication

After the exploratory campaign, freeze the baseline 128x128 Tanh policy and selected
kinematic 64x64 policy. Train both from fresh weights using three new seeds (17, 29,
61). Every run receives exactly 1,048,576 interactions, for 6,291,456 training steps
overall. PPO settings and reward remain those of the architecture campaign. Save and
evaluate checkpoints every 262,144 interactions. All final checkpoints are reported;
there is no selection across seeds or intermediate checkpoints.

The evaluation suite uses the two known reset phases under reset seed 101. Extra
seed labels mostly repeat these physical starts, so the analysis treats training
seeds as the independent units. Report every seed's completion count and mean 100m
game time when both phases finish. Compute paired speed changes only when both
designs finish both phases. Failed phases remain visible in reliability counts.
Three training seeds give descriptive replication evidence, not a strong significance
claim, architectural novelty claim, or a replicated AI-research advantage.

Each run has a separate 50,000-step evaluation/replay cap and a two-hour training
wall cap. Up to three workers run concurrently. Final phase-zero trajectories are
fully verified on replay. Gait is assessed from predetermined replay frames and
videos; speed alone is insufficient evidence for upright running.

```powershell
.venv/Scripts/python.exe -m qwop_lab.replication init --out runs/my-replication
.venv/Scripts/python.exe -m qwop_lab.replication batch --out runs/my-replication
.venv/Scripts/python.exe -m qwop_lab.replication status --out runs/my-replication
.venv/Scripts/python.exe -m qwop_lab.replication finish --out runs/my-replication
```

The first launch (`runs/replication-001`) failed to connect all six environments
before any game steps. The replacement (`runs/replication-002`) hash-links the
original failures and verifies all their ledgers spent zero steps. Clearing stale
headless browser sessions restored startup. Browser lifecycle and registration timeout
handling changed before the replacement freeze; experimental settings and game
physics remained the same. Separate startup diagnostics are accounted outside the
six training allocations. A spent training failure cannot use this zero-step recovery
path or automatically acquire a new budget.

During replication-002, baseline seed 29 was interrupted at 237,568 interactions
when Windows denied replacement of its progress JSON file. Its saved PPO model
had completed 690 optimizer epochs (115 rollouts); the final 2,048 collected
interactions had not been used in a gradient update. The failed manifest, result,
weights, and ledger remain immutable.

The explicit `resume-amendment.json` permits one continuation from those weights
and optimizer for the remaining 811,008 interactions, with the game and action
RNG reset. Those discarded rollout interactions remain charged, so the final
interaction allowance is still exactly 1,048,576. This is not exact mid-episode
resume. A bounded JSON-write retry is recorded in the continuation implementation.
Live monitoring requested Windows delete sharing to reduce interference. A post-study
regression check showed sharing flags alone did not prevent replacement failures on
this filesystem; the maintained writer now retries transient permission errors.

The final analysis must identify the interrupted run and separately show the two
uninterrupted seed pairs (17 and 61). No continuation weights or result are allowed
to replace the original failure evidence. The recorded continuation runs only when
one of the three training slots is available.

```powershell
.venv/Scripts/python.exe -m qwop_lab.replication_resume amend --out runs/replication-002
.venv/Scripts/python.exe -m qwop_lab.replication_resume resume --out runs/replication-002
.venv/Scripts/python.exe -m qwop_lab.replication_resume finish --out runs/replication-002
```

## Implementation references

- [Stable-Baselines3 custom policy and feature extractors](https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html)
- [Upstream QWOP PPO training template and recommended duration](https://github.com/smanolloff/qwop-gym/blob/main/qwop_gym/tools/templates/train_ppo.yml)
- [Neural architecture search for RL agents](https://arxiv.org/abs/2011.14632)

Codex token usage and subscription costs are not instrumented by this local campaign.
Do not interpret the equal interaction allowances as an equal-dollar research comparison.
