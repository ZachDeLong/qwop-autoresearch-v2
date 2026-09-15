Review of qwop-autoresearch-v2 — September 15, 2026

**Recommendation: retry with a fresh experiment framework, while retaining the existing game integration and historical agents as reference material.** The saved results show useful learning and a meaningful speed improvement. They do not establish a clean comparison between ordinary RL and an AI research agent, and they do not establish that the available methods have reached their performance limit.

The user wants both a fast, convincing runner and a credible AI research experiment. Those should be two explicit outcomes with separate measurements.

**Scope and verification**

Reviewed current master at `2e6783f9f9a93eed1fd28a00a4a47a9cd760bbec`, the historical experiment commits, and `v2-final` at `d5201f8c4f989079b3195cfba96626310118c820`. Inspected the pinned dependency's `v1.0.1` source at `da087ce63075ad910ee571a669999993b794fc77`, upstream documentation, and upstream training configurations.

Recomputed statistics from five saved evaluation files, checked action ranges and sequence lengths, hashed action sequences to identify duplicates, compared baseline provenance, parsed all 15 Python files across the two project checkouts, and exercised the current step counter using independent module states and stub environments. Detailed results are in [review_evidence.json](C:/Users/zachd/Documents/ChatGPT/QWOP/review_evidence.json).

These are source and artifact checks. I did not execute the game, visually assess a trained gait, reproduce training, or independently verify the historical score in a live browser. The local Python 3.12 installation lacks the training dependencies. Existing source files and model artifacts were not changed.

**What the results actually show**

| Saved experiment | Mean finish time, using the repo's conversion | Finishes | Distinct action sequences |
|---|---:|---:|---:|
| Original v2 PPO | 158.75 s | 100/100 | 2, each repeated 50 times |
| Speed round 1 | 134.35 s | 100/100 | 2, each repeated 50 times |
| Speed round 2 / v2 final | 130.40 s | 100/100 | 2, each repeated 50 times |

The reduction is 28.35 seconds, or 17.86% less finish time. The final saved budget is 7,037,101 steps out of 10 million. The final policy is a 128-by-128 Tanh PPO network using argmax actions. Speed training warm-starts from an existing policy and changes rewards and optimization settings.

That is a useful prototype result. The supported claim is that this training sequence produced better recorded performance under this evaluation procedure. A repeatable advantage over a fairly tuned baseline remains untested.

**The comparison does not isolate the AI research contribution.**

The original [baseline adapter](C:/Users/zachd/Documents/ChatGPT/QWOP/qwop-autoresearch-v2/baseline/agent.py:44) samples from a categorical action distribution. The [final agent](C:/Users/zachd/Documents/ChatGPT/QWOP/qwop-v2-review/claude/agent.py:59) chooses argmax. Therefore the early reliability comparison changes inference behavior as well as training, architecture, and accumulated tuning. Deterministic inference should be available to both sides. Its benefit here is useful, but cannot by itself demonstrate superior research.

There is also a provenance mismatch: all 100 episode records in current `baseline/eval_standard_ppo.json` are exactly equal to the original Claude evaluation at commit `0716e10`. Only the summary's agent path and wall-clock duration differ. The new path is `claude/agent_policy.py`, which is not present in the reviewed trees. Reusing the previous winner as a speed-tuning baseline is reasonable; describing it as an independent vanilla CleanRL baseline obscures what was compared.

No matched campaign with equal initial conditions, total search/training budgets, repeated training seeds, and a non-AI tuning control is preserved. The final speed improvement also lacks an equal-budget control that simply continues training with the original reward. Consequently we cannot separate extra training from the individual reward and optimizer changes.

**The evaluation measures repetition much better than robustness.**

The [harness](C:/Users/zachd/Documents/ChatGPT/QWOP/qwop-autoresearch-v2/eval_harness.py:89) creates one environment and repeatedly calls `reset()` without specifying or recording seeds. Both original and final artifacts contain two exact action sequences repeated 50 times each. This supports consistency on the observed reset cycle. It does not amount to 100 independent challenges or establish reliability on unfamiliar states.

The training scripts seed Python, NumPy's legacy RNG, and Torch, but do not pass a seed to `make_counted_env()`. The pinned environment generates its seed using a separate `np.random.default_rng()`, so those script seeds do not fix the game's seed.

Upstream explicitly documents that default soft resets can change replay outcomes and that hard resets can reproduce them. The old log treats the alternating outcomes as a late surprise, even though reset semantics should have been validated before sequence search. See [upstream reset documentation](https://github.com/smanolloff/qwop-gym/blob/v1.0.1/doc/ENV.md#resetting).

The final harness is also reused during approach selection. A frozen file alone does not create an untouched test set. Separate development validation from final assessment. More seeds are useful only if checks establish that they actually generate distinct situations; record observation/trajectory hashes as well as seed labels. Use independently trained policies to measure training variability. If adding timing or state perturbations, report that as a separately specified robustness benchmark.

**The early objective rewarded the wrong progress.**

Early selection ranks finishers by terminal distance. The log describes 103.5 m as an improvement over a 103.2 m run even though the selected stochastic run took 172.4 seconds and deterministic runs were around 159 seconds. Once the runner finishes, distance overshoot provides a poor measure of speed or convincing locomotion.

The inference that PPO had reached a wall near 105 m is particularly weak: the pinned game wrapper forcibly ends an episode beyond 105 m. The old distance objective was approaching a termination boundary, not demonstrating a locomotion capability ceiling. The speed pivot was the correct direction.

Even current [best-replay selection](C:/Users/zachd/Documents/ChatGPT/QWOP/qwop-autoresearch-v2/eval_harness.py:184) still maximizes distance. In the final saved evaluation this selects the 130.65-second outcome instead of the faster 130.15-second outcome.

The upstream success signal is broader than simply crossing a mathematical 100 m line: it includes ending beyond 100 m and a fallback beyond 105 m. Decide whether the benchmark is official game completion or first passage across 100 m, and measure the chosen event consistently. See [termination behavior](https://github.com/smanolloff/qwop-gym/blob/v1.0.1/qwop_gym/envs/v1/game/extensions.js#L151).

**The research conclusions are stronger than the experiments support.**

Roughly 2.5 million steps in the log go to mass stochastic rollouts, crossover, and single-action refinement, yet the first final policy uses the original PPO network deterministically. Those searches supplied observations, but their best sequence was not the final controller.

Mutating an action while replaying the old suffix removes feedback: the suffix expects a body state that may no longer exist. The experiments also used soft resets, which confound comparisons between candidates and incumbents. Failure of these implementations does not establish that all local search fails, or that generating many rollouts is the only route to better gaits. Policy parameter search, state-conditioned continuation, and searches with verified reset conditions were not ruled out.

Similarly, changing several reward and optimizer settings together prevents attributing the speed gain to one cause. The log stops before the speed rounds and retains contradictory earlier claims. Treat statements such as “ReLU is worse,” “normalization hurts,” and “PPO has hit a wall” as observations from particular runs that need replication.

**The construction makes results difficult to reproduce and inspect.**

| Issue | Evidence and consequence |
|---|---|
| Current checkout omits the experiment | `claude/` and results exist in the historical tag, but master contains the scaffold and a baseline JSON. The old work is recoverable. |
| Incomplete baseline dependency | The baseline searches a sibling v1 checkout for weights and continues with random weights when missing. Evaluation should fail explicitly if the required checkpoint is absent. |
| Setup is underspecified | Broad dependency lower bounds, no recorded complete runtime, machine-specific browser/driver defaults, and no documented game-source patch/bootstrap procedure in this repo. Pin and identify the game source and patch as well as the Python package. |
| Training entry points depend on external state | They do not activate the budget themselves; standalone use requires an undocumented setup action or an already-active count file. |
| Trainer copies diverge | Three similar PPO trainers duplicate the algorithm. `train_speed2.py` prints different reward and optimizer settings from those it actually uses. |
| Checkpoint identity is ambiguous | Periodic saves capture the current model, while `best_time` and its action sequence may come from an earlier model. Round 2 writes `speed2_model.pt`, but the evaluator loads `speed_model.pt`; promotion requires an additional operation absent from the trainer. |
| Recovery is limited | Checkpoints contain model weights rather than optimizer state, RNG states, effective config, and budget/run identity. Budget exhaustion can interrupt training before a new checkpoint is saved. |
| Current counter cannot aggregate separate process caches | In a controlled check, two isolated module states each took 1,000 steps, but the shared file recorded 1,000 rather than 2,000. Hard interruption can also lose steps since the last checkpoint. This concerns a future parallel runner; it does not prove the historical serial count was wrong. |
| Replays lack necessary conditions | Saved actions omit seed/reset history. A fresh browser reset need not reproduce an episode selected from later in the reset cycle. |
| Renderer does not request drawing | `replay_renderer.py` calls neither `env.render()` nor `auto_draw=True`; pinned qwop-gym defaults to disabled automatic drawing. As written, stepping does not request updated game frames. |
| Video timing is arbitrary | Playback defaults to 10 action steps per second rather than deriving presentation timestamps from the game clock. Manual screen recording replaces an automatically produced, verified video artifact. |

The handwritten PPO also merges `terminated` and `truncated` before computing value targets. If the TimeLimit is an administrative cutoff, it needs appropriate timeout bootstrapping. If timeout is deliberately failure in a finite-horizon task, encode that task definition and available time explicitly. This is a correctness concern to resolve, not evidence that it caused the successful agents' speed plateau.

The `time * 10` conversion itself has source support: the pinned JavaScript exports `scoreTime / 10`. Retain raw time, converted score time, physics frames, and wall-clock time separately. Confirm the displayed timer and benchmark category before comparing against human records. There is no basis here to label every saved finish time a factor-of-ten error.

**What is worth keeping**

- Historical `ppo_model.pt`, `speed_model.pt`, evaluation JSON, action sequences, and commit lineage as regression references and warm-start candidates.
- The idea of a fixed action/physics contract and an evaluator independent of training rewards.
- The simple policy interface, expanded with explicit checkpoint configuration and per-episode reset support for policies with memory.
- The existing qwop-gym integration initially. Rewriting physics adds a major fidelity question before answering the learning question.
- The observation that speed tuning helped this particular policy, as a hypothesis to retest under controlled conditions.

**A concrete retry design**

Build a small pipeline with separately testable responsibilities: game adapter → trainer → validation/evaluation → artifact recorder. Put an experiment runner around it, with the research agent proposing changes and reading results. The runner should generate effective configurations, budget records, checkpoint identities, and metrics automatically.

Start with four milestones:

1. **Establish a trustworthy replay and benchmark.** Pin dependencies and game patch, specify timer and completion semantics, verify hard/soft resets, replay the old checkpoints where possible, and export correctly timed videos. Save raw per-step metrics and environment identity. Proceed when saved and replayed outcomes agree under explicitly identical conditions.
2. **Establish competent conventional baselines.** Use one maintained PPO implementation, plus QR-DQN as a serious alternative for this discrete-action environment. Give both deterministic evaluation and compare learning curves with measured compute costs. Start with a small pilot, then run several independent training seeds on viable configurations. Profile the browser, policy inference, updates, and logging before deciding whether a faster simulator is necessary.
3. **Test the research process.** Compare fixed baseline training, a prespecified non-AI parameter search, and the AI-guided search. Each campaign gets the same starting artifacts and environment-step budget across all trials and development evaluations. Track wall time, hardware, and AI cost separately. Repeat whole search campaigns if making a claim about the research agent's effectiveness, rather than only rerunning its final winner.
4. **Optimize the runner's behavior.** Select controllers on completion and speed; assess gait quality through representative videos and a defined rubric. Investigate dense progress/time rewards, curricula, memory or phase features, or demonstration-assisted initialization as measured hypotheses. If demonstrations or previous checkpoints are allowed, give controls equivalent access or report a separate track.

QR-DQN is worth a baseline because the environment author already reports fast-running results with it and provides training configurations. This is supporting evidence for feasibility, not a prediction that this repo will reproduce those scores under its 10-million-step budget. The upstream sweep specifies 32 million steps and a reduced action set, so comparisons require matching those conditions. See [the author's results](https://smanolloff.github.io/projects/qwop-gym/), [upstream sweep configuration](https://github.com/smanolloff/qwop-gym/blob/main/qwop_gym/tools/templates/wandb/qrdqn.yml), and [the maintained QR-DQN implementation](https://sb3-contrib.readthedocs.io/en/master/modules/qrdqn.html).

For development, show both completion and speed instead of rejecting every temporary reliability regression. Require a prespecified completion threshold for the final selected controller. “100% of this finite suite” is a valid target; it is not a guarantee of universal reliability. Plot success against speed to make the tradeoff visible.

For the final research assessment, use cases not used for candidate selection and training seeds not used to choose the recipe. Keep the native game benchmark separate from deliberately perturbed robustness cases. A single environment can support a claim about AI-assisted QWOP optimization, not a broad claim about AI research ability across tasks. These choices are consistent with [RL evaluation guidance](https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html#how-to-evaluate-an-rl-algorithm).

**Effort and stopping conditions**

The historical log reports roughly 262 environment steps per second. At that rate, 10 million steps takes about 10.6 hours; multiple training seeds and search campaigns multiply that cost. The counter's per-step disk I/O was improved after the original experiments, so benchmark current throughput before planning compute. Do not infer current performance from the old number.

Commit first to benchmark/replay validation and a measured baseline pilot. Expand the training budget when the setup is reproducible, learning curves and videos are informative, and one can name the next hypothesis. If the AI-guided search fails to beat ordinary search across repeated campaigns, that is a meaningful experimental result; the fast-runner effort can still succeed independently.

The strongest reason to retry is that the repository leaves substantial methodological and engineering questions unanswered. Preserve what it learned, replace the framework that made those lessons difficult to trust, and use the next run to answer specific questions.
