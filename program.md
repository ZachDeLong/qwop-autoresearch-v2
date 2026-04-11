# QWOP Autoresearch v3 — Speed Run

## Goal

Finish the QWOP 100m race as fast as possible. The metric is **game-engine time** — how many in-game seconds the athlete takes to cross the finish line. Lower is better.

**Constraint**: The agent must finish 100% of evaluation episodes. A fast agent that sometimes falls is worse than a slower agent that always finishes. Optimize speed, but never sacrifice reliability.

**Baseline to beat**: Standard CleanRL PPO with default rewards achieves **158.75s mean finish time** (100% success rate on 100 episodes). This is the "standard RL" result — no speed-specific tuning, just good PPO with the environment's default reward.

**Context**: The QWOP human speedrun record is ~48-50 seconds. There is a massive gap between 158.75s and human performance, which means fundamentally faster gaits exist.

## What "Fast" Means

The finish time is **QWOP game-engine time**, not wall-clock time or step count.

- `frames_per_step=4` is frozen — you cannot change how many physics frames each action covers
- The eval harness reads `info["time"] * 10` as the canonical game time
- A faster agent runs the 100m with a faster *gait* — longer strides, better coordination, less wasted motion
- You cannot game this by reducing steps or frames — the game physics are fixed

## The Rules

### Frozen (do NOT modify)
- `eval_harness.py` — the evaluation script
- `replay_renderer.py` — the video replay script
- `step_counter.py` — the budget enforcement wrapper
- `baseline/` — the baseline agent and results
- `program.md` — this file
- The qwop-gym package itself
- `frames_per_step=4` — this is part of the physics contract

### Your workspace
Everything in `claude/` is yours. Create whatever file structure you need.

### Budget
- **10,000,000 env steps** total (calls to `env.step()`)
- **10,000 free sanity-check steps** before the budget starts
- You MUST use `make_counted_env()` from `step_counter.py` for all training/search
- Call `activate_budget()` when you're done with sanity checks and ready to train
- Call `print_budget_status()` between runs to track usage
- If you exhaust the budget, go to evaluation with what you have

### Deliverable
You must produce `claude/agent.py` with one of these interfaces:

```python
# Option 1: Policy-based
def get_action(obs: np.ndarray) -> int:
    """Given a 60-float observation, return an action in [0, 15]."""

# Option 2: Sequence-based
def get_action_sequence() -> list[int]:
    """Return the full action sequence to replay."""
```

The eval harness will call this to evaluate your agent on 100 episodes.

## Environment Reference

- **Observation**: 60 floats in [-1, 1] — 12 body parts x 5 values (pos_x, pos_y, angle, vel_x, vel_y)
- **Action**: Discrete(16) — all 2^4 combinations of Q, W, O, P keys
- **Reward**: `velocity * speed_rew_mult - dt * time_cost_mult / frames_per_step + terminal_bonus`
  - Default: `speed_rew_mult=0.01`, `time_cost_mult=10`, `success_reward=50`, `failure_cost=10`
  - You can pass different values to `make_counted_env()` or write custom reward wrappers
- **Termination**: Fall (failure) or 100m finish (success)
- **Deterministic**: Same actions from same reset produce same outcome. You can exploit this.
- **Quirk**: The environment alternates between two slightly different initial states on consecutive resets. A fixed action sequence only handles one state. A policy-based agent (observing current state) handles both.
- **Max episode steps**: 5000 (enforced by TimeLimit wrapper)
- **frames_per_step**: 4 (each step advances 4 physics frames — FROZEN)

## Phases

### Phase 0: Orientation (free — no budget)
1. Read this file
2. Read `step_counter.py` and `eval_harness.py` to understand the interfaces
3. Explore qwop-gym source code to understand the reward function and env mechanics
4. **Plan your approach and explain your reasoning**
5. Install any packages you need
6. Write your code and verify it works with sanity-check steps (up to 10,000 free)

### Phase 1: Training / Search (budget starts)
1. Call `activate_budget()` to start the clock
2. Run your approach
3. **Between each sub-run, narrate**: what you tried, results, what's next
4. Track budget with `print_budget_status()`
5. Commit meaningful checkpoints to git
6. Pivot if something isn't working — don't waste budget on diminishing returns

### Phase 2: Evaluation (free — no budget)
1. Ensure `claude/agent.py` implements the correct interface
2. Run: `python eval_harness.py --agent claude/agent.py --out results/eval_claude.json --label claude`
3. Report the results

## Narration Format

Between each sub-run, write a log entry in `claude/experiment_log.md`:

```markdown
### Run N: <approach name> (<steps used>)
**Hypothesis**: <what you're trying and why>
**Result**: <best finish time, success rate, key observations>
**Budget remaining**: <steps left>
**Next move**: <what you'll do next and why>
```

## V2 Lessons (things your predecessor learned)

### What worked
- 128x128 Tanh network with orthogonal init was the best architecture found
- Deterministic policy (argmax of logits) gives 100% success rate — eliminates sampling variance
- Mass stochastic rollout collection finds better gaits than sequence mutation
- The environment's default reward (velocity-based) teaches basic locomotion

### What didn't work
- Random hill climbing on action sequences — QWOP gait is chaotic, mutations cascade through physics
- Crossover between sequences — different gaits break at splice points
- Single-action refinement — same chaos issue
- Fixed action sequences — env alternates initial states, sequences only work 50% of the time
- Reward shaping with velocity bonuses — hurt distance in v1 (but speed-focused shaping was not thoroughly explored)
- Observation normalization (RunningMeanStd) — faster throughput but lower performance
- Deeper networks (256x3) — slower and worse
- ReLU — significantly worse than Tanh for this task

### Speed-specific findings
- Increasing `time_cost_mult` from 10 to 30 helped (penalizes slow running more)
- Increasing `success_reward` from 50 to 100 helped (stronger incentive to finish)
- Terminal speed bonuses (reward for fast finish) had modest effect — the signal is sparse (once per episode) and must propagate through ~1000 steps of GAE
- Per-step speed signal may be more effective than terminal bonus (not tested)
- Lowering gamma (making the agent more impatient) was not tested
- The standard PPO gait finishes at ~158.75s. Speed-tuned PPO reached ~130.4s. Human record is ~48-50s.

## Tips

- You're not limited to PPO or even RL. Consider: evolutionary strategies, CMA-ES, direct action sequence search, population-based methods, or hybrids.
- The deterministic property is powerful — if you find a good action prefix, you can search for extensions.
- 10M steps is enough for several full approaches. Don't blow it all on one thing.
- The default env reward has a time penalty — it already discourages slow running, but not aggressively.
- Think about what makes QWOP gaits fast vs slow. Human speedrunners use very specific, efficient stride patterns.
- A per-step speed signal (rewarding high velocity at every step) may train better than a terminal bonus (rewarding fast finish at the end).
- Lower gamma values make the agent prioritize near-term rewards — potentially useful for speed.
- The gap between 130s and 50s suggests there are qualitatively different, faster gaits to discover.
