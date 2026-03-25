# QWOP Autoresearch v2 — Claude Code Directives

## Goal

Achieve the highest possible QWOP distance by designing and implementing your own approach from scratch. You have full creative freedom — choose any algorithm, framework, or strategy you want.

**Baseline to beat**: Vanilla CleanRL PPO achieved 100.2m (v1 baseline). Your v1 predecessor, constrained to only tuning PPO hyperparams, reached 105.1m.

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

- **Observation**: 60 floats in [-1, 1] — 12 body parts × 5 values (pos_x, pos_y, angle, vel_x, vel_y)
- **Action**: Discrete(16) — all 2^4 combinations of Q, W, O, P keys
- **Reward**: Velocity-based with time penalty. See qwop-gym source for formula.
- **Termination**: Fall or 100m finish
- **Deterministic**: Same actions from same reset → same outcome. You can exploit this.
- **Max episode steps**: 5000 (enforced by TimeLimit wrapper)
- **frames_per_step**: 4 (each step advances 4 physics frames)

## Phases

### Phase 0: Orientation (free — no budget)
1. Read this file
2. Read `step_counter.py` and `eval_harness.py` to understand the interfaces
3. Read the v1 results: `../qwop-autoresearch/results.tsv` and `../qwop-autoresearch/train.py`
4. Explore qwop-gym source code to understand the reward function and env mechanics
5. **Plan your approach and explain your reasoning** — this is content for the video
6. Install any packages you need
7. Write your code and verify it works with sanity-check steps (up to 10,000 free)

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
**Result**: <best distance, success rate, key observations>
**Budget remaining**: <steps left>
**Next move**: <what you'll do next and why>
```

## V1 Lessons (things your predecessor learned)

- Reward shaping (velocity bonuses) consistently hurt performance — raw reward was better
- Observation normalization (RunningMeanStd) improved throughput but reduced final distance
- Deeper networks (256×3) were slower and worse than 128×128
- ReLU was significantly worse than Tanh for this task
- The winning v1 config was: 128×128 Tanh network, NUM_STEPS=512, ENT_COEF=0.02
- After that first good config, ~8 more experiments all tied or regressed
- The environment is deterministic — replay-based approaches are viable but unexplored

## Tips

- You're not limited to PPO or even RL. Consider: evolutionary strategies, CMA-ES, direct action sequence search, population-based methods, or hybrids.
- The deterministic property is powerful — if you find a good action prefix, you can search for extensions.
- 10M steps is enough for several full approaches. Don't blow it all on one thing.
- The v1 agent occasionally fails to finish despite averaging 100+m. Consistency matters for eval.
- Think about what's unique here vs a standard RL benchmark. QWOP has unusual physics — locomotion emerges from 4 binary key combinations.
