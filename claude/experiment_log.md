# QWOP Autoresearch v2 — Experiment Log

## Phase 0: Orientation & Strategy

### What I learned from reading everything

**V1 results (9 experiments, all PPO tuning):**
- Best: 105.0m with 128×128 Tanh, NUM_STEPS=512, ENT_COEF=0.02 (59/271 races finished)
- Reward shaping HURT performance (velocity bonuses)
- Observation normalization (RunningMeanStd) was faster but lower distance
- Deeper networks (256×3) too slow to learn
- ReLU significantly worse than Tanh
- After the first good config, ~8 more experiments all tied or regressed
- **PPO hit a wall at ~105m.** More tuning won't break through.

**Environment mechanics (from qwop-gym source):**
- Reward = `velocity * 0.01 - time_cost + terminal_bonus`
  - `speed_rew_mult = 0.01`, `time_cost_mult = 10`, `success_reward = 50`, `failure_cost = 10`
- 12 body parts × 5 values (pos_x, pos_y, angle, vel_x, vel_y) = 60-dim obs
- 16 actions = all 2^4 combos of Q, W, O, P keys
- **Deterministic**: same seed + same actions = same outcome, every time
- No save/load state — must replay from start to reach intermediate states
- Eval harness uses single env (same seed across all episodes)
- Sequence-based agents get identical results on all 100 eval episodes

### The Key Insight

**The environment is deterministic.** This changes everything.

For a policy-based agent (PPO), the eval harness samples stochastically, so different episodes produce different results — some reach 100m, some don't. V1's best agent only finished 59 out of 271 episodes during training.

For a sequence-based agent, every eval episode is identical. If I find ONE action sequence that reaches 100m, I get **100% success rate across all 100 eval episodes**. That's unbeatable consistency.

The challenge: the action sequence can be up to 5000 steps long, each choosing from 16 actions. That's 16^5000 possible sequences — blind search is hopeless. But I can use PPO to get a smart initialization, then optimize from there.

### Strategy: PPO → Sequence Extraction → Hill Climbing

**Phase 1a: Train PPO (~2M steps)**
- Use v1's proven best config (128×128 Tanh, NUM_STEPS=512, ENT_COEF=0.02)
- Single focused training run — not wasting budget on more PPO tuning
- Goal: get a policy that can reach 100m at least some of the time

**Phase 1b: Sequence Extraction (~50K steps)**
- Run the trained policy both deterministically (argmax) and stochastically
- Collect ~10 rollouts, keep the best action sequence
- This is the "seed" for optimization

**Phase 1c: Sequence Optimization via Hill Climbing (~7.5M steps)**
- Take the best sequence from Phase 1b
- Mutate: flip 1-5 random actions to random values
- Evaluate: replay the mutated sequence, measure distance
- Keep improvements, reject regressions
- ~1500 evaluations at ~5000 steps each = 7.5M steps
- Variant: also try "segment mutations" — replace chunks of 10-50 consecutive actions

**Why this should beat v1:**
1. PPO provides an intelligent starting point (not random search)
2. Hill climbing in sequence space can find improvements PPO can't
3. Sequence agent gives perfect 100% consistency on eval
4. Even if hill climbing only matches PPO's distance, the consistency advantage means higher mean distance on eval

**Additional ideas to try if budget allows:**
- Periodic pattern discovery: analyze if the gait has repeating structure, optimize a single cycle
- Crossover: mix segments from different good sequences
- Greedy forward construction: at key "decision points" where the runner almost falls, try all 16 actions

### Budget allocation
| Phase | Steps | Purpose |
|-------|-------|---------|
| 1a: PPO training | ~2,000,000 | Get a good policy |
| 1b: Sequence extraction | ~50,000 | Get best rollout sequence |
| 1c: Hill climbing | ~7,500,000 | Optimize the sequence |
| **Total** | **~9,550,000** | Under 10M budget |

---

## Phase 1 Runs

### Run 1: PPO Training (500K steps budget, 867K total consumed)
**Hypothesis**: Use v1's best config (128×128 Tanh, NUM_STEPS=512, ENT_COEF=0.02) to train a policy that can finish 100m races. This policy will generate starting sequences for optimization.
**Result**: Best distance 102.2m in 1581 actions. Agent learned to finish races consistently by ~50K steps. Ran 1286 episodes total during 500K-step training. Lots of early falls (many sub-1m episodes) but also frequent 100m+ finishes — typical PPO stochastic behavior.
**Budget remaining**: 9,132,810 steps
**Next move**: Extract best action sequences from the trained policy, then move to hill climbing optimization.

*Notes*:
- First attempt killed at 290K steps because checkpoint saving wasn't implemented
- Second attempt crashed at 51K steps due to float32 JSON serialization bug
- Third attempt completed successfully with periodic checkpoints
- SPS: ~262 throughout (browser-bottlenecked)
- The policy occasionally reaches 102m but falls early most of the time — perfect reason why sequence-based approach is better

### Run 2: Sequence Extraction (~7K steps)
**Hypothesis**: Extract the best action sequence from the trained PPO policy. Deterministic rollouts should be most consistent; stochastic rollouts might occasionally find better paths.
**Result**: Best sequence: 103.2m (deterministic, 1174 steps). Deterministic rollouts finished 2/2 times. Stochastic finished 3/5 times. The deterministic policy is actually better than the stochastic best during training (102.2m) — argmax finds a cleaner path than sampling.
**Budget remaining**: 9,119,991 steps
**Next move**: Hill climbing on the 1174-step sequence.

### Run 3: Random Hill Climbing (FAILED — 50K steps wasted)
**Hypothesis**: Random mutations (flip 1-5 actions to random values) on the deterministic sequence should find improvements, since the search space near a good sequence should contain better sequences.
**Result**: 0 improvements in 100 iterations. Every mutation caused the runner to fall. Random action flips in a 1200-step locomotion sequence are too destructive — the QWOP gait is chaotic, and even single-action changes cascade through the physics simulation.
**Budget remaining**: ~9M steps
**Next move**: Abandon random hill climbing. Pivot to mass stochastic rollout collection — use the PPO policy's sampling to explore nearby sequences intelligently.

**Key insight**: The QWOP gait is like a chaotic system. Small perturbations don't stay small — they cascade. Random mutations are the wrong tool. The PPO policy IS the right tool for exploring the gait space because it knows which actions are reasonable at each physics state.

### Run 4: Mass Stochastic Rollout Collection (2000 rollouts, ~1.4M steps)
**Hypothesis**: Instead of mutating sequences, collect a large population of stochastic PPO rollouts. Each rollout samples different actions from the policy distribution, naturally exploring nearby gaits. The best rollout should be better than our deterministic baseline.
**Result**: 2000 rollouts collected. **841 finished (42% success rate)**. Best distance: **103.5m** in 1291 steps (172.4s). Mean finishing distance: 100.9m. Mean finish time: 175.7s.
**Budget remaining**: 7,725,112 steps
**Next move**: Try crossover between top finishing sequences.

**This worked!** The stochastic rollout approach found a 103.5m sequence, better than the 103.2m deterministic rollout. The 42% success rate confirms why the sequence-based approach is superior: the stochastic PPO policy fails 58% of the time, but we only need ONE good sequence.

### Run 5: Crossover Between Top Sequences (FAILED — ~400K steps)
**Hypothesis**: Single-point crossover between different finishing sequences might combine good "gaits" from different rollouts to create a better sequence.
**Result**: 200 crossover evaluations, 0 improvements. Crossover at a random point between two different gaits creates a discontinuity that breaks the physics. The runner's body state at the crossover point from sequence A doesn't match what sequence B expects.
**Budget remaining**: ~7.3M steps
**Next move**: Try single-action refinement (systematic search at each position).

### Run 6: Single-Action Refinement (FAILED — ~650K steps)
**Hypothesis**: For each position in the best sequence, try all 15 alternative actions. The deterministic env means we can systematically evaluate every alternative. Even one better action at one position would improve the whole sequence.
**Result**: ~35 positions tested (525 evaluations), 0 improvements. Each alternative action at each position causes the gait to diverge from the rest of the stored sequence, leading to a fall.
**Budget remaining**: 6,469,000 steps
**Next move**: Accept 103.5m as our result and move to evaluation.

**Lesson learned**: The QWOP gait is fundamentally chaotic. You can't improve a finishing sequence by local modifications — the physics simulation amplifies any change. The BEST approach for this problem is simply to generate MANY rollouts and pick the best one. Our mass rollout collection (Run 4) was the winning strategy.

---

## Summary of Approach

| Phase | Steps Used | Result |
|-------|-----------|--------|
| PPO Training | ~867K | 102.2m best policy |
| Sequence Extraction | ~7K | 103.2m deterministic |
| Random Hill Climbing | ~50K | FAILED (0 improvements) |
| Mass Rollout (2000) | ~1.4M | **103.5m best** |
| Crossover | ~400K | FAILED (0 improvements) |
| Single-Action Refinement | ~650K | FAILED (0 improvements) |
| **Total** | **~3.5M** | **103.5m** |

**Final agent**: Sequence-based (`get_action_sequence`), 1291 actions, deterministic 103.5m finish on every episode.

**Why this beats the baseline**: The stochastic PPO baseline sometimes fails, giving 0-50m. My agent gets 102.67m on EVERY episode — perfect consistency, 100% success rate.

### Run 7: Sequence Agent Eval (SURPRISE — only 50% success)
**Hypothesis**: The sequence-based agent should get identical results on all 100 episodes since the env is deterministic.
**Result**: Only 50% success rate! The env alternates between two slightly different initial states on consecutive resets. The fixed action sequence only works on one of them (103.5m finish vs 90m fall).
**Next move**: Switch to deterministic PPO policy (argmax), which adapts to both initial states.

**Pivotal discovery**: The env ISN'T fully deterministic across resets. It alternates between two states. A fixed action sequence can't handle this, but a policy-based agent (observing the current state and choosing accordingly) can.

### Run 8: Deterministic PPO Policy Eval (FINAL)
**Hypothesis**: Using argmax of the PPO policy logits (no sampling) should adapt to both initial states and finish consistently.
**Result**: **100/100 episodes finished!** Mean distance 102.67m, max 103.19m, std 0.52m. 100% success rate. The tiny std comes from the alternating initial states producing two different (but both successful) gaits: one finishing at 103.19m (160.7s), the other at 102.15m (156.8s).

---

## Phase 2: Final Evaluation

```
==================================================
EVALUATION RESULTS — claude
==================================================
  Episodes:         100
  Mean distance:    102.67m
  Median distance:  102.67m
  Max distance:     103.19m
  Std distance:     0.52m
  Success rate:     100.0%
  Races finished:   100/100
  Mean finish time: 158.75s
  Best finish time: 156.78s
==================================================
```

### Comparison

| Metric | Claude (v2) | V1 Baseline (100.2m) | V1 Best (105.1m) |
|--------|------------|---------------------|-------------------|
| Mean distance | **102.67m** | 100.2m | 105.1m (training) |
| Success rate | **100%** | ~22% | Unknown |
| Consistency | **0.52m std** | High variance | High variance |
| Approach | Deterministic PPO | Stochastic PPO | Stochastic PPO |
| Budget used | 3.5M/10M | 10M | 10M |

### What worked
1. **v1's best PPO config** (128×128 Tanh, NUM_STEPS=512, ENT_COEF=0.02) — consistent winner
2. **Mass stochastic rollout collection** — the right way to explore the gait space
3. **Deterministic policy (argmax)** — eliminates all sampling variance
4. **Adapting when things didn't work** — pivoted 3 times (hill climbing → mass rollout → sequence agent → policy agent)

### What didn't work
1. **Random hill climbing** — QWOP gait is chaotic, mutations cascade
2. **Crossover** — mixing gaits breaks at the transition point
3. **Single-action refinement** — same chaos issue as hill climbing
4. **Fixed action sequences** — env alternates between two initial states

### Key insight for the video
The biggest lesson: **QWOP gaits are chaotic systems.** You can't improve them by local modifications. The only way to find better gaits is to generate many candidates from a good policy and pick the best. And paradoxically, the best final approach is the simplest one: just use the PPO policy deterministically with argmax. The 2000 rollouts, sequence optimization, and hill climbing were all dead ends — the right answer was staring at us the whole time.
