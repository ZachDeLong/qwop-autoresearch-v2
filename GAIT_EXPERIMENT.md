# Upright movement: first controlled reward screen

The previous six final policies finished the two development reset phases by
knee-scooting. The next question is whether a posture-dependent forward reward
changes that behavior at an equal, longer training budget.

Both arms start from identical fresh 128x128 Tanh actor/critic weights with seed
83. PPO settings remain `FRESH_PPO_CONFIG`. Each receives 2,097,152 interactions,
twice the previous replication's per-run allocation. No historical weights or
smoke-test weights are reused. Four fixed checkpoints per arm are evaluated;
only the final policies determine the experiment's result.

The baseline's reward is preserved exactly. The treatment multiplies the native
positive speed reward by `0.1 + 1.9 * posture_quality`. Negative speed, time cost,
failure cost, and native completion bonus are unchanged. There is no added reward
for standing still. This is an exploratory change to the objective, not
potential-based shaping with a claim that the original optimum is preserved.

The native engine calculates the first speed reward against a zero-valued previous
reaction, which includes the initial torso-distance offset. Both arms inherit this
convention, and the treatment also scales that first reset-related speed term.
The no-standing-bonus property applies to subsequent transitions with zero forward
displacement; there is no recurring reward simply for maintaining posture.

Posture quality is the product of three continuous scores clamped to [0, 1]:

- Pelvis height: zero at 0.30 m and full credit at 0.65 m.
- Minimum left/right knee clearance: zero at track height and full credit at 0.12 m.
- Torso alignment: full credit within 45 degrees of upright, zero at 90 degrees.

The separate binary upright metric requires all three full-credit thresholds.
These thresholds are exploratory and fixed before study training. An episode
that falls quickly can have a high upright fraction: report completion, distance,
and upright forward distance/fraction together. Forward distance sums positive
sampled displacements and can exceed net progress; it is not a race score.

Geometry uses unclipped raw physics positions and angles, with local hip, neck,
and knee anchors reconstructed from the locked game. The track's top surface is
its body-centre y (10.74275) minus half the 64-pixel texture height divided by the
world scale (20). Ten world units equal one displayed metre. A knee close to the
ground is a geometric proxy, not a measured contact or contact force. Upright
movement alone does not prove alternating footfalls, flight phases, or running.

Evaluation uses reset seed 101 with zero or one explicit soft resets and always
the native reward, for both arms. These are the two familiar development phases,
not a robustness suite. Save the raw body poses, per-step gait measurements,
ordinary finish metrics, episode summaries, and exact hashes. Replay the final
phase-zero case and verify every normalized state, raw pose, gait measurement,
distance, termination flag, and offset-corrected game time. Inspect predetermined
20%, 50%, and 80% replay samples and the full videos before describing a gait.

The total allowance is 4,194,304 training interactions plus 90,000 reserved for
evaluation and replay. Each arm has a two-hour training wall-time cap. Protocol,
source copies/hashes, local package versions, and runtime identities are frozen
before training. Failures retain their artifacts and charges; there is no
automatic continuation or replacement budget. The runner returns failure when
either worker fails.

The Mac runtime is distinct from the original Windows studies. The game's locked
bytes are reproduced exactly, including CRLF line endings, and the Mac browser
loads the assets over loopback HTTP. The runtime now also hashes the asset bundle,
HTML entry point, and seeded-RNG script. New results are compared between the two
current arms, not treated as exact replays of archived Windows trajectories.

Before study freeze, a historical-policy diagnostic and a separate 8,192-step
per-arm engineering smoke test validate geometry, serialization, training,
evaluation, replay, and audit. Their interactions are accounted separately and
their weights are discarded. The first browser startup attempt on this Mac used
zero game steps and failed on a `file://` access error; its ledger is retained.
The first smoke pair exposed a clock-verification issue near 64 game seconds:
float32 ULP spacing grows beyond the previous fixed 5e-6 bound. All physical states
matched up to the rejection. New replay checks allow the greater of 5e-6 or the
combined half-ULP rounding bounds of both clocks, plus 1e-8 for initial offset
rounding. Physical-state tolerances remain zero. A separately accounted smoke
pair verifies the corrected full-length timeout replay.

Interpret this as one paired reward screen. Favoring the treatment would require
more upright forward movement with useful progress, less knee proximity, and
supporting replay evidence. Completion regressions remain visible. A promising
result would still require fresh-seed replication and a specified robustness
suite; a failed treatment is retained as evidence rather than silently tuned.

The completed [gait-001 report](results/gait-001/report.md) records the outcome:
both policies finish, the treatment is 22.38% slower, and its improved geometric
proxies do not amount to convincing upright running in the reviewed samples.
The full checkpoint series, verified comparison video, raw trajectories, audit,
and separately charged engineering failures are retained.
