# Gait-study engineering evidence

These diagnostics preceded the main training runs. No weights were reused. All failed attempts remain in the archive; videos and proprietary game assets are excluded.

| Attempt | Charged steps | Outcome |
| --- | ---: | --- |
| gait-diagnostic-001 | 0 | Browser startup failed on file:// access before any game steps. |
| gait-diagnostic-002 | 3,000 | Historical speed-policy geometry check and verified replay. |
| gait-smoke-001 | 45,991 | Training/evaluation succeeded; both long replays rejected by the old clock bound. |
| gait-smoke-002 | 46,384 | Complete smoke pair, including full timeout replays and independent artifact audit. |

Total additional engineering cost: **95,375 environment steps**.

The first smoke's physical states matched through the failed checks. The fixed clock check was additionally validated against 5,000 saved transitions with no physical-state tolerance, then exercised through both complete timeout replays in smoke-002. Its audit also recomputed all raw-pose gait metrics, verified identical paired initial policy weights, and reconciled the budgets.

The historical diagnostic finishes both reset phases. Across those episodes its mean pelvis height is about 0.406 m, knees are near the ground for about 98% of samples, and less than 0.04% of positive forward displacement qualifies as upright under the exploratory thresholds.
