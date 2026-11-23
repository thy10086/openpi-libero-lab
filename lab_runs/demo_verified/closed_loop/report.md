# Paired rollout report

Backend and policy: **libero / trained-state-only-bc**

| K | Success | Mean actions | Mean calls | Inference p50 (ms) | Inference p95 (ms) | Outcomes |
|---:|---:|---:|---:|---:|---:|---|
| 5 | 2/3 | 124 | 25 | 0.552 | 0.929 | success=2, timeout=1 |
| 10 | 2/3 | 125 | 12.667 | 0.5 | 0.953 | success=2, timeout=1 |

First K (5) is the paired baseline.

- K=10: 0 wins, 3 ties, 0 losses, 0 error pairs.

The denominator includes every attempted episode. Error pairs are counted separately.
Successful inference timing includes the client-side call and, if remote, preprocessing and transport.
Timing and small smoke-test runs are descriptive.

## Rollout recordings

- [libero_spatial-0-k10](videos/libero_spatial-0-k10.gif)
- [libero_spatial-0-k5](videos/libero_spatial-0-k5.gif)
