# Paired rollout report

Backend and policy: **toy / deterministic-chunk-planner**

| K | Success | Mean actions | Mean calls | Inference p50 (ms) | Inference p95 (ms) | Outcomes |
|---:|---:|---:|---:|---:|---:|---|
| 5 | 10/10 | 9.5 | 2.5 | 0.002 | 0.003 | success=10 |
| 1 | 10/10 | 8.8 | 8.8 | 0.002 | 0.002 | success=10 |
| 10 | 5/10 | 10 | 1.5 | 0.002 | 0.003 | success=5, timeout=5 |

First K (5) is the paired baseline.

- K=1: 0 wins, 10 ties, 0 losses, 0 error pairs.
- K=10: 0 wins, 5 ties, 5 losses, 0 error pairs.

The denominator includes every attempted episode. Error pairs are counted separately.
Successful inference timing includes the client-side call and, if remote, preprocessing and transport.
Timing and small smoke-test runs are descriptive.
