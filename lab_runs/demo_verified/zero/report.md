# Paired rollout report

Backend and policy: **libero / zero-action-simulator-smoke**

| K | Success | Mean actions | Mean calls | Inference p50 (ms) | Inference p95 (ms) | Outcomes |
|---:|---:|---:|---:|---:|---:|---|
| 5 | 0/3 | 220 | 44 | 0.015 | 0.042 | timeout=3 |

First K (5) is the paired baseline.


The denominator includes every attempted episode. Error pairs are counted separately.
Successful inference timing includes the client-side call and, if remote, preprocessing and transport.
Timing and small smoke-test runs are descriptive.

## Rollout recordings

- [libero_spatial-0-k5](videos/libero_spatial-0-k5.gif)
