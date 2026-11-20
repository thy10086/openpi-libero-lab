from collections import Counter, deque
from dataclasses import asdict, dataclass
from math import ceil, isfinite
from statistics import mean
from time import perf_counter
from typing import Any, Callable, Protocol, Sequence


Action = Sequence[float]
Observation = dict[str, Any]


class Environment(Protocol):
    def reset(self) -> Observation: ...
    def step(self, action: Action) -> tuple[Observation, bool]: ...
    def close(self) -> None: ...


class Policy(Protocol):
    def predict(self, observation: Observation) -> Sequence[Action]: ...


@dataclass(frozen=True)
class Case:
    task: str
    initial_state: int
    seed: int


@dataclass
class Episode:
    case: Case
    execute_steps: int
    outcome: str
    steps: int
    inference_calls: int
    inference_ms: list[float]
    rollout_ms: float
    error: str | None = None


def rollout(env: Environment, policy: Policy, case: Case, execute_steps: int,
            max_steps: int, action_dim: int) -> Episode:
    if min(execute_steps, max_steps, action_dim) <= 0:
        raise ValueError("step counts and action dimension must be positive")

    queue: deque[Action] = deque()
    latency = []
    steps = 0
    calls = 0
    started = perf_counter()
    stage = "environment_error"
    outcome = "timeout"
    error = None
    try:
        observation = env.reset()
        for _ in range(max_steps):
            if not queue:
                stage = "policy_error"
                calls += 1
                before = perf_counter()
                chunk = policy.predict(observation)
                latency.append((perf_counter() - before) * 1000)
                if len(chunk) < execute_steps:
                    raise ValueError(f"policy returned {len(chunk)} actions, need {execute_steps}")
                for action in chunk[:execute_steps]:
                    if len(action) != action_dim or not all(isfinite(x) for x in action):
                        raise ValueError(f"expected a finite {action_dim}-dimensional action")
                    queue.append(action)

            stage = "environment_error"
            observation, succeeded = env.step(queue.popleft())
            steps += 1
            if succeeded:
                outcome = "success"
                break
    except Exception as exc:
        outcome = stage
        error = f"{type(exc).__name__}: {exc}"

    return Episode(case, execute_steps, outcome, steps, calls, latency,
                   (perf_counter() - started) * 1000, error)


def paired_rollouts(cases: Sequence[Case], values: Sequence[int],
                    make_env: Callable[[Case, int], Environment],
                    make_policy: Callable[[Case], Policy], max_steps: int,
                    action_dim: int) -> list[Episode]:
    if not cases or len(set(cases)) != len(cases):
        raise ValueError("cases must be nonempty and uniquely identified")
    if not values or min(values) <= 0 or len(set(values)) != len(values):
        raise ValueError("execute step values must be unique positive integers")
    episodes = []
    for index, case in enumerate(cases):
        # Alternate order to reduce systematic warm-up bias in timings.
        order = values if index % 2 == 0 else list(reversed(values))
        for value in order:
            policy = make_policy(case)
            env = make_env(case, value)
            try:
                episode = rollout(env, policy, case, value, max_steps, action_dim)
            finally:
                env.close()
            episodes.append(episode)
    return episodes


def percentile(data: list[float], fraction: float) -> float | None:
    if not data:
        return None
    return round(sorted(data)[ceil(fraction * len(data)) - 1], 3)


def summarize(episodes: list[Episode], values: Sequence[int]) -> dict:
    by_value = {
        value: {episode.case: episode for episode in episodes if episode.execute_steps == value}
        for value in values
    }
    baseline = by_value[values[0]]
    summary = []
    comparisons = []
    for value, rows in by_value.items():
        if set(rows) != set(baseline) or not rows:
            raise ValueError("all settings must share the same nonempty cases")
        counts = Counter(row.outcome for row in rows.values())
        latencies = [item for row in rows.values() for item in row.inference_ms]
        summary.append({
            "execute_steps": value,
            "episodes": len(rows),
            "successes": counts["success"],
            "success_rate": round(counts["success"] / len(rows), 4),
            "outcomes": dict(counts),
            "mean_actions": round(mean(row.steps for row in rows.values()), 3),
            "mean_inference_calls": round(mean(row.inference_calls for row in rows.values()), 3),
            "inference_p50_ms": percentile(latencies, 0.50),
            "inference_p95_ms": percentile(latencies, 0.95),
            "rollout_p50_ms": percentile([row.rollout_ms for row in rows.values()], 0.50),
        })
        if value == values[0]:
            continue
        paired = Counter()
        for case, row in rows.items():
            reference = baseline[case]
            if any(x.outcome.endswith("_error") for x in (row, reference)):
                paired["invalid_pairs"] += 1
                continue
            delta = int(row.outcome == "success") - int(reference.outcome == "success")
            paired[{1: "wins", 0: "ties", -1: "losses"}[delta]] += 1
        comparisons.append({
            "baseline": values[0], "candidate": value,
            **{key: paired[key] for key in ("wins", "ties", "losses", "invalid_pairs")},
        })
    return {"settings": summary, "paired": comparisons,
            "episodes": [asdict(episode) for episode in episodes]}
