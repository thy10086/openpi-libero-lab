import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path
import platform
from urllib.parse import urlsplit

from .audit import action_window, audit_stream, read_rows
from .evaluation import Case, paired_rollouts, summarize
from .libero_backend import HORIZONS
from .toy import ChunkPlanner, MovingTarget


def positive(raw: str) -> int:
    value = int(raw)
    if value < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return value


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")


def source_hash(path: Path) -> str:
    # Python 3.10 compatibility.
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit(args: argparse.Namespace) -> int:
    sources = []
    for value in args.input:
        split, separator, path = value.partition("=")
        if not separator:
            split, path = "unspecified", value
        if not split or not path:
            raise ValueError("input must be PATH or SPLIT=PATH")
        sources.append((split, Path(path)))

    inputs = [{"split": split, "file": path.name, "sha256": source_hash(path)}
              for split, path in sources]
    rows = ((split, row) for split, path in sources for row in read_rows(path))
    data = audit_stream(rows, fps=args.fps, gap_factor=args.gap_factor,
                        state_dim=args.state_dim, action_dim=args.action_dim)
    data.update(dataset=args.dataset, revision=args.revision, inputs=inputs,
                scope="record columns; video integrity and full episode coverage are not measured")
    if data["rows"] == 0:
        raise ValueError("no data rows")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, data)
    print(f"{data['rows']} rows, {data['episodes_seen']} episodes, "
          f"{data['flagged_episodes']} flagged; {args.output}")
    return int(args.strict and bool(data["issue_counts"]))


def window(args: argparse.Namespace) -> int:
    data = action_window(read_rows(args.input), args.episode, args.frame,
                         args.horizon, args.action_dim)
    print(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


def train_bc(args: argparse.Namespace) -> int:
    from .behavior import train

    inputs = [{"file": path.name, "sha256": source_hash(path)} for path in args.input]
    data = train(args.input, args.output, args.fps, args.seed, args.epochs,
                 args.horizon, args.task_index)
    data.update(inputs=inputs, dataset=args.dataset, revision=args.revision, fps=args.fps)
    write_json(args.output / "training.json", data)
    print(f"Validation MAE: learned {data['validation_mae']:.6f}, "
          f"train-mean baseline {data['mean_action_validation_mae']:.6f}")
    return 0


def report(output: Path, data: dict) -> None:
    with (output / "episodes.csv").open("w", newline="", encoding="utf-8") as stream:
        columns = ["task", "initial_state", "seed", "execute_steps", "outcome", "steps",
                   "inference_calls", "inference_ms", "rollout_ms", "error"]
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for episode in data["episodes"]:
            fields = {**episode["case"], **{k: v for k, v in episode.items() if k != "case"}}
            writer.writerow(fields)

    lines = [
        "# Paired rollout report", "",
        f"Backend and policy: **{data['backend']} / {data['policy']}**", "",
        "| K | Success | Mean actions | Mean calls | Inference p50 (ms) | Inference p95 (ms) | Outcomes |",
        "|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in data["settings"]:
        outcomes = ", ".join(f"{key}={value}" for key, value in sorted(row["outcomes"].items()))
        lines.append(
            f"| {row['execute_steps']} | {row['successes']}/{row['episodes']} | "
            f"{row['mean_actions']} | {row['mean_inference_calls']} | "
            f"{row['inference_p50_ms']} | {row['inference_p95_ms']} | {outcomes} |"
        )
    lines += ["", f"First K ({data['execute_steps'][0]}) is the paired baseline.", ""]
    for row in data["paired"]:
        lines.append(f"- K={row['candidate']}: {row['wins']} wins, {row['ties']} ties, "
                     f"{row['losses']} losses, {row['invalid_pairs']} error pairs.")
    lines += [
        "", "The denominator includes every attempted episode. Error pairs are counted separately.",
        "Successful inference timing includes the client-side call and, if remote, preprocessing and transport.",
        "Timing and small smoke-test runs are descriptive.",
    ]
    if data.get("videos"):
        lines += ["", "## Rollout recordings", ""]
        lines.extend(f"- [{Path(video).stem}]({video})" for video in data["videos"])
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(output / "result.json", data)


def evaluate(args: argparse.Namespace) -> int:
    values = args.execute_steps
    if len(set(values)) != len(values):
        raise ValueError("execution steps must be unique")
    if args.output.exists():
        raise ValueError(f"output already exists: {args.output}")
    if args.seed < 0:
        raise ValueError("seed must be nonnegative")

    policy = None
    extra = {}
    horizon = 10
    if args.command == "toy":
        cases = [Case(task, trial, args.seed + trial) for task in ("stationary", "target_moves")
                 for trial in range(args.trials)]
        make_env = lambda case, _: MovingTarget(case)
        max_steps = args.max_steps or 14
        make_policy = lambda _: ChunkPlanner(10)
        label = "deterministic-chunk-planner"
        dim = 1
    else:
        from .libero_backend import LiberoSuite, RemotePolicy, ZeroPolicy

        if len(set(args.tasks)) != len(args.tasks):
            raise ValueError("task IDs must be unique")
        if args.start_index < 0:
            raise ValueError("start index must be nonnegative")
        if args.policy == "openpi":
            url = urlsplit(args.server)
            if url.scheme not in ("ws", "wss") or not url.hostname or url.username or url.password:
                raise ValueError("server must be a ws:// or wss:// URL without credentials")
            if not args.checkpoint_label:
                raise ValueError("provide the server checkpoint with --checkpoint-label")
            if not 0 < args.timeout <= 600:
                raise ValueError("timeout must be between 0 and 600 seconds")
            policy = RemotePolicy(args.server, args.timeout)
            make_policy = lambda _: policy
            label = args.checkpoint_label
        elif args.policy == "bc":
            from .behavior import StateOnlyPolicy

            if args.checkpoint is None:
                raise ValueError("--policy bc requires --checkpoint PATH")
            baseline = StateOnlyPolicy(args.checkpoint)
            make_policy = lambda _: baseline
            label = "trained-state-only-bc"
            horizon = baseline.horizon
            extra["checkpoint_sha256"] = source_hash(args.checkpoint)
        else:
            make_policy = lambda _: ZeroPolicy(10)
            label = "zero-action-simulator-smoke"
        video_dir = args.output / "videos" if args.record_video else None
        suite = LiberoSuite(args.suite, args.tasks, args.trials, args.seed,
                            args.start_index, video_dir)
        cases, make_env = suite.cases, suite.make_env
        max_steps = args.max_steps or HORIZONS[args.suite]
        extra.update(suite=args.suite, tasks=args.tasks, start_index=args.start_index,
                     server=args.server if policy else None)
        dim = 7

    if policy is None and max(values) > horizon:
        raise ValueError(f"this policy provides at most {horizon} actions per inference")
    try:
        if policy is not None:
            policy.open()
        episodes = paired_rollouts(cases, values, make_env, make_policy, max_steps, dim)
    finally:
        if policy is not None:
            policy.close()
    data = summarize(episodes, values)
    data.update(backend=args.command, policy=label, execute_steps=values,
                seed=args.seed, trials=args.trials, max_steps=max_steps,
                python=platform.python_version(), **extra)
    args.output.mkdir(parents=True, exist_ok=True)
    data["videos"] = [path.relative_to(args.output).as_posix()
                      for path in sorted(args.output.glob("videos/*.gif"))]
    report(args.output, data)
    print((args.output / "report.md").read_text(encoding="utf-8"))
    return 2 if any(row.outcome.endswith("_error") for row in episodes) else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__ or "OpenPI trajectory and evaluation lab")
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("audit", help="stream an episode-level quality audit")
    inspect.add_argument("--input", nargs="+", required=True, help="ordered PATH or SPLIT=PATH inputs")
    inspect.add_argument("--fps", type=float, required=True)
    inspect.add_argument("--gap-factor", type=float, default=1.5)
    inspect.add_argument("--state-dim", type=positive, default=8)
    inspect.add_argument("--action-dim", type=positive, default=7)
    inspect.add_argument("--dataset", default="local")
    inspect.add_argument("--revision", default=None)
    inspect.add_argument("--strict", action="store_true")
    inspect.add_argument("--output", type=Path, required=True)
    inspect.set_defaults(handler=audit)

    sample = commands.add_parser("window", help="show one padded action target and mask")
    sample.add_argument("--input", type=Path, required=True)
    sample.add_argument("--episode", type=int, required=True)
    sample.add_argument("--frame", type=int, required=True)
    sample.add_argument("--horizon", type=positive, default=10)
    sample.add_argument("--action-dim", type=positive, default=7)
    sample.set_defaults(handler=window)

    learner = commands.add_parser("train-bc", help="fit a small state-only imitation baseline on CPU")
    learner.add_argument("--input", nargs="+", type=Path, required=True)
    learner.add_argument("--fps", type=float, required=True)
    learner.add_argument("--epochs", type=positive, default=25)
    learner.add_argument("--horizon", type=positive, default=10)
    learner.add_argument("--seed", type=int, default=7)
    learner.add_argument("--task-index", type=int)
    learner.add_argument("--dataset", default="local")
    learner.add_argument("--revision")
    learner.add_argument("--output", type=Path, required=True)
    learner.set_defaults(handler=train_bc)

    for backend in ("toy", "libero"):
        run = commands.add_parser(backend, help=f"run paired closed-loop trials in {backend}")
        run.add_argument("--execute-steps", type=positive, nargs="+", default=[5, 10])
        run.add_argument("--trials", type=positive, default=3)
        run.add_argument("--seed", type=int, default=7)
        run.add_argument("--max-steps", type=positive)
        run.add_argument("--output", type=Path, required=True)
        run.set_defaults(handler=evaluate)
        if backend == "libero":
            run.add_argument("--suite", choices=HORIZONS, default="libero_spatial")
            run.add_argument("--tasks", type=int, nargs="+", default=[0])
            run.add_argument("--start-index", type=int, default=0)
            run.add_argument("--record-video", action="store_true")
            run.add_argument("--policy", choices=("zero", "bc", "openpi"), required=True)
            run.add_argument("--checkpoint", type=Path, help="checkpoint for the state-only BC policy")
            run.add_argument("--server", default="ws://127.0.0.1:8000")
            run.add_argument("--timeout", type=float, default=30)
            run.add_argument("--checkpoint-label")
    args = parser.parse_args()
    try:
        return args.handler(args)
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
