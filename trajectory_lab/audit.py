from collections import Counter, defaultdict
from dataclasses import dataclass, field
import json
from math import isfinite
from pathlib import Path
from typing import Iterable


REQUIRED = ("episode_index", "frame_index", "timestamp", "task_index",
            "observation.state", "action")


@dataclass
class Track:
    frames: int = 0
    last_frame: int | None = None
    last_time: float | None = None
    issues: Counter = field(default_factory=Counter)
    tasks: set[int] = field(default_factory=set)
    splits: set[str] = field(default_factory=set)


def integer(value: object) -> bool:
    return type(value) is int and value >= 0


def number(value: object) -> bool:
    return type(value) in (int, float) and isfinite(value)


def vector(value: object, dimensions: int) -> bool:
    return (isinstance(value, (list, tuple)) and len(value) == dimensions
            and all(number(item) for item in value))


def audit_stream(records: Iterable[tuple[str, dict]], *, fps: float, gap_factor: float,
                 state_dim: int, action_dim: int) -> dict:
    if not (isfinite(fps) and fps > 0 and isfinite(gap_factor) and gap_factor >= 1
            and min(state_dim, action_dim) > 0):
        raise ValueError("positive FPS and dimensions required; gap factor must be at least 1")
    tracks: dict[int, Track] = defaultdict(Track)
    rejected_rows: Counter = Counter()
    row_count = 0

    for split, row in records:
        row_count += 1
        episode = row.get("episode_index")
        if not integer(episode):
            rejected_rows["invalid_episode_index"] += 1
            continue
        track = tracks[episode]
        track.frames += 1
        track.splits.add(split)
        frame = row.get("frame_index")
        stamp = row.get("timestamp")
        task = row.get("task_index")

        if not integer(frame):
            track.issues["invalid_frame_index"] += 1
        else:
            if track.last_frame is not None and frame != track.last_frame + 1:
                track.issues["nonconsecutive_frame_index"] += 1
            track.last_frame = frame

        if not number(stamp) or stamp < 0:
            track.issues["invalid_timestamp"] += 1
        else:
            if track.last_time is not None:
                gap = stamp - track.last_time
                if gap <= 0:
                    track.issues["nonincreasing_timestamp"] += 1
                elif gap > gap_factor / fps + 1e-6:
                    track.issues["timestamp_gap"] += 1
            track.last_time = stamp

        if integer(task):
            track.tasks.add(task)
        else:
            track.issues["invalid_task_index"] += 1
        if not vector(row.get("observation.state"), state_dim):
            track.issues["invalid_state_vector"] += 1
        if not vector(row.get("action"), action_dim):
            track.issues["invalid_action_vector"] += 1

    output = []
    issue_counts: Counter = Counter(rejected_rows)
    for episode, track in sorted(tracks.items()):
        if len(track.tasks) > 1:
            track.issues["mixed_task_indices"] += 1
        if len(track.splits - {"unspecified"}) > 1:
            track.issues["episode_split_overlap"] += 1
        issue_counts.update(track.issues)
        output.append({
            "episode_index": episode, "frames": track.frames,
            "task_indices": sorted(track.tasks), "splits": sorted(track.splits),
            "issues": dict(sorted(track.issues.items())),
        })
    clean = sum(not episode["issues"] for episode in output)
    return {
        "rows": row_count, "episodes_seen": len(output), "clean_episodes": clean,
        "flagged_episodes": len(output) - clean, "rejected_rows": dict(rejected_rows),
        "issue_counts": dict(sorted(issue_counts.items())), "episodes": output,
        "fps": fps, "gap_factor": gap_factor, "state_dim": state_dim, "action_dim": action_dim,
        "split_check_requested": any(e["splits"] != ["unspecified"] for e in output),
    }


def read_rows(path: Path) -> Iterable[dict]:
    if path.suffix.lower() == ".jsonl":
        with path.open(encoding="utf-8-sig") as stream:
            for line_number, line in enumerate(stream, start=1):
                if line.strip():
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError(f"{path}:{line_number} must contain a JSON object")
                    yield row
        return

    if path.suffix.lower() != ".parquet":
        raise ValueError(f"unsupported dataset file: {path}")
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise RuntimeError("install pyarrow to read Parquet input") from exc

    source = pq.ParquetFile(path)
    missing = set(REQUIRED).difference(source.schema_arrow.names)
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")
    for batch in source.iter_batches(batch_size=4096, columns=list(REQUIRED)):
        yield from batch.to_pylist()


def action_window(rows: Iterable[dict], episode: int, frame: int, horizon: int,
                  action_dim: int) -> dict:
    if min(horizon, action_dim) <= 0 or min(episode, frame) < 0:
        raise ValueError("invalid episode, frame, horizon or action width")
    targets = [row for row in rows if row.get("episode_index") == episode
               and integer(row.get("frame_index")) and frame <= row["frame_index"] < frame + horizon]
    start = next((row for row in targets if row["frame_index"] == frame), None)
    if start is None:
        raise ValueError(f"episode {episode} frame {frame} is missing")
    actions = []
    for offset, row in enumerate(targets):
        if row["frame_index"] != frame + offset or not vector(row.get("action"), action_dim):
            raise ValueError("nonconsecutive frame or invalid action inside the requested window")
        actions.append(row["action"])
    missing = horizon - len(actions)
    return {
        "episode_index": episode, "frame_index": frame, "timestamp": start["timestamp"],
        "task_index": start["task_index"], "observation.state": start["observation.state"],
        "actions": actions + [[0.0] * action_dim for _ in range(missing)],
        "valid": [True] * len(actions) + [False] * missing,
    }
