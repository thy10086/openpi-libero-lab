"""Derive the LIBERO-Spatial task 0 sample from pinned LeRobot v3 data shards."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


REVISION = "a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4"
TASK_INDEX = 34
TASK = "pick up the black bowl between the plate and the ramekin and place it on the plate"


def prepare(shards: list[Path], metadata: Path, output: Path) -> None:
    expected = {row["episode_index"]: row["length"]
                for row in pq.read_table(metadata, columns=["episode_index", "length"]).to_pylist()}
    target = []
    source_hashes = []
    for shard in sorted(shards):
        table = pq.read_table(shard)
        subset = table.filter(pc.equal(table["task_index"], TASK_INDEX))
        if len(subset):
            target.append(subset)
            source_hashes.append({"path": "data/chunk-000/" + shard.name,
                                  "sha256": sha256(shard.read_bytes()).hexdigest()})
    if not target:
        raise ValueError("task 34 is missing in these shards")
    result = pa.concat_tables(target)
    counts = Counter(result["episode_index"].to_pylist())
    if any(expected.get(key) != count for key, count in counts.items()):
        raise ValueError("at least one episode lacks its declared frame count")
    if output.exists():
        raise ValueError(f"sample already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(result, output)
    manifest = {
        "dataset": "lerobot/libero", "revision": REVISION, "fps": 10,
        "dataset_task_index": TASK_INDEX, "task": TASK, "simulator_task": "libero_spatial/0",
        "episode_metadata": {
            "path": "meta/episodes/chunk-000/file-000.parquet",
            "sha256": sha256(metadata.read_bytes()).hexdigest(),
        },
        "sources": source_hashes, "episodes": dict(sorted(counts.items())),
        "rows": len(result), "file": output.name, "sha256": sha256(output.read_bytes()).hexdigest(),
    }
    output.with_suffix(".provenance.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"{len(counts)} complete episodes and {len(result)} rows for {TASK}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", nargs="+", type=Path, required=True)
    parser.add_argument("--episode-metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.shards, args.episode_metadata, args.output)


if __name__ == "__main__":
    main()
