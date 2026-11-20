"""Small CPU imitation baseline on the numeric columns of LIBERO demonstrations."""

from collections import defaultdict
from pathlib import Path
from random import Random

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .audit import audit_stream, read_rows


def make_model(horizon: int) -> nn.Module:
    return nn.Sequential(nn.Linear(8, 128), nn.ReLU(), nn.Linear(128, 128),
                         nn.ReLU(), nn.Linear(128, horizon * 7), nn.Unflatten(1, (horizon, 7)))


def window_tensors(episodes: list[list[dict]], horizon: int):
    states, chunks, masks = [], [], []
    for rows in episodes:
        for index, row in enumerate(rows):
            block = [item["action"] for item in rows[index:index + horizon]]
            missing = horizon - len(block)
            states.append(row["observation.state"])
            chunks.append(block + [[0.0] * 7 for _ in range(missing)])
            masks.append([True] * len(block) + [False] * missing)
    return (torch.tensor(states, dtype=torch.float32),
            torch.tensor(chunks, dtype=torch.float32), torch.tensor(masks))


def train(paths: list[Path], output: Path, fps: float, seed: int, epochs: int, horizon: int,
          task_index: int | None) -> dict:
    if output.exists():
        raise ValueError(f"output already exists: {output}")
    if min(epochs, horizon) < 1 or seed < 0:
        raise ValueError("epochs and horizon must be positive; seed must be nonnegative")
    episodes = defaultdict(list)
    for path in paths:
        for row in read_rows(path):
            if task_index is None or row["task_index"] == task_index:
                episodes[row["episode_index"]].append(row)
    audit = audit_stream((("train_pool", row) for rows in episodes.values() for row in rows),
                         fps=fps, gap_factor=1.5, state_dim=8, action_dim=7)
    if audit["issue_counts"]:
        raise ValueError(f"training data has audit violations: {audit['issue_counts']}")
    ids = sorted(episodes)
    if len(ids) < 3:
        raise ValueError("at least three completed demonstrations are needed")
    if any(rows[0]["frame_index"] != 0 for rows in episodes.values()):
        raise ValueError("training input starts mid-episode; include the preceding shard")

    Random(seed).shuffle(ids)
    val_count = max(1, round(len(ids) * 0.2))
    val_ids, train_ids = ids[:val_count], ids[val_count:]
    train_rows = [episodes[key] for key in train_ids]
    x, y, valid = window_tensors(train_rows, horizon)
    vx, vy, vvalid = window_tensors([episodes[key] for key in val_ids], horizon)

    state_mean = x.mean(0)
    state_std = x.std(0, unbiased=False).clamp_min(1e-6)
    actions = torch.tensor([row["action"] for ep in train_rows for row in ep], dtype=torch.float32)
    action_mean = actions.mean(0)
    action_std = actions.std(0, unbiased=False).clamp_min(1e-6)
    x = (x - state_mean) / state_std
    vx = (vx - state_mean) / state_std
    yn = (y - action_mean) / action_std

    torch.manual_seed(seed)
    torch.set_num_threads(1)
    model = make_model(horizon)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    batches = DataLoader(TensorDataset(x, yn, valid), batch_size=128, shuffle=True,
                         generator=torch.Generator().manual_seed(seed))
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        for inputs, targets, mask in batches:
            optimizer.zero_grad(set_to_none=True)
            predictions = model(inputs)
            loss = nn.functional.mse_loss(predictions[mask], targets[mask])
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            eval_actions = model(vx) * action_std + action_mean
            mae = (eval_actions[vvalid] - vy[vvalid]).abs().mean().item()
        history.append({"epoch": epoch, "validation_mae": round(mae, 6)})

    output.mkdir(parents=True)
    torch.save({
        "model": model.state_dict(), "horizon": horizon,
        "state_mean": state_mean, "state_std": state_std,
        "action_mean": action_mean, "action_std": action_std,
    }, output / "policy.pt")
    return {
        "architecture": "state-only MLP", "seed": seed, "horizon": horizon,
        "task_index_filter": task_index,
        "train_episodes": sorted(train_ids), "validation_episodes": sorted(val_ids),
        "train_tasks": sorted({row["task_index"] for ep in train_rows for row in ep}),
        "validation_tasks": sorted({row["task_index"] for key in val_ids for row in episodes[key]}),
        "train_windows": len(x), "validation_windows": len(vx),
        "mean_action_validation_mae": round((vy[vvalid] - action_mean).abs().mean().item(), 6),
        "history": history, "validation_mae": history[-1]["validation_mae"],
        "torch_version": str(torch.__version__),
    }


class StateOnlyPolicy:
    def __init__(self, checkpoint: Path) -> None:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.horizon = state["horizon"]
        self.model = make_model(self.horizon)
        self.model.load_state_dict(state["model"])
        self.model.eval()
        self.state_mean, self.state_std = state["state_mean"], state["state_std"]
        self.action_mean, self.action_std = state["action_mean"], state["action_std"]

    @torch.inference_mode()
    def predict(self, observation: dict) -> list[list[float]]:
        from examples.libero.main import policy_state

        state = torch.tensor(policy_state(observation["raw"]), dtype=torch.float32)
        actions = self.model(((state - self.state_mean) / self.state_std).unsqueeze(0))[0]
        return (actions * self.action_std + self.action_mean).clamp(-1.0, 1.0).tolist()
