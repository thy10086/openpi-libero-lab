"""Small deterministic simulator for validating the paired runner on CPU."""

from random import Random

from .evaluation import Case


class MovingTarget:
    def __init__(self, case: Case) -> None:
        self.case = case

    def reset(self) -> dict:
        self.position = Random(self.case.seed).uniform(-0.1, 0.1)
        self.target = 1.2
        self.ticks = 0
        self.hold = 0
        return self.observation()

    def observation(self) -> dict:
        return {"position": self.position, "target": self.target}

    def step(self, action: list[float]) -> tuple[dict, bool]:
        self.position += action[0]
        self.ticks += 1
        if self.case.task == "target_moves" and self.ticks == 4:
            self.target = -0.55
        self.hold = self.hold + 1 if abs(self.target - self.position) < 0.04 else 0
        return self.observation(), self.hold >= 2

    def close(self) -> None:
        pass


class ChunkPlanner:
    def __init__(self, horizon: int) -> None:
        self.horizon = horizon

    def predict(self, observation: dict) -> list[list[float]]:
        position = observation["position"]
        target = observation["target"]
        actions = []
        for _ in range(self.horizon):
            action = max(-0.25, min(0.25, target - position))
            actions.append([action])
            position += action
        return actions
