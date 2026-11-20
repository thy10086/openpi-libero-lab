"""Bridge the paired runner to LIBERO and the official OpenPI wire protocol."""

from pathlib import Path

from .evaluation import Case


HORIZONS = {
    "libero_spatial": 220, "libero_object": 280, "libero_goal": 300,
    "libero_10": 520, "libero_90": 400,
}


class LiberoSuite:
    def __init__(self, name: str, task_ids: list[int], trials: int, seed: int,
                 start_index: int, videos: Path | None) -> None:
        from libero.libero import benchmark

        self.suite = benchmark.get_benchmark_dict()[name]()
        self.tasks = {}
        self.cases = []
        self.start_index = start_index
        self.videos = videos
        for task_id in task_ids:
            if not 0 <= task_id < self.suite.n_tasks:
                raise ValueError(f"task ID {task_id} out of range for {name}")
            task = self.suite.get_task(task_id)
            states = self.suite.get_task_init_states(task_id)
            if start_index + trials > len(states):
                raise ValueError(f"task {task_id} has only {len(states)} initial states")
            key = f"{name}/{task_id}"
            self.tasks[key] = (task, states)
            self.cases.extend(Case(key, i, seed) for i in range(start_index, start_index + trials))

    def make_env(self, case: Case, execute_steps: int) -> "LiberoEnv":
        task, states = self.tasks[case.task]
        video = None
        if self.videos is not None and case.initial_state == self.start_index:
            video = self.videos / f"{case.task.replace('/', '-')}-k{execute_steps}.gif"
        return LiberoEnv(task, states[case.initial_state], case.seed, video)


class LiberoEnv:
    def __init__(self, task, state, seed: int, video: Path | None) -> None:
        from examples.libero import main as upstream

        self.env, self.prompt = upstream._get_libero_env(task, upstream.LIBERO_ENV_RESOLUTION, seed)
        self.initial_state = state
        self.dummy_action = upstream.LIBERO_DUMMY_ACTION
        self.seed = seed
        self.video = video
        self.frames = []

    def observe(self, observation: dict) -> dict:
        if self.video is not None:
            from PIL import Image

            frame = Image.fromarray(observation["agentview_image"][::-1, ::-1].copy())
            self.frames.append(frame.resize((160, 160), Image.Resampling.BILINEAR))
        return {"raw": observation, "prompt": self.prompt}

    def reset(self) -> dict:
        import numpy as np

        np.random.seed(self.seed)
        self.env.seed(self.seed)
        self.env.reset()
        self.env.set_init_state(self.initial_state)
        for _ in range(10):
            observation, _, _, _ = self.env.step(self.dummy_action)
        return self.observe(observation)

    def step(self, action) -> tuple[dict, bool]:
        observation, _, _, _ = self.env.step(list(action))
        return self.observe(observation), bool(self.env.check_success())

    def close(self) -> None:
        self.env.close()
        if self.frames:
            self.video.parent.mkdir(parents=True, exist_ok=True)
            self.frames[0].save(self.video, save_all=True, append_images=self.frames[1:],
                                duration=100, loop=0)


class ZeroPolicy:
    """Fixed no-op control used to smoke-test the real simulator."""

    def __init__(self, horizon: int) -> None:
        self.horizon = horizon

    def predict(self, observation: dict) -> list[list[float]]:
        from examples.libero.main import LIBERO_DUMMY_ACTION

        return [list(LIBERO_DUMMY_ACTION) for _ in range(self.horizon)]


class RemotePolicy:
    def __init__(self, url: str, timeout: float) -> None:
        self.url = url
        self.timeout = timeout
        self.connection = None

    def open(self) -> None:
        from openpi_client import msgpack_numpy
        from websockets.sync.client import connect

        self.connection = connect(self.url, compression=None, max_size=None,
                                  open_timeout=self.timeout, close_timeout=2)
        try:
            metadata = self.connection.recv(timeout=self.timeout)
            msgpack_numpy.unpackb(metadata)
        except Exception:
            self.close()
            raise

    def predict(self, observation: dict):
        from examples.libero.main import make_policy_input, prepare_images
        from openpi_client import msgpack_numpy

        if self.connection is None:
            self.open()
        raw = observation["raw"]
        image, wrist = prepare_images(raw, 224)
        packet = make_policy_input(raw, image, wrist, observation["prompt"])
        try:
            self.connection.send(msgpack_numpy.Packer().pack(packet))
            response = self.connection.recv(timeout=self.timeout)
            if isinstance(response, str):
                raise RuntimeError(f"policy server error: {response}")
            return msgpack_numpy.unpackb(response)["actions"]
        except Exception:
            # A late response must never be interpreted as the next request's result.
            self.close()
            raise

    def close(self) -> None:
        if self.connection is not None:
            self.connection.close()
            self.connection = None
