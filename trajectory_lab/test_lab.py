import unittest

from .audit import action_window, audit_stream
from .evaluation import Case, paired_rollouts, rollout, summarize
from .toy import ChunkPlanner, MovingTarget


def row(episode, frame, time, *, action=None):
    return {"episode_index": episode, "frame_index": frame, "timestamp": time,
            "task_index": 1, "observation.state": [0, 1],
            "action": [0.1] if action is None else action}


class AuditTest(unittest.TestCase):
    def check(self, data):
        return audit_stream(data, fps=10, gap_factor=1.5, state_dim=2, action_dim=1)

    def test_accepts_ordered_tracks_across_shards(self):
        result = self.check([("train", row(9, 0, 0.0)), ("train", row(9, 1, 0.1)),
                             ("validation", row(23, 0, 0.0))])
        self.assertEqual(result["clean_episodes"], 2)
        self.assertFalse(result["issue_counts"])

    def test_rejects_split_leak_gap_and_nonfinite_action(self):
        result = self.check([("train", row(9, 0, 0.0)), ("validation", row(9, 3, 0.4,
                                                                       action=[float("nan")]))])
        self.assertEqual(result["issue_counts"], {"episode_split_overlap": 1,
                         "invalid_action_vector": 1, "nonconsecutive_frame_index": 1,
                         "timestamp_gap": 1})

    def test_tail_window_is_masked_and_stays_in_episode(self):
        rows = [row(9, 0, 0.0), row(9, 1, 0.1), row(10, 0, 0.0, action=[9])]
        result = action_window(rows, 9, 1, 3, 1)
        self.assertEqual(result["actions"], [[0.1], [0.0], [0.0]])
        self.assertEqual(result["valid"], [True, False, False])

    def test_window_rejects_gap_and_absent_start(self):
        with self.assertRaises(ValueError):
            action_window([row(9, 0, 0), row(9, 2, 0.2)], 9, 0, 3, 1)
        with self.assertRaises(ValueError):
            action_window([row(10, 0, 0)], 9, 0, 3, 1)


class RolloutTest(unittest.TestCase):
    def test_paired_runs_reset_each_case_and_count_actual_calls(self):
        cases = [Case("stationary", 0, 7), Case("target_moves", 0, 7)]
        runs = paired_rollouts(cases, [5, 1, 10], lambda case, _: MovingTarget(case),
                               lambda _: ChunkPlanner(10), max_steps=14, action_dim=1)
        result = summarize(runs, [5, 1, 10])
        self.assertEqual([row["successes"] for row in result["settings"]], [2, 2, 1])
        self.assertEqual(result["paired"][1]["losses"], 1)
        for run in runs:
            if run.execute_steps == 1:
                self.assertEqual(run.inference_calls, run.steps)

    def test_short_chunk_counts_as_policy_error_not_task_timeout(self):
        case = Case("stationary", 0, 7)
        run = rollout(MovingTarget(case), ChunkPlanner(1), case,
                      execute_steps=5, max_steps=14, action_dim=1)
        self.assertEqual(run.outcome, "policy_error")
        self.assertEqual((run.inference_calls, run.steps), (1, 0))

    def test_error_pair_kept_separate(self):
        case = Case("stationary", 0, 7)
        runs = paired_rollouts([case], [1, 10], lambda case, _: MovingTarget(case),
                               lambda _: ChunkPlanner(2), max_steps=14, action_dim=1)
        self.assertEqual(summarize(runs, [1, 10])["paired"][0]["invalid_pairs"], 1)


if __name__ == "__main__":
    unittest.main()
