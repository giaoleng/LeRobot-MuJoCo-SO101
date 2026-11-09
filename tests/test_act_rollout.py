import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest

ROOT = Path(__file__).resolve().parents[1]


def module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


rollout = module('act_rollout', 'scripts/eval/act_rollout.py')
summary = module('summary', 'scripts/eval/summarize_rollouts.py')


class FakeEnv:
    def __init__(self, success_after=2, alive=True):
        self.env = self; self.success_after = success_after; self.alive = alive
        self.time = 0; self.actions = 0; self.seeds = []
    def reset(self, seed):
        self.seeds.append(seed); self.time = 0; self.actions = 0
    def get_sim_time(self): return self.time
    def is_viewer_alive(self): return self.alive
    def step_env(self): self.time += .1
    def loop_every(self, HZ): return True
    def check_success(self): return self.actions >= self.success_after
    def step(self, action): self.actions += 1
    def render(self): pass


class RolloutTests(unittest.TestCase):
    def run_env(self, env, record, **kw):
        return rollout.run_evaluation(env, lambda step: ([0] * 6, 2.5), lambda: None,
                                      num_trials=kw.get('num_trials', 1), hz=20, timeout_s=.5,
                                      seed=42, timeout_clock='sim', record=record, clock=lambda: 100.)

    def test_two_successes_change_seed_and_reset_episode(self):
        rows = []; env = FakeEnv()
        result = self.run_env(env, rows.append, num_trials=2)
        self.assertEqual(result, {'completed': 2, 'successes': 2, 'timeouts': 0})
        self.assertEqual(env.seeds, [42, 43])
        self.assertEqual(rows[0]['control_steps'], 2)
        self.assertEqual(rows[1]['inference_ms'], [2.5, 2.5])

    def test_timeout_counts_as_completed_failure(self):
        rows = []; result = self.run_env(FakeEnv(success_after=100), rows.append)
        self.assertEqual(result['timeouts'], 1)
        self.assertFalse(rows[0]['success']); self.assertTrue(rows[0]['completed'])
        self.assertEqual(rows[0]['reason'], 'timeout')

    def test_interrupted_episode_excluded_from_success_denominator(self):
        rows = []; result = self.run_env(FakeEnv(alive=False), rows.append)
        self.assertEqual(result['completed'], 0)
        self.assertEqual(rows[0]['reason'], 'viewer_closed')
        self.assertEqual(summary.summarize(rows), [])

    def test_policy_error_is_logged_then_raised(self):
        rows = []
        def broken(step): raise RuntimeError('test')
        with self.assertRaises(RuntimeError):
            rollout.run_evaluation(FakeEnv(), broken, lambda: None, num_trials=1, hz=20,
                                   timeout_s=1, record=rows.append, timeout_clock='sim')
        self.assertEqual(rows[0]['error_type'], 'RuntimeError')
        self.assertFalse(rows[0]['completed'])

    def test_jsonl_round_trip_into_existing_aggregator(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'episodes.jsonl'
            rec = rollout.JsonlRecorder(path, {'split': 'id', 'checkpoint': '/weights/act'})
            try: self.run_env(FakeEnv(), rec)
            finally: rec.close()
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(summary.summarize(rows)[0]['success_rate_pct'], 100.)
            self.assertEqual(rows[0]['checkpoint'], '/weights/act')
            with self.assertRaises(FileExistsError): rollout.JsonlRecorder(path, {})

    def test_invalid_latency_does_not_reach_environment(self):
        env = FakeEnv(); rows = []
        with self.assertRaises(ValueError):
            rollout.run_evaluation(env, lambda step: ([0]*6, float('nan')), lambda: None,
                                   num_trials=1, hz=20, timeout_s=1, record=rows.append)
        self.assertEqual(env.actions, 0)


if __name__ == '__main__': unittest.main()
