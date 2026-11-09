import importlib.util
import pathlib
import unittest

SCRIPT = pathlib.Path(__file__).resolve().parents[1] / 'scripts/eval/summarize_rollouts.py'
spec = importlib.util.spec_from_file_location('summarize_rollouts', SCRIPT)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class TestSummary(unittest.TestCase):
    def test_two_policies(self):
        rows = mod.summarize([
            {'policy': 'act', 'split': 'id', 'success': True, 'started': True,
             'grasped': True, 'placed': True, 'completion_time_s': 10.0,
             'inference_ms': [10, 20]},
            {'policy': 'act', 'split': 'id', 'success': False, 'started': False,
             'grasped': False, 'placed': False, 'inference_ms': [30]},
            {'policy': 'smolvla', 'split': 'ood', 'success': False},
        ])
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]['success_rate_pct'], 50.0)
        self.assertEqual(rows[0]['startup_fail_pct'], 50.0)
        self.assertEqual(rows[0]['avg_success_time_s'], 10)
        self.assertEqual(rows[1]['startup_fail_pct'], None)

    def test_reject_invalid_records(self):
        for event in [[], {'policy': [], 'success': True},
                      {'policy': '', 'success': True},
                      {'policy': 'act', 'success': 1},
                      {'policy': 'act', 'success': True, 'inference_ms': [float('nan')]},
                      {'policy': 'act', 'success': True, 'inference_ms': [-1]},
                      {'policy': 'act', 'success': True, 'completion_time_s': True},
                      {'policy': 'act', 'success': True, 'started': 'false'}]:
            with self.subTest(event=event), self.assertRaises(ValueError):
                mod.summarize([event])

    def test_wilson(self):
        lo, hi = mod.wilson_interval(0, 20)
        self.assertLess(lo, hi)
        self.assertEqual(lo, 0.0)


if __name__ == '__main__':
    unittest.main()
