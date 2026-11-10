"""统一仿真评测入口的无模型/无物理回归测试。"""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

from scripts.eval.act_rollout import run_evaluation
from scripts.eval.summarize_rollouts import summarize

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("eval_mujoco", ROOT / "scripts/eval/eval_mujoco.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def schema():
    return {
        "config": {
            "type": "smolvla",
            "input_features": {
                "observation.state": {"type": "STATE", "shape": [6]},
                "observation.images.top": {"type": "VISUAL", "shape": [3, 128, 128]},
                "observation.images.wrist": {"type": "VISUAL", "shape": [3, 128, 128]},
            },
            "output_features": {"action": {"shape": [6]}},
        },
        "dataset": {"features": {
            "observation.state": {
                "dtype": "float32", "shape": [6],
                "names": ["x", "y", "z", "roll", "pitch", "yaw"]},
            "observation.images.top": {"dtype": "image", "shape": [128, 128, 3]},
            "observation.images.wrist": {"dtype": "image", "shape": [128, 128, 3]},
            "action": {"dtype": "float32", "shape": [6]},
        }},
    }


class FakeEnv:
    def __init__(self):
        self.env = self
        self.steps = 0
        self.sim = 0.0
    def reset(self, seed):
        self.steps = 0
        self.sim = 0.0
    def is_viewer_alive(self):
        return True
    def get_sim_time(self):
        return self.sim
    def step_env(self):
        self.sim += 0.1
    def loop_every(self, HZ):
        return True
    def check_success(self):
        return self.steps >= 2
    def step(self, action):
        self.steps += 1
    def render(self):
        pass


class UnifiedEvaluation(unittest.TestCase):
    def test_all_policy_labels_and_summary(self):
        rows = []
        for name in runner.POLICIES:
            result = run_evaluation(
                FakeEnv(), lambda step: ([0] * 6, 1.0), lambda: None,
                policy_name=name, num_trials=2, hz=20, timeout_s=3,
                seed=10, timeout_clock="sim",
                record=lambda event: rows.append({"split": "id", **event}),
            )
            self.assertEqual(result["successes"], 2)
        summary = summarize(rows)
        self.assertEqual({r["policy"] for r in summary}, set(runner.POLICIES))
        self.assertTrue(all(r["episodes"] == 2 and r["success_rate_pct"] == 100 for r in summary))

    def test_schema_requires_action_and_features(self):
        data = schema()
        runner.check_schema(data["config"], data["dataset"], policy_name="smolvla",
                            top_feature="observation.images.top", wrist_feature="observation.images.wrist")
        data["config"]["output_features"]["action"]["shape"] = [7]
        with self.assertRaises(ValueError):
            runner.check_schema(data["config"], data["dataset"], policy_name="smolvla",
                                top_feature="observation.images.top",
                                wrist_feature="observation.images.wrist")

    def test_rejects_mixed_action_units(self):
        data = schema()
        data["dataset"]["action_semantics"] = "so101_degrees_gripper_percent"
        with self.assertRaisesRegex(ValueError, "动作语义"):
            runner.check_schema(data["config"], data["dataset"], policy_name="smolvla",
                                top_feature="observation.images.top", wrist_feature="observation.images.wrist")

    def test_rejects_unsupported_checkpoint_conditioning(self):
        data = schema()
        data["config"]["input_features"]["observation.environment_state"] = {
            "shape": [4], "type": "ENV",
        }
        with self.assertRaisesRegex(ValueError, "不支持"):
            runner.check_schema(data["config"], data["dataset"], policy_name="smolvla",
                                top_feature="observation.images.top", wrist_feature="observation.images.wrist")

    def test_dry_run_needs_no_model_or_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            argv = ["--policy", "diffusion", "--checkpoint", str(Path(tmp) / "missing"),
                    "--dataset-root", str(Path(tmp) / "missing_ds"), "--dry-run"]
            with contextlib.redirect_stdout(io.StringIO()) as buf:
                runner.main(argv)
            self.assertIn("DRY RUN", buf.getvalue())
            self.assertIn('"policy": "diffusion"', buf.getvalue())

    def test_execute_requires_explicit_action_contract(self):
        with self.assertRaises(SystemExit):
            runner.parse_args(["--policy", "act", "--checkpoint", "/tmp/ckpt",
                               "--dataset-root", "/tmp/data"])

    def test_ood_cannot_be_only_a_new_label(self):
        with self.assertRaises(SystemExit):
            runner.parse_args([
                "--policy", "smolvla", "--checkpoint", "/tmp/c",
                "--dataset-root", "/tmp/d",
                "--action-semantics", runner.ACTION_SEMANTICS, "--split", "ood",
            ])
        options = runner.parse_args([
            "--policy", "smolvla", "--checkpoint", "/tmp/c",
            "--dataset-root", "/tmp/d",
            "--action-semantics", runner.ACTION_SEMANTICS, "--split", "ood",
            "--spawn-x-min", "0.24", "--spawn-x-max", "0.25",
        ])
        self.assertEqual(options.split, "ood")

    def test_cannot_mix_camera_features(self):
        with self.assertRaises(SystemExit):
            runner.parse_args(["--policy", "act", "--checkpoint", "/tmp/ckpt",
                               "--dataset-root", "/tmp/data", "--top-feature", "same",
                               "--wrist-feature", "same", "--dry-run"])


if __name__ == "__main__":
    unittest.main()
