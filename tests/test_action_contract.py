"""无 MuJoCo 的控制接口回归；执行原始方法，隔离物理和viewer。"""
import ast
import copy
from pathlib import Path
import unittest
from types import SimpleNamespace as NS
import dataclasses
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def isolated_env_class():
    tree = ast.parse((ROOT / 'mujoco_env/y_env.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'SimpleEnv')
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in {'step', 'get_action_target', '_make_gripper_ctrl'}]
    space = {'np': np, 'copy': copy}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), 'y_env.py', 'exec'), space)
    return space['SimpleEnv']


class ActionContract(unittest.TestCase):
    def setUp(self):
        self.env = isolated_env_class()()
        e = self.env
        e.action_type = 'joint_angle'; e.state_type = 'joint_angle'; e.n_arm_joints = 5
        e.last_q = np.zeros(5); e.q = np.zeros(6)
        e.joint_mins = np.full(5, -1.); e.joint_maxs = np.ones(5)
        e.gripper_continuous = True; e.gripper_open_rad = 1.7; e.gripper_motion_span = 1.9
        e.n_gripper_actuators = 1; e.gripper_actuator_scales = np.ones(1)
        e.get_joint_state = lambda: np.zeros(6)  # Physics has not advanced.

    def test_target_differs_from_unadvanced_feedback(self):
        feedback = self.env.step([.2, .3, .4, .5, 2., 1.])
        np.testing.assert_array_equal(feedback, np.zeros(6))
        np.testing.assert_allclose(self.env.get_action_target(), [.2, .3, .4, .5, 1., 1.])
        self.assertAlmostEqual(self.env.q[-1], -.2, places=5)

    def test_delta_becomes_absolute_label(self):
        self.env.action_type = 'delta_joint_angle'; self.env.last_q = np.full(5, .2)
        self.env.step([.1, .1, .1, .1, .1, 0.])
        np.testing.assert_allclose(self.env.get_action_target(), [.3, .3, .3, .3, .3, 0.])

    def test_invalid_action_does_not_mutate_target(self):
        for action in ([0] * 7, [0] * 5, [float('nan')] * 6, [float('inf')] * 6):
            with self.assertRaises(ValueError): self.env.step(action)
            np.testing.assert_array_equal(self.env.q, np.zeros(6))

    def test_eef_mode_requires_seven_values(self):
        self.env.action_type = 'eef_pose'
        with self.assertRaises(ValueError): self.env.step([0] * 6)

    def test_collectors_use_command_target(self):
        for name in ('manual', 'batch'):
            tree = ast.parse((ROOT / f'scripts/collect/{name}_collect_data.py').read_text())
            self.assertTrue(any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == 'get_action_target' for n in ast.walk(tree)))


class SamplingClock(unittest.TestCase):
    def test_default_clock_and_mismatch_rejection(self):
        tree = ast.parse((ROOT / 'scripts/collect/batch_collect_data.py').read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'CollectConfig')
        validate = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'validate_config')
        space = {'dataclass': dataclasses.dataclass, 'PROJECT_ROOT': ROOT}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[cls, validate], type_ignores=[])), 'batch_collect_data.py', 'exec'), space)
        config = space['CollectConfig']()
        self.assertEqual(config.fps, config.control_hz)
        space['validate_config'](config)
        config.control_hz = config.fps * 2
        with self.assertRaises(ValueError): space['validate_config'](config)


if __name__ == '__main__': unittest.main()
