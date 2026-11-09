"""真机接口的离线回归用例；不导入Torch/LeRobot，不接触串口或相机。"""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from scripts.real.runtime import ActionGuard, JOINTS, ROOT, load_config, require_hardware_config, validate_files
from scripts.real.task_conditioning import TaskConditioner
from scripts.real.evaluate_policy import read_outcome


class RealInterfaces(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config(ROOT / 'configs/real/so101.yaml')
        self.control = copy.deepcopy(self.cfg['control'])
        self.control['joint_limits'] = {key: ([-90, 90] if key != 'gripper.pos' else [0, 100]) for key in JOINTS}

    def test_target_vocabulary_and_feature_order(self):
        c = TaskConditioner(self.cfg['conditioning'])
        self.assertEqual(c.encode('red_car'), [0., 1., 0., 0.])
        self.assertEqual(c.raw_features('red_car')['target.red_car'], 1.)
        with self.assertRaises(ValueError): c.encode('unknown')

    def test_missing_hardware_limits_block_execution(self):
        missing = copy.deepcopy(self.control)
        missing['joint_limits']['shoulder_pan.pos'] = None
        with self.assertRaises(ValueError): ActionGuard(missing)

    def test_reference_profile_does_not_enable_hardware(self):
        with self.assertRaises(ValueError): require_hardware_config(self.cfg)
        confirmed = copy.deepcopy(self.cfg)
        confirmed['robot']['hardware_confirmed'] = True
        require_hardware_config(confirmed)

    def test_action_limits_and_velocity(self):
        guard = ActionGuard(self.control)
        observation = {key: (50. if key == 'gripper.pos' else 0.) for key in JOINTS}
        command = guard.filter({key: 80. for key in JOINTS}, observation, .05)
        self.assertAlmostEqual(command['shoulder_pan.pos'], 1.)
        self.assertAlmostEqual(command['gripper.pos'], 51.5)
        guard.accept(command)
        with self.assertRaises(ValueError):
            guard.filter({key: float('nan') for key in JOINTS}, observation, .05)

    def test_wrong_joint_names_rejected(self):
        guard = ActionGuard(self.control)
        with self.assertRaises(ValueError): guard.filter({'action': 1.}, {}, .05)

    def test_outcome_requires_consistent_boolean_flags(self):
        answers = iter(['{"success":true,"grasped":false}', '{"success":false,"grasped":false}'])
        self.assertFalse(read_outcome(lambda prompt: next(answers))['success'])

    def test_checkpoint_category_contract(self):
        c = TaskConditioner(self.cfg['conditioning'])
        feature = {c.feature_key: c.dataset_feature()}
        model = {'input_features': {c.feature_key: {'shape': [4]}}}
        with tempfile.TemporaryDirectory() as tmp:
            c.save_manifest(Path(tmp) / 'task_conditioning.json')
            c.check_checkpoint(model, feature, tmp)
            with self.assertRaises(FileExistsError): c.save_manifest(Path(tmp) / 'task_conditioning.json')
            bad = copy.deepcopy(model); bad['input_features'][c.feature_key]['shape'] = [3]
            with self.assertRaises(ValueError): c.check_checkpoint(bad, feature, tmp)

    def test_packaged_training_schema_matches_deploy_contract(self):
        cfg = copy.deepcopy(self.cfg)
        c = TaskConditioner(cfg['conditioning'])
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp); cfg['policy']['checkpoint'] = tmp
            cfg['policy']['dataset_root'] = str(checkpoint / 'unused_dataset')
            features = {key: {'shape': [6], 'names': list(JOINTS), 'dtype': 'float32'}
                        for key in ('action', 'observation.state')}
            features[c.feature_key] = c.dataset_feature()
            model = {'type': 'act', 'input_features': {'observation.state': {'shape': [6]},
                      c.feature_key: {'shape': [4]}}, 'output_features': {'action': {'shape': [6]}}}
            for file in ('model.safetensors', 'policy_preprocessor.json', 'policy_postprocessor.json'):
                (checkpoint / file).touch()
            (checkpoint / 'config.json').write_text(json.dumps(model))
            (checkpoint / 'dataset_info.json').write_text(json.dumps({'features': features}))
            c.save_manifest(checkpoint / 'task_conditioning.json')
            validate_files(cfg)
            features['action']['names'] = ['bad'] * 6
            (checkpoint / 'dataset_info.json').write_text(json.dumps({'features': features}))
            with self.assertRaises(ValueError): validate_files(cfg)


if __name__ == '__main__': unittest.main()
