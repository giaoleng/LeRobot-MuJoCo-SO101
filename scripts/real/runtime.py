"""LeRobot 0.5.1适配层；串口、相机与模型仅在显式执行时创建。"""
from __future__ import annotations

import copy
import json
import math
import os
from numbers import Real
from pathlib import Path
import time

from scripts.real.task_conditioning import TaskConditioner

ROOT = Path(__file__).resolve().parents[2]
JOINTS = tuple(f'{name}.pos' for name in
               ('shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper'))


def local_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def finite(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f'{name}必须是有限数值')
    return float(value)


def load_config(path):
    import yaml
    cfg = yaml.safe_load(local_path(path).read_text(encoding='utf-8'))
    if not isinstance(cfg, dict):
        raise ValueError('配置必须是YAML mapping')
    for section in ('robot', 'policy', 'conditioning', 'control', 'logging', 'evaluation'):
        if not isinstance(cfg.get(section), dict):
            raise ValueError(f'缺少配置节{section}')
    cfg = copy.deepcopy(cfg)
    robot, policy, control = cfg['robot'], cfg['policy'], cfg['control']
    if robot.get('use_degrees') is not True:
        raise ValueError('该入口要求真机臂关节degrees，use_degrees必须为true')
    if policy.get('action_semantics') != 'so101_degrees_gripper_percent':
        raise ValueError('真机动作语义必须为so101_degrees_gripper_percent')
    if policy.get('type') not in ('act', 'smolvla', 'diffusion'):
        raise ValueError('policy.type必须为act、smolvla或diffusion')
    if policy.get('processors') not in ('saved', 'dataset'):
        raise ValueError('policy.processors必须为saved或dataset')
    if not robot.get('port') or not robot.get('id') or not isinstance(robot.get('cameras'), dict) or not robot['cameras']:
        raise ValueError('robot.port/id/cameras不可为空')
    if Path(str(robot['id'])).name != robot['id'] or robot['id'] in ('.', '..'):
        raise ValueError('robot.id必须是普通文件名标识')
    for field in ('fps', 'duration_s', 'inference_timeout_s', 'max_observation_age_s'):
        if finite(control.get(field), field) <= 0:
            raise ValueError(f'{field}必须为正数')
    if type(control.get('max_steps')) is not int or control['max_steps'] <= 0:
        raise ValueError('max_steps必须为正整数')
    for mapping in ('joint_limits', 'max_step', 'max_speed'):
        if set(control.get(mapping, {})) != set(JOINTS):
            raise ValueError(f'{mapping}必须精确覆盖六个.pos关节')
    for key in JOINTS:
        for mapping in ('max_step', 'max_speed'):
            if finite(control[mapping][key], f'{mapping}.{key}') <= 0:
                raise ValueError(f'{mapping}.{key}必须为正数')
        bounds = control['joint_limits'][key]
        if bounds is not None:
            if not isinstance(bounds, list) or len(bounds) != 2 or finite(bounds[0], key) >= finite(bounds[1], key):
                raise ValueError(f'{key}限位必须为[min,max]')
            if key == 'gripper.pos' and not 0 <= bounds[0] < bounds[1] <= 100:
                raise ValueError('真机夹爪限位必须在0..100内')
    for key in ('checkpoint', 'dataset_root'):
        policy[key] = str(local_path(policy[key]))
    robot['calibration_dir'] = str(local_path(robot['calibration_dir']))
    for section, key in [('control', 'stop_file'), ('logging', 'output'), ('evaluation', 'output')]:
        cfg[section][key] = str(local_path(cfg[section][key]))
    if type(cfg['evaluation'].get('episodes')) is not int or cfg['evaluation']['episodes'] <= 0:
        raise ValueError('evaluation.episodes必须为正整数')
    TaskConditioner(cfg['conditioning'])
    return cfg


def plan(cfg, target_id):
    conditioner = TaskConditioner(cfg['conditioning'])
    return {'policy': cfg['policy']['type'], 'checkpoint': cfg['policy']['checkpoint'],
            'dataset_root': cfg['policy']['dataset_root'], 'robot_port': cfg['robot']['port'],
            'target_id': target_id, 'task': conditioner.task(target_id),
            'conditioning': conditioner.mode, 'fps': cfg['control']['fps'],
            'action_semantics': cfg['policy']['action_semantics'],
            'hardware_confirmed': cfg['robot'].get('hardware_confirmed', False),
            'joint_limits_to_fill': [k for k, v in cfg['control']['joint_limits'].items() if v is None]}


def validate_files(cfg):
    policy = cfg['policy']; checkpoint = Path(policy['checkpoint'])
    model_cfg = json.loads((checkpoint / 'config.json').read_text(encoding='utf-8'))
    if model_cfg.get('type') != policy['type']:
        raise ValueError('配置的policy.type与checkpoint config.json不一致')
    if not (checkpoint / 'model.safetensors').is_file():
        raise FileNotFoundError(f'缺少{checkpoint / "model.safetensors"}')
    meta = Path(policy['dataset_root']) / 'meta'
    info_path = checkpoint / 'dataset_info.json'
    if not info_path.is_file():
        info_path = meta / 'info.json'
    info = json.loads(info_path.read_text(encoding='utf-8'))
    if info.get('action_semantics', policy['action_semantics']) != policy['action_semantics']:
        raise ValueError('训练数据动作语义与真机配置不一致')
    features = info['features']
    for name in ('action', 'observation.state'):
        ft = features.get(name, {})
        if ft.get('shape') != [6] or len(ft.get('names', [])) != 6 or set(ft['names']) != set(JOINTS):
            raise ValueError(f'{name}必须保存六个命名真机关节；不能直接套用仿真末端state')
    if model_cfg.get('output_features', {}).get('action', {}).get('shape') != [6]:
        raise ValueError('checkpoint动作维度必须为6')
    state = model_cfg.get('input_features', {}).get('observation.state')
    if state is None or state.get('shape') != [6]:
        raise ValueError('checkpoint必须包含六维observation.state')
    for key, value in model_cfg.get('input_features', {}).items():
        if key not in features:
            raise ValueError(f'数据feature缺少checkpoint输入{key}')
        if value.get('type') == 'VISUAL':
            camera = key.removeprefix('observation.images.')
            if camera not in cfg['robot']['cameras']:
                raise ValueError(f'缺少相机映射{camera}')
            cam = cfg['robot']['cameras'][camera]
            if features[key].get('shape') != [cam['height'], cam['width'], 3]:
                raise ValueError(f'{camera}的配置尺寸与训练数据不一致')
    condition = TaskConditioner(cfg['conditioning'])
    if policy['type'] == 'act' and condition.mode == 'language':
        raise ValueError('ACT需要category或none模式，不会自动读取语言任务作为类别条件')
    if policy['type'] == 'smolvla' and condition.mode != 'language':
        raise ValueError('SmolVLA使用language模式')
    if condition.mode == 'category' and policy['type'] != 'act':
        raise ValueError('category模式在本入口只用于ACT')
    if 'observation.environment_state' in model_cfg.get('input_features', {}) and condition.mode != 'category':
        raise ValueError('checkpoint需要类别条件，请使用匹配的category模式')
    condition.check_checkpoint(model_cfg, features, checkpoint)
    if policy['processors'] == 'saved':
        for name in ('policy_preprocessor.json', 'policy_postprocessor.json'):
            if not (checkpoint / name).is_file():
                raise FileNotFoundError(f'缺少{checkpoint / name}')
    elif not (meta / 'stats.json').is_file():
        raise FileNotFoundError(f'缺少{meta / "stats.json"}')
    return model_cfg, info


class ActionGuard:
    def __init__(self, control):
        self.control = control
        self.previous = None
        if any(control['joint_limits'][k] is None for k in JOINTS):
            raise ValueError('执行前必须填齐这台SO-101的joint_limits')

    def reset(self):
        self.previous = None

    def filter(self, action, observation, dt):
        if set(action) != set(JOINTS):
            raise ValueError('动作键必须精确覆盖六个真机关节')
        dt = min(max(finite(dt, 'dt'), 0.), 2. / self.control['fps'])
        output = {}
        for key in JOINTS:
            target = finite(action[key], key)
            current = finite(observation[key], f'observation.{key}')
            low, high = self.control['joint_limits'][key]
            if not low <= current <= high:
                raise ValueError(f'{key}当前读数超出配置限位，先检查标定/限位')
            delta = min(self.control['max_step'][key], self.control['max_speed'][key] * dt)
            lower, upper = max(low, current - delta), min(high, current + delta)
            if self.previous is not None:
                lower = max(lower, self.previous[key] - delta)
                upper = min(upper, self.previous[key] + delta)
            if lower > upper:
                raise ValueError(f'{key}反馈与前一命令差异过大，停止本轮动作')
            output[key] = min(max(target, lower), upper)
        return output

    def accept(self, sent):
        if set(sent) != set(JOINTS):
            raise ValueError('驱动返回的实际动作键不完整')
        self.previous = {key: finite(sent[key], key) for key in JOINTS}


def require_hardware_config(cfg):
    if cfg['robot'].get('hardware_confirmed') is not True:
        raise ValueError('当前是SO-101参考配置；核对实际串口、相机、标定和限位后，才能设置robot.hardware_confirmed=true执行')


class RealPolicyRunner:
    def __init__(self, cfg):
        require_hardware_config(cfg)
        self.cfg = cfg
        declared, self.info = validate_files(cfg)
        self.frame_features = {key: self.info['features'][key] for key in declared['input_features']}
        self.condition = TaskConditioner(cfg['conditioning'])
        self.guard = ActionGuard(cfg['control'])
        calibration = Path(cfg['robot']['calibration_dir']) / f"{cfg['robot']['id']}.json"
        if not calibration.is_file():
            raise FileNotFoundError(f'缺少已有标定文件{calibration}；本入口不自动执行标定')
        from importlib.metadata import version
        if version('lerobot') != '0.5.1':
            raise RuntimeError('本适配层使用LeRobot==0.5.1接口，请在该环境使用')
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import get_policy_class, make_pre_post_processors
        from lerobot.policies.utils import build_inference_frame, make_robot_action
        from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
        self.torch = torch
        self.device = torch.device(cfg['policy']['device'])
        model_path = cfg['policy']['checkpoint']
        model_cfg = PreTrainedConfig.from_pretrained(model_path, local_files_only=True)
        model_cfg.device = str(self.device)
        # 完整策略checkpoint会恢复这些权重，避免构造时再次请求基础权重。
        if hasattr(model_cfg, 'pretrained_backbone_weights'):
            model_cfg.pretrained_backbone_weights = None
        if hasattr(model_cfg, 'load_vlm_weights'):
            model_cfg.load_vlm_weights = False
        self.policy = get_policy_class(cfg['policy']['type']).from_pretrained(
            model_path, config=model_cfg, local_files_only=True).to(self.device).eval()
        if cfg['policy']['processors'] == 'saved':
            self.pre, self.post = make_pre_post_processors(
                self.policy.config, model_path,
                preprocessor_overrides={'device_processor': {'device': str(self.device)}})
        else:
            stats_path = Path(cfg['policy']['dataset_root']) / 'meta/stats.json'
            raw_stats = json.loads(stats_path.read_text(encoding='utf-8'))
            stats = {name: {key: torch.tensor(value) for key, value in values.items()}
                     for name, values in raw_stats.items()}
            self.pre, self.post = make_pre_post_processors(self.policy.config, dataset_stats=stats)
        cameras = {name: OpenCVCameraConfig(**camera) for name, camera in cfg['robot']['cameras'].items()}
        robot_cfg = SO101FollowerConfig(
            port=cfg['robot']['port'], id=cfg['robot']['id'], cameras=cameras,
            calibration_dir=Path(cfg['robot']['calibration_dir']), use_degrees=True,
            max_relative_target={k.removesuffix('.pos'): v for k, v in cfg['control']['max_step'].items()},
            disable_torque_on_disconnect=cfg['robot'].get('disable_torque_on_disconnect', True))
        self.robot = SO101Follower(robot_cfg)
        self.build_frame, self.make_action = build_inference_frame, make_robot_action

    def predict(self, raw, target_id):
        observation = {**raw, **self.condition.raw_features(target_id)}
        frame = self.build_frame(observation=observation, ds_features=self.frame_features,
                                 device=self.device, task=self.condition.task(target_id), robot_type='so101_follower')
        if self.device.type == 'cuda':
            self.torch.cuda.synchronize(self.device)
        start = time.monotonic()
        with self.torch.inference_mode():
            action = self.post(self.policy.select_action(self.pre(frame)))
        result = self.make_action(action, self.info['features'])
        return result, (time.monotonic() - start) * 1000.

    def run_episode(self, target_id):
        self.condition.task(target_id)
        stop_file = Path(self.cfg['control']['stop_file'])
        if stop_file.exists():
            raise RuntimeError(f'停止文件已存在：{stop_file}')
        self.guard.reset(); self.policy.reset(); self.pre.reset(); self.post.reset()
        start = time.monotonic(); latencies = []; steps = clipped = 0; reason = 'step_limit'
        try:
            self.robot.connect(calibrate=False)
            if not self.robot.is_calibrated:
                raise RuntimeError('电机标定与本地文件不匹配，先完成独立标定')
            start = time.monotonic(); previous_send = start - 1. / self.cfg['control']['fps']
            for _ in range(self.cfg['control']['max_steps']):
                tick = time.monotonic()
                if stop_file.exists():
                    reason = 'stop_file'; break
                if tick - start >= self.cfg['control']['duration_s']:
                    reason = 'duration_limit'; break
                raw = self.robot.get_observation()
                target, latency = self.predict(raw, target_id)
                now = time.monotonic()
                if latency / 1000 > self.cfg['control']['inference_timeout_s']:
                    raise TimeoutError('推理超过配置时限，本步不下发动作')
                if now - tick > self.cfg['control']['max_observation_age_s']:
                    raise TimeoutError('观测已超过配置有效期，本步不下发动作')
                if stop_file.exists():
                    reason = 'stop_file'; break
                if now - start >= self.cfg['control']['duration_s']:
                    reason = 'duration_limit'; break
                command = self.guard.filter(target, raw, now - previous_send)
                clipped += int(any(abs(command[k] - target[k]) > 1e-6 for k in JOINTS))
                sent = self.robot.send_action(command)
                self.guard.accept(sent)
                previous_send = time.monotonic(); steps += 1; latencies.append(latency)
                remaining = 1. / self.cfg['control']['fps'] - (time.monotonic() - tick)
                if remaining > 0:
                    time.sleep(remaining)
            return {'target_id': target_id, 'task': self.condition.task(target_id),
                    'reason': reason, 'control_steps': steps, 'clipped_steps': clipped,
                    'completion_time_s': time.monotonic() - start, 'inference_ms': latencies}
        finally:
            # 各资源独立释放，单个相机失败不能跳过串口清理。
            import logging
            try:
                if self.robot.bus.is_connected:
                    self.robot.bus.disconnect(self.cfg['robot'].get('disable_torque_on_disconnect', True))
            except Exception:
                logging.exception('释放SO-101串口失败')
            for camera in self.robot.cameras.values():
                try:
                    if camera.is_connected:
                        camera.disconnect()
                except Exception:
                    logging.exception('释放相机失败')
