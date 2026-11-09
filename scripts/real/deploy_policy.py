#!/usr/bin/env python3
"""SO-101本地checkpoint部署；默认仅打印计划，--execute才连接硬件。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.real.runtime import RealPolicyRunner, finite, load_config, local_path, plan, validate_files
from scripts.real.task_conditioning import TaskConditioner
from scripts.eval.act_rollout import JsonlRecorder


def parser(description=__doc__):
    p = argparse.ArgumentParser(description=description)
    p.add_argument('--config', default=str(ROOT / 'configs/real/so101.yaml'))
    p.add_argument('--target-id')
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--policy-type', choices=('act', 'smolvla', 'diffusion'))
    p.add_argument('--device')
    p.add_argument('--conditioning-mode', choices=('none', 'category', 'language'))
    p.add_argument('--duration-s', type=float)
    p.add_argument('--output', type=Path)
    group = p.add_mutually_exclusive_group()
    group.add_argument('--execute', action='store_true', help='启用模型和真实机器人')
    group.add_argument('--dry-run', action='store_true', help='仅打印计划，不加载模型或硬件')
    p.add_argument('--check-files', action='store_true', help='检查本地权重、schema和processor文件')
    p.add_argument('--export-conditioning', type=Path, help='写出训练/部署共用的task_conditioning.json，不覆盖已有文件')
    return p


def prepare(args):
    cfg = load_config(args.config)
    if args.checkpoint:
        cfg['policy']['checkpoint'] = str(local_path(args.checkpoint))
    if args.policy_type:
        cfg['policy']['type'] = args.policy_type
    if args.device:
        cfg['policy']['device'] = args.device
    if args.conditioning_mode:
        cfg['conditioning']['mode'] = args.conditioning_mode
    if args.duration_s is not None:
        if finite(args.duration_s, 'duration_s') <= 0:
            raise ValueError('duration_s必须为正数')
        cfg['control']['duration_s'] = args.duration_s
    condition = TaskConditioner(cfg['conditioning'])
    target_id = args.target_id or condition.order[0]
    condition.task(target_id)
    if args.export_conditioning:
        condition.save_manifest(local_path(args.export_conditioning))
    if args.check_files:
        validate_files(cfg)
    print(json.dumps(plan(cfg, target_id), ensure_ascii=False, indent=2))
    return cfg, target_id


def metadata(cfg):
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    files = ['scripts/real/deploy_policy.py', 'scripts/real/evaluate_policy.py',
             'scripts/real/runtime.py', 'scripts/real/task_conditioning.py']
    return {'policy': cfg['policy']['type'], 'checkpoint': cfg['policy']['checkpoint'],
            'robot_id': cfg['robot']['id'], 'control_hz': cfg['control']['fps'],
            'action_semantics': cfg['policy']['action_semantics'], 'git_commit': commit,
            'working_tree_dirty': dirty,
            'source_sha256': {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in files},
            'conditioning': TaskConditioner(cfg['conditioning']).manifest()}


def main(argv=None):
    args = parser().parse_args(argv)
    cfg, target = prepare(args)
    if not args.execute:
        return
    output = local_path(args.output) if args.output else Path(cfg['logging']['output'])
    recorder = JsonlRecorder(output, metadata(cfg))
    try:
        runner = RealPolicyRunner(cfg)
        result = runner.run_episode(target)
        recorder({'kind': 'deployment', **result})
    except BaseException as exc:
        recorder({'kind': 'deployment', 'target_id': target, 'reason': 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'error',
                  'error_type': type(exc).__name__})
        raise
    finally:
        recorder.close()


if __name__ == '__main__':
    main()
