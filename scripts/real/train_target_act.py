#!/usr/bin/env python3
"""训练与真机部署共用类别词表的ACT，并保存processor与数据特征快照。"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.real.runtime import JOINTS, load_config, local_path
from scripts.real.task_conditioning import TaskConditioner, TargetConditionedDataset


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default=str(ROOT / 'configs/real/so101.yaml'))
    p.add_argument('--repo-id', default='local/real_so101')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--steps', type=int, default=5)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--chunk-size', type=int, default=10)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--execute', action='store_true')
    args = p.parse_args(argv)
    if min(args.steps, args.batch_size, args.chunk_size) <= 0:
        p.error('steps/batch-size/chunk-size必须为正整数')
    cfg = load_config(args.config)
    conditioner = TaskConditioner(cfg['conditioning'])
    if conditioner.mode != 'category' or cfg['policy']['type'] != 'act':
        p.error('该训练入口要求act/category配置')
    print(json.dumps({'output': str(local_path(args.output)), 'steps': args.steps,
                      'dataset_root': cfg['policy']['dataset_root'],
                      'conditioning': conditioner.manifest()}, ensure_ascii=False, indent=2))
    if not args.execute:
        return
    output = local_path(args.output)
    if output.exists():
        raise FileExistsError(f'拒绝覆盖训练输出：{output}')
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    from importlib.metadata import version
    if version('lerobot') != '0.5.1':
        raise RuntimeError('训练入口要求LeRobot==0.5.1')
    import random
    import numpy as np
    import torch
    from lerobot.configs.types import FeatureType, NormalizationMode
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.feature_utils import dataset_to_policy_features
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    dataset_root = Path(cfg['policy']['dataset_root'])
    info = json.loads((dataset_root / 'meta/info.json').read_text(encoding='utf-8'))
    for key in ('observation.state', 'action'):
        ft = info['features'].get(key, {})
        if ft.get('shape') != [6] or len(ft.get('names', [])) != 6 or set(ft['names']) != set(JOINTS):
            raise ValueError('训练数据必须为六个命名真机关节，而非仿真末端状态')
    meta = LeRobotDatasetMetadata(args.repo_id, root=dataset_root)
    features = copy.deepcopy(info['features'])
    features[conditioner.feature_key] = conditioner.dataset_feature()
    policy_features = dataset_to_policy_features(features)
    outputs = {'action': policy_features['action']}
    inputs = {key: ft for key, ft in policy_features.items()
              if key in ('observation.state', conditioner.feature_key) or ft.type is FeatureType.VISUAL}
    policy_cfg = ACTConfig(input_features=inputs, output_features=outputs,
                           chunk_size=args.chunk_size, n_action_steps=args.chunk_size,
                           pretrained_backbone_weights=None, device=cfg['policy']['device'])
    policy_cfg.normalization_mapping['ENV'] = NormalizationMode.IDENTITY
    delta = {'action': [index / meta.fps for index in policy_cfg.action_delta_indices]}
    delta.update({key: [0.] for key in policy_cfg.image_features})
    dataset = LeRobotDataset(args.repo_id, root=dataset_root, delta_timestamps=delta)
    task_to_target = {instruction: target for target, instruction in conditioner.targets.items()}
    if len(task_to_target) != len(conditioner.targets):
        raise ValueError('类别指令必须唯一，才能从训练帧task映射到target_id')
    dataset = TargetConditionedDataset(dataset, conditioner, task_to_target=task_to_target)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                         shuffle=True, num_workers=0, drop_last=True)
    if len(loader) == 0:
        raise ValueError('训练数据不足以形成一个完整batch')
    policy = ACTPolicy(policy_cfg).to(policy_cfg.device).train()
    pre, post = make_pre_post_processors(policy_cfg, dataset_stats=meta.stats)
    optimizer = policy_cfg.get_optimizer_preset().build(policy.parameters())
    step = 0
    while step < args.steps:
        for batch in loader:
            loss, _ = policy.forward(pre(batch))
            if not torch.isfinite(loss).all():
                raise ValueError('ACT loss非有限，停止训练')
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.)
            optimizer.step(); step += 1
            print(f'step={step} loss={loss.item():.6f}')
            if step >= args.steps:
                break
    output.mkdir(parents=True, exist_ok=False)
    policy.save_pretrained(output); pre.save_pretrained(output); post.save_pretrained(output)
    conditioner.save_manifest(output / 'task_conditioning.json')
    info['features'] = features
    info['action_semantics'] = cfg['policy']['action_semantics']
    (output / 'dataset_info.json').write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding='utf-8')
    (output / 'training_recipe.json').write_text(json.dumps({
        'seed': args.seed, 'steps': args.steps, 'batch_size': args.batch_size,
        'chunk_size': args.chunk_size, 'dataset_root': str(dataset_root),
        'lerobot_version': version('lerobot'), 'conditioning': conditioner.manifest(),
    }, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
