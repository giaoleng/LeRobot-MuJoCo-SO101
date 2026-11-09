"""固定类别词表和语言任务映射，供真实数据记录、训练及部署共用。"""
from __future__ import annotations

import json
from pathlib import Path


class TaskConditioner:
    def __init__(self, config):
        self.mode = config.get('mode', 'language')
        self.targets = dict(config.get('targets', {}))
        self.order = list(config.get('target_order', self.targets))
        self.feature_key = config.get('feature_key', 'observation.environment_state')
        if self.mode not in ('none', 'language', 'category'):
            raise ValueError('conditioning.mode must be none, language or category')
        if len(set(self.order)) != len(self.order) or set(self.order) != set(self.targets):
            raise ValueError('target_order必须无重复且完整覆盖targets')
        if not self.order or any(not isinstance(k, str) or not k for k in self.order):
            raise ValueError('targets必须包含非空字符串类别')
        for key, text in self.targets.items():
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f'目标{key}缺少语言指令')
        if self.feature_key != 'observation.environment_state':
            raise ValueError('ACT类别条件使用observation.environment_state')

    @property
    def feature_names(self):
        return [f'target.{key}' for key in self.order]

    def task(self, target_id):
        if target_id not in self.targets:
            raise ValueError(f'未知target_id={target_id!r}，可用类别：{self.order}')
        return self.targets[target_id]

    def encode(self, target_id):
        self.task(target_id)
        return [float(key == target_id) for key in self.order]

    def raw_features(self, target_id):
        return dict(zip(self.feature_names, self.encode(target_id), strict=True)) if self.mode == 'category' else {}

    def dataset_feature(self):
        return {'dtype': 'float32', 'shape': [len(self.order)], 'names': self.feature_names}

    def manifest(self):
        return {'mode': self.mode, 'target_order': self.order, 'feature_key': self.feature_key,
                'targets': self.targets}

    def save_manifest(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x', encoding='utf-8') as handle:
            json.dump(self.manifest(), handle, ensure_ascii=False, indent=2)

    def check_checkpoint(self, checkpoint_config, dataset_features, checkpoint_dir):
        if self.mode != 'category':
            return
        key = self.feature_key
        declared = checkpoint_config.get('input_features', {}).get(key)
        feature = dataset_features.get(key)
        if not declared or declared.get('shape') != [len(self.order)]:
            raise ValueError(f'ACT checkpoint必须以{key}输入训练，维数={len(self.order)}')
        if not feature or feature.get('names') != self.feature_names:
            raise ValueError('数据中的目标类别特征顺序与配置不一致')
        path = Path(checkpoint_dir) / 'task_conditioning.json'
        saved = json.loads(path.read_text(encoding='utf-8'))
        if saved != self.manifest():
            raise ValueError('checkpoint的task_conditioning.json与当前目标映射不一致')


class TargetConditionedDataset:
    """为ACT训练逐帧注入one-hot类别；外部仍负责配置ENV feature和统计量。"""
    def __init__(self, dataset, conditioner, *, task_to_target):
        self.dataset = dataset
        self.conditioner = conditioner
        self.task_to_target = dict(task_to_target)
        if conditioner.mode != 'category':
            raise ValueError('TargetConditionedDataset需要category模式')
        for target in self.task_to_target.values():
            conditioner.task(target)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        import torch
        sample = dict(self.dataset[index])
        task = sample.get('task')
        if not isinstance(task, str) or task not in self.task_to_target:
            raise ValueError('训练帧缺少明确task_to_target映射')
        sample[self.conditioner.feature_key] = torch.tensor(
            self.conditioner.encode(self.task_to_target[task]), dtype=torch.float32)
        return sample
