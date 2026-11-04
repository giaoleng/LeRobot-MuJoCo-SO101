# 在您的数据集上训练 Action-Chunking-Transformer (ACT)
# 在您的自定义数据集上训练 ACT 模型。在本例中，我们将 chunk_size 设置为 10。

import sys
from pathlib import Path

# 获取项目根目录（scripts/的父目录）
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.feature_utils import dataset_to_policy_features
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.configs.types import FeatureType
from lerobot.datasets.factory import resolve_delta_timestamps
import torchvision
import json

device = torch.device("cuda")

# Number of offline training steps (we'll only do offline training for this example.)
# 根据您的喜好进行调整。需要 5000 步才能得到值得评估的东西。
training_steps = 3000
log_freq = 100

# ## 策略配置和初始化
# chunk_size = 10

# 当从头开始时（即不是从预训练的策略开始），我们需要在之前指定两件事
# 创建政策：
# - 输入/输出形状：适当调整策略大小
#   - dataset stats: for normalization and denormalization of input/outputs
dataset_metadata = LeRobotDatasetMetadata("so101_pnp", root=str(PROJECT_ROOT / 'data' / 'demo_data_so101'))
features = dataset_to_policy_features(dataset_metadata.features)
output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
input_features = {key: ft for key, ft in features.items() if key not in output_features}
input_features.pop("observation.wrist_image")
# 策略使用配置类初始化，在本例中为`DiffusionConfig`。对于这个例子，
# 我们将只使用默认值，因此除了输入/输出功能之外不需要传递任何参数。
cfg = ACTConfig(input_features=input_features, output_features=output_features, chunk_size=10, n_action_steps=10)
# 这使我们能够通过动作分块来构造数据
delta_timestamps = resolve_delta_timestamps(cfg, dataset_metadata)
# 我们现在可以使用此配置和数据集统计信息实例化我们的策略。
policy = ACTPolicy(cfg, dataset_stats=dataset_metadata.stats)
policy.train()
policy.to(device)

# ## 加载数据集

from torchvision import transforms

class AddGaussianNoise(object):
    """
    Adds Gaussian noise to a tensor.
    """
    def __init__(self, mean=0., std=0.01):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        # 添加噪声：张量仍然是张量。
        noise = torch.randn(tensor.size()) * self.std + self.mean
        return tensor + noise

    def __repr__(self):
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"

# 创建一个转换管道，将 PIL 图像转换为张量，然后添加噪声。
transform = transforms.Compose([
    AddGaussianNoise(mean=0., std=0.02),
    transforms.Lambda(lambda x: x.clamp(0, 1))
])

# 然后我们可以使用这些 delta_timestamps 配置实例化数据集。
dataset = LeRobotDataset("so101_pnp", delta_timestamps=delta_timestamps, root=str(PROJECT_ROOT / 'data' / 'demo_data_so101'), image_transforms=transform)

# Then we create our optimizer and dataloader for offline training.
optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
dataloader = torch.utils.data.DataLoader(
    dataset,
    num_workers=4,
    batch_size=64,
    shuffle=True,
    pin_memory=device.type != "cpu",
    drop_last=True,
)

# ＃＃ 火车
# 训练后的检查点将保存在“(project_root)/checkpoints/act_y”文件夹中。

# 运行训练循环。
step = 0
done = False
while not done:
    for batch in dataloader:
        inp_batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        loss, _ = policy.forward(inp_batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if step % log_freq == 0:
            print(f"step: {step} loss: {loss.item():.3f}")
        step += 1
        if step >= training_steps:
            done = True
            break

# 将策略保存到磁盘。
checkpoint_dir = PROJECT_ROOT / "checkpoints" / "act_y"
policy.save_pretrained(str(checkpoint_dir))

# 将数据集元数据/统计信息与检查点一起保存，这样部署就不会
# 需要访问原始数据集目录。
def _to_jsonable(x):
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    if isinstance(x, dict):
        return {str(k): _to_jsonable(v) for k, v in x.items()}
    try:
        import numpy as _np

        if isinstance(x, _np.ndarray):
            return x.tolist()
        if isinstance(x, (_np.integer,)):
            return int(x)
        if isinstance(x, (_np.floating,)):
            return float(x)
    except Exception:
        pass
    return str(x)


deploy_metadata = {
    "dataset_repo_id": "so101_pnp",
    "features": dataset_metadata.features,
    "stats": dataset_metadata.stats,
    "act_config": {
        "chunk_size": 10,
        "n_action_steps": 10,
    },
}

(checkpoint_dir / "deploy_metadata.json").write_text(
    json.dumps(_to_jsonable(deploy_metadata), indent=2, sort_keys=True),
    encoding="utf-8",
)

# ## 测试推理
# 要评估数据集上的策略，您可以计算数据集中的真实操作之间的误差。

class EpisodeSampler(torch.utils.data.Sampler):
    def __init__(self, dataset: LeRobotDataset, episode_index: int):
        from_idx = dataset.episode_data_index["from"][episode_index].item()
        to_idx = dataset.episode_data_index["to"][episode_index].item()
        self.frame_ids = range(from_idx, to_idx)

    def __iter__(self):
        return iter(self.frame_ids)

    def __len__(self) -> int:
        return len(self.frame_ids)


policy.eval()
actions = []
gt_actions = []
images = []
episode_index = 0
episode_sampler = EpisodeSampler(dataset, episode_index)
test_dataloader = torch.utils.data.DataLoader(
    dataset,
    num_workers=4,
    batch_size=1,
    shuffle=False,
    pin_memory=device.type != "cpu",
    sampler=episode_sampler,
)
policy.reset()
for batch in test_dataloader:
    inp_batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
    action = policy.select_action(inp_batch)
    actions.append(action)
    gt_actions.append(inp_batch["action"][:,0,:])
    images.append(inp_batch["observation.image"])
actions = torch.cat(actions, dim=0)
gt_actions = torch.cat(gt_actions, dim=0)
print(f"Mean action error: {torch.mean(torch.abs(actions - gt_actions)).item():.3f}")

# 绘制动作和 gt_actions
import matplotlib.pyplot as plt
action_dim = 7

fig, axs = plt.subplots(action_dim, 1, figsize=(10, 10))

for i in range(action_dim):
    axs[i].plot(actions[:, i].cpu().detach().numpy(), label="pred")
    axs[i].plot(gt_actions[:, i].cpu().detach().numpy(), label="gt")
    axs[i].legend()
plt.show()
