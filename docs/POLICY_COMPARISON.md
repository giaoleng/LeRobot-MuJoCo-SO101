# SO-101 MuJoCo：ACT / Diffusion Policy / SmolVLA 统一仿真评测

## 当前状态

- 已有 MuJoCo 任务：SO-101 抓取蓝色方块并放入容器。
- 已有脚本专家及 `media/simulation_demo.mp4`；它不是训练策略的演示。
- 本增量包提供 DP 和 SmolVLA 的 **LeRobot 训练启动入口**，以及 rollout 日志统计工具。
- 尚未接入现有 `mujoco_env/y_env.py` 的策略推理适配器；统一 `eval_mujoco.py` 需要依据现有 env 的 `reset/step`、camera、action 和 success 接口进行开发。

## 数据要求

三模型使用相同 LeRobotDataset v3 数据源，至少包含 `action`、`observation.state`、图像，以及 SmolVLA 所需的任务文本与 camera rename map。先核对 `meta/info.json` 的 `features`、图像键、fps、状态与动作维度，训练/仿真关节单位须一致。

**模型原理不同，训练规则不强行相同**：ACT/DP 从头训练；SmolVLA 微调官方预训练权重。需报告此差异，不能把全部差距归因于架构。

## 远程训练：私有数据示例

数据、依赖和预训练权重需要提前放在 BitaHub。BitaHub 的出网受限，不要假设训练中可以下载权重。

```bash
export DATASET_ROOT=/private/datasets/so101_sim
export DATASET_REPO_ID=local/so101_sim

# 只打印命令；不触发 GPU、不读取数据
bash scripts/train/train_diffusion.sh --dry-run

# 远程训练（需要正确安装 LeRobot、CUDA，且私有数据就绪）
bash scripts/train/train_diffusion.sh

export SMOLVLA_CHECKPOINT=/private/weights/smolvla_base
# 仅在数据集相机键与预训练模型期望键不一致时启用；先阅读 meta/info.json：
# export SMOLVLA_RENAME_MAP='{"observation.images.top":"observation.images.camera1","observation.images.wrist":"observation.images.camera2"}'
bash scripts/train/finetune_smolvla.sh --dry-run
bash scripts/train/finetune_smolvla.sh
```

`TRAIN_STEPS`、`BATCH_SIZE`、`OUTPUT_DIR` 可通过环境变量覆盖；训练脚本其余参数可以追加在命令末尾。默认值只是启动示例，不是经过调参的最终配置。

### 隐私

训练入口默认包含 `--policy.push_to_hub=false`，并设置 `WANDB_MODE=disabled`；脚本不会自动推送数据与模型。但其他项目脚本、依赖与启动环境仍需单独检查，不能据此保证整个系统完全无外网通信。不要公开数据、checkpoint、rollout 日志和真实机器人地址。

## 评测协议

统一设置：相同测试 Seeds、初始物体位置集合、Episode 最大仿真时长、控制频率、success checker、观测映射、GPU 型号。至少 50 次 Episode，最好 100 次；区分训练分布内 ID 和未见位置 OOD。记录原始成功数而非仅汇报百分比。

建议每个 Episode 输出一行 JSONL：

```json
{"policy":"act","split":"id","seed":42,"success":true,"started":true,"grasped":true,"placed":true,"completion_time_s":13.2,"inference_ms":[22.4,19.0]}
```

上面的数据是格式示例，不代表实际结果。三个模型分别以 `policy=act`、`policy=diffusion`、`policy=smolvla` 记录；对无信息的阶段字段可以省略，不要随意填 `false`。

```bash
python scripts/eval/summarize_rollouts.py outputs/eval/rollouts.jsonl \
  --csv outputs/eval/summary.csv
```

输出：总成功率及 Wilson 95% 区间、成功 Episode 的平均完成时间、动作推理延迟 P50/P95，以及在所有 Episode 中的启动/抓取/放置失败比例（只有日志完整时才计算）。

注意：汇总工具 **不是** Mujoco Rollout，不能替代 `eval_mujoco.py`。此外，统计时 `inference_ms` 是每次推理调用耗时，不是完整动作执行延迟；不同 chunk_size 需要同时报告。

## 正式对比表（待测）

| 模型 | 成功次数/测试数 | ID成功率 | OOD成功率 | 成功平均时长 | 推理P50/P95 | 启动失败率 |
|---|---|---|---|---|---|---|
| ACT | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| Diffusion Policy | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| SmolVLA | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |

## 参考

- LeRobot training: https://huggingface.co/docs/lerobot/main/cheat-sheet
- SmolVLA: https://huggingface.co/docs/lerobot/main/smolvla
- LeRobot source: https://github.com/huggingface/lerobot
