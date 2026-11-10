# SO-101 MuJoCo：ACT / Diffusion Policy / SmolVLA 统一仿真评测

## 当前状态（2026-11-09）

- 项目维护者确认：SO-101 MuJoCo 的 ACT、Diffusion Policy 和 SmolVLA 训练与仿真评测已完成；下表与 README 10.6 节保持一致。
- MuJoCo 任务是多颜色方块干扰下抓取蓝色方块并放入容器。
- 仓库保留脚本专家与 `media/simulation_demo.mp4`；该演示属于专家控制器，不能当作三种学习策略的评测视频。
- 早期工程文档中的“待测”与“尚未接入统一 eval_mujoco.py”属于当时的代码完善阶段；**当前仓库没有公布完整的原始训练与评测产物**，因此本次文档同步未独立重跑或核验实验指标。

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

## 统一策略推理与仿真评测

现已增加 `scripts/eval/eval_mujoco.py`，支持 `--policy act|diffusion|smolvla` 三种策略使用同一 `SimpleEnv` 和 `run_evaluation` Episode 调度，输出含实际策略名的逐 Episode JSONL；`scripts/eval/summarize_rollouts.py` 继续负责指标汇总。

```bash
python scripts/eval/eval_mujoco.py --policy act \
  --checkpoint /private/weights/act/pretrained_model \
  --dataset-root /private/datasets/so101_sim \
  --dry-run

# 实际评测须同时指定动作语义、相机特征与独立输出路径。
# 参考根 README 10.5.1 节中的完整命令，不应照搬占位路径。
python -m unittest discover -s tests -p test_unified_eval.py -v
```

该统一入口使用 LeRobot 0.5.1 策略工厂和本地处理器；强制检验 checkpoint policy type、图像与状态输入、六维动作输出及动作单位。如果权重需要额外条件输入，须扩展适配器而不是静默填充假数据。当前自动测试为无 GPU 契约测试；真实模型推理/仿真仍须用私有 checkpoint 在远程环境验收。

历史 README 指标是维护者确认的已完成实验结果，**不是本次新增脚本的自动回放结果**，本次没有重新运行或修改已有百分比。

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

## 实验结果汇总

以下指标与 [README 第 10.6 节](../README.md#106-仿真实验结果) 相同。项目维护者确认实验已完成；训练日志、Checkpoint、逐 Episode 评测记录和数据集保存在私有存储，未上传 GitHub。

| 指标 | ACT | Diffusion Policy | SmolVLA |
|---|---:|---:|---:|
| 任务成功率 | **82%** | 69% | 77% |
| OOD 成功率 | 62% | 49% | **63%** |
| 完成时间（相对脚本专家） | **1.42×** | 1.58× | 1.67× |
| 启动失败率 | **4%** | 13% | 14% |
| 抓取失败率 | **17%** | 22% | 21% |

完整的测试 Episode 数量、成功次数和推理延迟 P50/P95 未在现有 README 中披露，因此不填写推测数值；这不代表这些实验尚未完成。本文档仅统一项目状态与已有结果，没有重新执行训练或评测。

## 参考

- LeRobot training: https://huggingface.co/docs/lerobot/main/cheat-sheet
- SmolVLA: https://huggingface.co/docs/lerobot/main/smolvla
- LeRobot source: https://github.com/huggingface/lerobot
