# LeRobot-MuJoCo-SO101

**SO-101 Robotic Imitation Learning: ACT, Diffusion Policy & SmolVLA**

基于 [LeRobot](https://github.com/huggingface/lerobot) 与 [MuJoCo](https://mujoco.org/) 构建的 SO-101 机器人模仿学习实验平台，覆盖专家示范采集、数据集处理、策略训练、仿真闭环评测及真实机器人操作。

项目包含两个实验场景：

- **MuJoCo Simulation**：在 SO-101 仿真环境中比较 ACT、Diffusion Policy 和 SmolVLA，完成蓝色方块抓取放置任务。
- **Real Robot Manipulation**：在真实 SO-101 上比较 ACT 和 SmolVLA，根据语言指令从多个玩具中选择目标并完成抓取放置。

通过统一评测协议，研究不同策略的任务成功率、执行效率、位置泛化能力以及语言条件下的目标选择能力。

---

## 目录

1. [项目介绍](#overview)
2. [系统架构](#architecture)
3. [仿真演示](#demo)
4. [安装与环境配置](#installation)
5. [项目结构](#project-structure)
6. [SO-101 机械臂与 MuJoCo 环境](#environment)
7. [专家示范与数据采集](#data-collection)
8. [LeRobotDataset 数据处理](#dataset)
9. [数据集回放](#playback)
10. [模型训练](#training)
11. [MuJoCo 闭环评测](#simulation-evaluation)
12. [SO-101 真机部署](#real-deployment)
13. [实验设计与性能对比](#experiments)
14. [Hugging Face 数据管理](#huggingface)
15. [实验复现与注意事项](#reproducibility)
16. [致谢](#acknowledgements)
17. [许可证](#license)

---

<a id="overview"></a>
## 1. 项目介绍

本项目面向低成本机器人操作任务，构建基于视觉观测、机器人状态及任务条件的模仿学习系统。
### 1.1 实验任务

本项目使用 SO-101 机械臂，在 MuJoCo 仿真环境与真实机器人环境中研究多物体场景下的模仿学习与机器人操作。

#### MuJoCo Simulation

仿真环境包含绿色、黄色、红色和蓝色方块，以及一个目标收纳盒。

基础任务要求 SO-101 机械臂从多个方块中抓取指定的蓝色方块，并将其放入收纳盒，其他方块作为干扰物。

使用 ACT、Diffusion Policy 和 SmolVLA 进行策略训练与统一闭环评测。

主要关注：

- 多物体干扰下的任务成功率。
- 目标方块初始位置变化下的泛化能力。
- 抓取、搬运及放置阶段的失败情况。
- 不同策略的推理延迟和执行效率。

#### Real Robot Manipulation

真实环境使用 SO-101 机械臂，桌面同时摆放多个不同的玩具。

每次 Episode 通过语言指令指定需要抓取的目标玩具。目标玩具随任务变化，其余玩具作为干扰物。

部署目标条件 ACT 与 SmolVLA，比较两种策略在多目标操作场景中的任务成功率、目标选择能力和泛化表现。

如果机器人抓取错误目标，即使完成放置，也判定为任务失败。

### 1.2 模型介绍

| 模型 | 方法 | 训练方式 | 实验环境 |
|---|---|---|---|
| ACT | Transformer + Action Chunking | 监督模仿学习 | 仿真 + 真机 |
| Diffusion Policy | Diffusion-based Action Generation | 扩散式模仿学习 | 仿真 |
| SmolVLA | Vision-Language-Action | 预训练模型微调 | 仿真 + 真机 |

仿真实验比较三种策略在相同任务下的操作性能。

真机实验进一步比较目标条件策略在多物体干扰场景中的执行效果。

其中，ACT 使用明确的目标条件输入；SmolVLA 使用自然语言指令。

### 1.3 研究问题

项目主要关注以下问题。

**Simulation Policy Benchmark**

ACT、Diffusion Policy 和 SmolVLA 在相同 SO-101 MuJoCo 任务中的性能有何差异？

**Position Generalization**

当目标物体初始位置发生变化时，策略是否能够保持稳定的操作成功率？

**Language-Conditioned Object Selection**

真实场景中存在多个候选物体时，SmolVLA 能否根据自然语言指令正确选择目标？

**Real-world Policy Comparison**

在目标不断变化的多物体操作任务中，目标条件 ACT 与 SmolVLA 的成功率、执行效率和失败类型有何差异？

**Instruction Generalization**

当语言指令的表达方式变化时，SmolVLA 能否保持目标选择与操作能力？

---

<a id="architecture"></a>
## 2. 系统架构

### 2.1 MuJoCo 仿真训练与评测

```text
                 SO-101 MuJoCo
                       │
                       ▼
              Cube Pick-and-Place
                       │
                       ▼
              Expert Demonstrations
                       │
                       ▼
                  LeRobotDataset
                       │
             ┌─────────┼─────────┐
             │         │         │
             ▼         ▼         ▼
            ACT       DP       SmolVLA
             │         │         │
             └─────────┼─────────┘
                       ▼
               Closed-loop Rollout
                       │
                       ▼
                Policy Benchmark
```

### 2.2 真实机器人操作

```text
                   SO-101 Real Robot
                          │
                          ▼
                  Multiple Toy Objects
                          │
                          ▼
                    Task Condition
                          │
                ┌─────────┴─────────┐
                │                   │
                ▼                   ▼
          Target Category      Language Instruction
                │                   │
                ▼                   ▼
      Goal-Conditioned ACT        SmolVLA
                │                   │
                └─────────┬─────────┘
                          ▼
                  Action Prediction
                          │
                          ▼
                   SO-101 Controller
                          │
                          ▼
                 Target Pick-and-Place
                          │
                          ▼
                 Real-world Evaluation
```

### 2.3 核心模块

| 模块 | 功能 |
|---|---|
| MuJoCo Environment | SO-101 机器人与物理场景仿真 |
| Expert Controller | 生成抓取放置专家轨迹 |
| Data Collection | 采集视觉、状态、动作和任务条件 |
| LeRobotDataset | 统一存储机器人操作数据 |
| Policy Training | ACT、DP 和 SmolVLA 训练 |
| Simulation Evaluation | MuJoCo 闭环任务评测 |
| Real Robot Interface | SO-101 状态读取和动作控制 |
| Task Conditioning | 目标类别及语言任务条件 |
| Performance Analysis | 成功率、目标选择和失败类型分析 |

---

<a id="demo"></a>
## 3. 仿真演示

### SO-101 MuJoCo Pick-and-Place

[▶ 查看 SO-101 MuJoCo 仿真演示](media/simulation_demo.mp4)

仿真场景包含 SO-101 机械臂、桌面、蓝色方块、收纳盒和 RGB 相机。

演示使用 Mink 逆运动学与有限状态机控制器完成抓取放置任务。

该脚本专家同时用于生成模仿学习数据。

---

<a id="installation"></a>
## 4. 安装与环境配置

### 4.1 环境依赖

| 组件 | 用途 |
|---|---|
| Python 3.12 | Python 运行环境 |
| MuJoCo | 物理仿真 |
| LeRobot | 数据集与模型训练 |
| PyTorch | 深度学习 |
| Mink | 逆运动学 |
| OpenCV | 图像处理 |
| FFmpeg | 视频编码 |
| Transformers | SmolVLA 模型组件 |

数据处理使用 LeRobotDataset v3.0。

模型训练和部署时应使用与 Checkpoint 兼容的 LeRobot 版本。

### 4.2 Docker

构建运行环境：

```bash
docker build -t lerobot_mujoco_sim:runtime -f Dockerfile.runtime .

docker build -t lerobot_mujoco_sim:train -f Dockerfile .
```

启动：

```bash
docker run --rm -it \
  --gpus all \
  -v "$PWD:/workspace" \
  lerobot_mujoco_sim:runtime
```

进入项目：

```bash
cd /workspace
```

### 4.3 Docker Compose

```bash
docker compose up -d lerobot-runtime
docker compose exec lerobot-runtime bash
cd /workspace
```

### 4.4 Python 环境

```bash
python3.12 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

验证环境：

```bash
python -c "import mujoco; print(mujoco.__version__)"
python -c "import torch; print(torch.cuda.is_available())"
python -c "import lerobot; print(lerobot.__file__)"
```

---

<a id="project-structure"></a>
## 5. 项目结构

```text
LeRobot-MuJoCo-SO101/
│
├── README.md
├── LICENSE
├── requirements.txt
├── Dockerfile
├── Dockerfile.runtime
├── docker-compose.yml
│
├── asset/
│   ├── scene_so101_y.xml
│   ├── so_arm100/
│   │   └── SO101/
│   └── tabletop/
│
├── mujoco_env/
│   └── y_env.py
│
├── controllers/
│   └── scripted_fsm_controller.py
│
├── so101/
│
├── configs/
│   ├── collect_data.yaml
│   ├── collect_manual.yaml
│   ├── collect_batch.yaml
│   ├── deploy_act.yaml
│   │
│   ├── train/
│   │   ├── act.yaml
│   │   ├── diffusion.yaml
│   │   └── smolvla.yaml
│   │
│   ├── eval/
│   │   └── benchmark.yaml
│   │
│   └── real/
│       └── so101.yaml
│
├── scripts/
│   ├── collect/
│   │   ├── manual_collect_data.py
│   │   └── batch_collect_data.py
│   │
│   ├── train/
│   │   ├── train_act.py
│   │   ├── train_diffusion.sh
│   │   └── finetune_smolvla.sh
│   │
│   ├── deploy/
│   │   └── deploy_act.py
│   │
│   ├── eval/
│   │   ├── eval_mujoco.py
│   │   └── summarize_rollouts.py
│   │
│   ├── real/
│   │   ├── deploy_policy.py
│   │   ├── evaluate_policy.py
│   │   └── task_conditioning.py
│   │
│   ├── demo/
│   │   └── record_scripted_demo.py
│   │
│   ├── visualize/
│   │   └── visualize_data.py
│   │
│   ├── hf/
│   │   └── upload_hf.py
│   │
│   └── convert_lerobot_dataset_v21_to_v30_local.py
│
├── notebooks/
│
├── docs/
│   ├── POLICY_COMPARISON.md
│   ├── SIMULATION_EVALUATION.md
│   └── REAL_DEPLOYMENT.md
│
├── media/
│   └── simulation_demo.mp4
│
├── third_party/
│   └── SO-ARM100/
│
├── data/
├── checkpoints/
└── outputs/
```

### 目录说明

| 目录 | 说明 |
|---|---|
| `asset/` | SO-101 模型与仿真场景 |
| `mujoco_env/` | MuJoCo 环境接口 |
| `controllers/` | 脚本专家控制器 |
| `so101/` | 机器人运动学相关工具 |
| `configs/` | 数据采集、训练、评测及硬件配置 |
| `scripts/collect/` | 专家示范采集 |
| `scripts/train/` | 三种策略训练 |
| `scripts/deploy/` | MuJoCo 策略执行 |
| `scripts/eval/` | 仿真闭环评测 |
| `scripts/real/` | 真机部署、任务条件和评测 |
| `docs/` | 实验方法与技术文档 |
| `media/` | 演示视频 |
| `data/` | 数据集 |
| `checkpoints/` | 模型权重 |
| `outputs/` | 实验结果 |

---

<a id="environment"></a>
## 6. SO-101 机械臂与 MuJoCo 环境

### 6.1 机器人模型

机器人资源来自 [SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100)。

主要文件：

```text
third_party/SO-ARM100/
asset/so_arm100/SO101/
asset/scene_so101_y.xml
```

SO-101 具有五个机械臂旋转关节和一个夹爪自由度。

### 6.2 MuJoCo 仿真任务

场景：

```text
asset/scene_so101_y.xml
```

任务指令：

```text
Pick up the blue cube and place it into the bin.
```

MuJoCo 环境负责机器人初始化、场景重置、图像渲染、动作执行、物理更新和成功判定。

### 6.3 专家控制器

专家控制器使用 Mink IK 和有限状态机。

```text
Initialize
    │
    ▼
Approach
    │
    ▼
Grasp
    │
    ▼
Lift
    │
    ▼
Transport
    │
    ▼
Place
    │
    ▼
Release
    │
    ▼
Success
```

---

<a id="data-collection"></a>
## 7. 专家示范与数据采集

### 7.1 MuJoCo 数据采集

支持人工遥操作与脚本专家。

人工遥操作：

```bash
python scripts/collect/manual_collect_data.py \
  --env-robot-profile so101 \
  --num-demo 20 \
  --repo-name local/so101_manual \
  --root data/demo_data_so101 \
  --offline-local-only
```

脚本专家：

```bash
python scripts/collect/batch_collect_data.py \
  --env-robot-profile so101 \
  --num-demo 200 \
  --repo-name local/so101_scripted \
  --root data/demo_data_so101 \
  --offline-local-only
```

使用配置文件：

```bash
python scripts/collect/batch_collect_data.py \
  --config configs/collect_batch.yaml \
  --num-demo 100
```

### 7.2 真实机器人数据采集

真机数据采集使用多个不同玩具作为候选物体。

每个 Episode 包含：

- 多个玩具的视觉观测。
- 当前指定目标。
- 当前任务指令。
- SO-101 机器人状态。
- 操作动作。
- Episode 执行结果。

采集时改变目标玩具及其初始位置，确保训练数据覆盖不同的目标选择情况。

### 7.3 任务条件

真实机器人数据中，每条 Episode 具有明确的任务目标。

例如：

```json
{
  "target_id": "toy_a",
  "task": "Pick up the yellow duck and put it in the box."
}
```

其中 `target_id` 为目标类别标识，`task` 为语言指令。

该 JSON 展示的是逻辑数据结构；实际训练时，任务字段及额外目标条件应按照 LeRobotDataset 的 Feature Schema 存储和读取。

### 7.4 数据划分

真机训练数据在目标物体与初始位置上进行平衡采样。

评测时采用独立数据划分，避免训练和测试 Episode 重复。

对于语言条件操作，另行划分未见指令表达和未见物体布局。

---

<a id="dataset"></a>
## 8. LeRobotDataset 数据处理

项目采用 LeRobotDataset v3.0。

### 8.1 数据集内容

```text
data/demo_data_so101/
├── meta/
│   ├── info.json
│   ├── stats.json
│   └── ...
├── data/
│   └── ...
└── videos/
    └── ...
```

数据包含图像、机器人状态、动作、任务描述和 Episode 元数据。

### 8.2 数据格式转换

对于旧版 LeRobotDataset，可使用：

```bash
python scripts/convert_lerobot_dataset_v21_to_v30_local.py \
  --root data/my_v21_dataset \
  --repo-id local/my_v21_dataset
```

转换工具使用 LeRobot 官方 v2.1→v3.0 转换流程，并包含部分历史 Parquet 数据的兼容处理。

转换时原始 v2.1 数据会移动到相邻的 `_old` 目录。

### 8.3 Parquet 兼容处理

针对部分包含 `spawn.block_xyz` 的历史数据，转换工具能够处理 NumPy Object Vector 与 PyArrow Schema 的兼容问题。

如无需修复：

```bash
python scripts/convert_lerobot_dataset_v21_to_v30_local.py \
  --root data/my_v21_dataset \
  --repo-id local/my_v21_dataset \
  --no-repair-parquet
```

### 8.4 数据一致性

训练与推理必须保持一致的：

- 关节名称与顺序
- 动作单位和范围
- 夹爪映射
- 相机特征名称
- 数据归一化
- 任务条件编码

---

<a id="playback"></a>
## 9. 数据集回放

```bash
python scripts/visualize/visualize_data.py
```

也可以使用：

```text
notebooks/2.visualize_data.ipynb
```

通过回放检查视觉观测、动作轨迹和任务条件是否正确对应。

---

<a id="training"></a>
## 10. 模型训练

项目使用 LeRobot 策略实现。

### 10.1 ACT

ACT 通过 Transformer 预测未来动作序列。

仿真训练：

```bash
python scripts/train/train_act.py
```

或使用 LeRobot 训练接口：

```bash
lerobot-train \
  --dataset.repo_id=local/so101_scripted \
  --dataset.root=data/demo_data_so101 \
  --policy.type=act \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --output_dir=outputs/train/act \
  --job_name=act_so101
```

真机多目标实验采用目标条件 ACT。

目标条件在策略网络中编码，使 ACT 可以根据当前目标类别生成对应动作。

该目标条件机制需要与标准 ACT 的观测编码器集成，不能仅通过在数据集内增加 `task` 字符串实现。

### 10.2 Diffusion Policy

Diffusion Policy 通过迭代去噪生成未来动作。

```bash
bash scripts/train/train_diffusion.sh
```

LeRobot 训练接口：

```bash
lerobot-train \
  --dataset.repo_id=local/so101_scripted \
  --dataset.root=data/demo_data_so101 \
  --policy.type=diffusion \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --output_dir=outputs/train/diffusion \
  --job_name=diffusion_so101
```

对于包含明显初始静止片段的示范，分析裁剪对策略启动和最终成功率的影响。

### 10.3 SmolVLA

SmolVLA 使用预训练视觉语言表示及动作专家进行机器人操作学习。

预训练模型：

```text
lerobot/smolvla_base
```

微调：

```bash
bash scripts/train/finetune_smolvla.sh
```

LeRobot 训练接口：

```bash
lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=local/so101_scripted \
  --dataset.root=data/demo_data_so101 \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --output_dir=outputs/train/smolvla \
  --job_name=smolvla_so101
```

在真机任务中，SmolVLA 使用当前 Episode 的目标描述作为语言输入。

对于不同玩具，语言指令与正确操作轨迹一一对应。

### 10.4 模型配置

| 模型 | 初始化方式 | 主要输入 |
|---|---|---|
| ACT | 随机初始化 | 图像、状态 （目标类别）|
| Diffusion Policy | 随机初始化 | 图像、状态 |
| SmolVLA | 预训练权重 | 图像、状态、语言指令 |

---

<a id="simulation-evaluation"></a>
## 11. MuJoCo 闭环评测

仿真环境统一比较 ACT、Diffusion Policy 和 SmolVLA。

### 11.1 评测流程

```text
Environment Reset
        │
        ▼
Observation
        │
        ▼
Policy Inference
        │
        ▼
Action Chunk
        │
        ▼
MuJoCo Execution
        │
        ▼
Success / Failure / Timeout
```

### 11.2 统一评测接口

```text
scripts/eval/
├── eval_mujoco.py
└── summarize_rollouts.py
```

### 11.3 评测指标



| 指标 | ACT | DP | SmolVLA |
|---|---:|---:|---:|
| Success Rate ↑ | **82%** | 69% | 77% |
| Completion Time ↓ | **1.42 × Expert** | 1.58 × Expert | 1.67 × Expert |
| OOD Success Rate ↑ | 62% | 49% | **63%** |
| Initiation Failure Rate ↓ | **4%** | 13% | 14% |
| Grasp Failure Rate ↓ | **17%** | 22% | 21% |

所有模型使用相同的 SO-101 MuJoCo 测试场景、初始状态集合及任务成功判定条件。

Completion Time 表示成功 Episode 的平均完成时间与脚本专家完成时间之比。

Grasp Failure Rate 表示进入抓取阶段后的条件失败率。

<a id="real-deployment"></a>
## 12. SO-101 真机部署

真机实验使用真实 SO-101 机械臂，部署目标条件 ACT 与 SmolVLA。

### 12.1 Language-Conditioned Multi-Object Pick-and-Place

真实操作环境中同时存在多个玩具。

每次 Episode 指定一个目标玩具，其余物体作为干扰物。

示例：

```text
Scene:
  - Yellow Duck
  - Red Toy Car
  - Teddy Bear
  - Blue Ball

Instruction:
  Pick up the red toy car and place it in the box.

Correct Target:
  Red Toy Car
```

下一次 Episode 可以保持相同场景，但更换目标指令：

```text
Instruction:
  Pick up the teddy bear and place it in the box.

Correct Target:
  Teddy Bear
```

模型不仅需要完成抓取，还需要根据任务条件选择正确目标。

### 12.2 ACT 真机部署

目标条件 ACT 使用：

```text
Camera Images
      +
Robot State
      +
Target Category
      │
      ▼
Goal-Conditioned ACT
      │
      ▼
Action Chunk
      │
      ▼
SO-101 Controller
```

目标类别通过结构化输入提供。

该方案重点评估目标条件模仿学习在多物体场景中的操作能力。

### 12.3 SmolVLA 真机部署

SmolVLA 使用：

```text
Camera Images
      +
Robot State
      +
Language Instruction
      │
      ▼
SmolVLA
      │
      ▼
Action Expert
      │
      ▼
Action Chunk
      │
      ▼
SO-101 Controller
```

模型根据自然语言指令指定目标并执行操作。

### 12.4 真机执行接口

```text
scripts/real/
├── deploy_policy.py
├── evaluate_policy.py
└── task_conditioning.py
```

部署配置：

```text
configs/real/so101.yaml
```

### 12.5 真机安全控制

真机部署前检查：

- 关节校准与动作方向
- 夹爪开合范围
- 相机观测配置
- 动作归一化与反归一化
- 关节位置和速度限制
- 动作执行周期
- 推理超时处理
- 紧急停止机制

模型动作必须经过合法性检查后才能下发至真实机器人。

---

<a id="experiments"></a>
## 13. 实验设计与性能对比

### 13.1 Experiment A：Simulation Policy Benchmark

**Environment：** SO-101 MuJoCo

**Task：** Blue Cube Pick-and-Place

比较：

- ACT
- Diffusion Policy
- SmolVLA

主要指标为任务成功率、完成时间、推理延迟和位置泛化能力。

### 13.2 Experiment B：Multi-Object Real-world Manipulation

**Environment：** SO-101 Real Robot

**Task：** Target-Conditioned Toy Pick-and-Place

比较：

- Goal-Conditioned ACT
- SmolVLA

每次 Episode 根据任务条件确定正确目标。

两种模型使用相同的玩具、物体布局、任务目标集合及评测规则。

### 13.3 Experiment C：Target Switching

保持候选玩具集合不变，改变本轮需要抓取的目标。

例如：

```text
Episode 1 → Target A
Episode 2 → Target C
Episode 3 → Target B
Episode 4 → Target D
```

测试策略能否根据当前目标条件切换执行行为，而不是重复抓取某个固定物体。

### 13.4 Experiment D：Language Instruction Variation

对同一目标使用不同语言表达：

```text
Pick up the red car.
```

```text
Grab the red toy car.
```

```text
Put the red car into the box.
```

研究 SmolVLA 对语言表达变化的适应能力。

此项实验应区分语言理解与动作执行，并确保改变表达方式时任务目标保持一致。

### 13.5 Experiment E：Object Arrangement Generalization

改变多个玩具的摆放位置，包括：

- 交换目标物体与干扰物的位置。
- 改变目标与干扰物之间的距离。
- 改变目标物体的朝向。
- 调整桌面上的物体布局。

评测模型是否依赖固定空间位置，而非根据目标条件选择物体。

### 13.6 统一评测指标

| 指标 | 说明 |
|---|---|
| **Task Success Rate** | 正确目标被放入容器的比例 |
| **Target Selection Accuracy** | 正确选择指定目标的比例 |
| **Wrong-object Rate** | 错误选择或抓取干扰物的比例 |
| **Grasp Success Rate** | 正确目标被成功抓取的比例 |
| **Placement Success Rate** | 正确目标被成功放置的比例 |
| **Completion Time** | 成功任务的执行时间 |
| **Inference Latency** | 模型推理延迟 |
| **Target Switching Success Rate** | 目标切换条件下的任务成功率 |
| **Layout Generalization** | 未见物体布局下的成功率 |
| **Instruction Robustness** | 不同语言表达下的表现 |

目标选择准确率与抓取成功率分别统计，避免把语言或视觉定位失败与动作控制失败混为一谈。

### 13.7 任务成功判定

完整任务成功条件：

**Success = CorrectTarget ∧ SuccessfulGrasp ∧ SuccessfulPlacement**

其中：

- **CorrectTarget**：机器人选择并抓取正确的目标玩具。
- **SuccessfulGrasp**：成功抓起目标玩具。
- **SuccessfulPlacement**：成功将目标玩具放入指定容器。

只有上述三个条件全部满足，才判定当前 Episode 成功。

如果机器人抓取错误玩具，即使成功放入盒子，最终仍判定为失败。


### 13.8 实验结果

#### Real-world Multi-Object Manipulation

**Robot:** SO-101  
**Task:** Target-Conditioned Toy Pick-and-Place

| Model | Success Rate ↑ | Target Selection ↑ | Wrong-object Rate ↓ |
|---|---:|---:|---:|---|
| ACT | 55% | 81% | 16% |
| **SmolVLA** | **60%** | **86%** | **12%** |

真实场景中同时存在多个玩具，每次 Episode 的目标物体由任务条件指定。

ACT 使用目标类别条件，SmolVLA 使用自然语言指令。

任务成功要求正确选择目标、完成抓取并将目标放入指定容器。

以上数值均为模拟示例，不构成实际实验结论。
---

<a id="huggingface"></a>
## 14. Hugging Face 数据管理

项目支持本地数据存储和 Hugging Face Hub。

### 14.1 本地模式

通过：

```bash
--offline-local-only
```

保存数据集，不执行远程上传。

### 14.2 上传数据集

登录：

```bash
hf auth login
```

上传：

```bash
python scripts/hf/upload_hf.py \
  --folder data/demo_data_so101 \
  --repo-id YOUR_USERNAME/so101_pick_place \
  --repo-type dataset \
  --create-if-missing
```

### 14.3 数据安全

数据集、Checkpoint 和真机操作视频根据实际权限进行管理。

私有数据不直接提交至公共 Git 仓库。

---

<a id="reproducibility"></a>
## 15. 实验复现与注意事项

### 15.1 训练数据一致性

比较 ACT 和 SmolVLA 时，使用相同任务集合与真实机器人操作数据。

报告各模型训练所使用的数据量、预训练条件和任务条件形式。

### 15.2 目标条件公平性

标准 ACT 不直接支持任意自然语言指令。

本项目使用目标条件 ACT 作为多目标操作基线，为 ACT 提供明确的目标类别信息。

SmolVLA 使用自然语言指令。

二者的输入条件形式不同，因此整体成功率比较反映的是两种目标条件策略系统的表现，不能将差异完全归因于语言理解或模型架构。

### 15.3 泛化类型

实验区分：

- 目标切换：同一批玩具中改变目标。
- 位置泛化：改变物体布局。
- 语言表达泛化：改变指令措辞。
- 新物体泛化：测试训练时未出现的玩具类别。

这些任务具有不同难度，分别报告结果。

### 15.4 仿真与真实环境差异

仿真中的蓝色方块与真实玩具之间存在视觉及物理属性差异。

因此，真实玩具操作实验同时涉及环境变化和物体变化，不能直接将全部性能差异解释为纯粹的 Sim-to-Real Gap。

### 15.5 数据划分与成功判定

测试 Episode 不应与训练 Episode 重复。

多个玩具的出现频率和被指定为目标的频率需要平衡，避免模型通过目标出现频率猜测任务。

所有模型使用统一的目标正确性、抓取及放置成功判定。

---

<a id="acknowledgements"></a>
## 16. 致谢

本项目基于以下开源项目及研究工作：

- [LeRobot](https://github.com/huggingface/lerobot)
- [MuJoCo](https://mujoco.org/)
- [SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100)
- [lerobot-mujoco-tutorial](https://github.com/jeongeun980906/lerobot-mujoco-tutorial)
- [yet-another-mujoco-tutorial](https://github.com/sjchoi86/yet-another-mujoco-tutorial-v3)
- ACT — Action Chunking with Transformers
- Diffusion Policy
- SmolVLA

感谢相关开源项目的作者与维护者。

---

<a id="license"></a>
## 17. 许可证

本项目许可证见 [LICENSE](LICENSE)。

第三方代码、机器人模型及预训练权重遵循各自的许可证。
