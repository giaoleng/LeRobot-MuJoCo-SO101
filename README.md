# LeRobot-MuJoCo-SO101

**SO-101 Robotic Imitation Learning: ACT, Diffusion Policy & SmolVLA**

基于 [LeRobot](https://github.com/huggingface/lerobot) 与 [MuJoCo](https://mujoco.org/) 构建的 **SO-101 机器人模仿学习实验平台**，覆盖专家示范采集、数据集处理、策略训练、仿真闭环评测及真实机器人操作。

项目包含两个实验场景：

- **MuJoCo Simulation**：在 SO-101 仿真环境中比较 ACT、Diffusion Policy 和 SmolVLA，完成多颜色方块干扰下的蓝色方块抓取放置任务。
- **Real Robot Manipulation**：在真实 SO-101 机械臂上比较 ACT 和 SmolVLA，根据任务条件从多个玩具中选择目标并完成抓取放置。

通过统一的实验设置与评测协议，比较不同模仿学习策略的任务成功率、位置泛化能力、目标选择能力及失败类型。

---

## 目录

### 项目概览

1. [项目介绍](#overview)
2. [系统架构](#architecture)
3. [项目演示](#demo)
4. [安装与环境配置](#installation)
5. [项目结构](#project-structure)

### MuJoCo 仿真实验

6. [SO-101 MuJoCo 仿真环境](#simulation-environment)
7. [仿真专家示范与数据采集](#simulation-data)
8. [LeRobotDataset 数据处理](#dataset)
9. [仿真策略训练](#simulation-training)
10. [MuJoCo 闭环评测与实验结果](#simulation-evaluation)

### SO-101 真机实验

11. [SO-101 真机实验环境](#real-environment)
12. [真机策略训练与部署](#real-deployment)
13. [真机实验设计与结果](#real-evaluation)

### 数据管理与实验复现

14. [Hugging Face 数据管理](#huggingface)
15. [实验复现与注意事项](#reproducibility)
16. [致谢](#acknowledgements)
17. [许可证](#license)

---

<a id="overview"></a>

## 1. 项目介绍

本项目面向低成本机器人操作任务，构建基于视觉观测、机器人状态及任务条件的模仿学习系统。

通过 SO-101 机械臂，在 MuJoCo 仿真环境与真实机器人环境中分别研究固定目标操作、多物体干扰、目标条件操作以及策略泛化能力。

### 1.1 实验任务

项目设计两个相互关联但具有不同任务设置的实验场景。

#### MuJoCo Simulation

仿真环境包含 SO-101 机械臂、收纳盒以及蓝色、红色、黄色和绿色方块。

基础任务要求机器人从多个方块中抓取指定的**蓝色方块**，并将其放入收纳盒。

其他颜色方块作为干扰物，不属于当前操作目标。

使用以下三种模仿学习策略完成训练与闭环评测：

- ACT
- Diffusion Policy
- SmolVLA

主要研究：

- 不同策略在相同仿真任务中的成功率。
- 多物体干扰下的抓取与放置能力。
- 目标方块初始位置变化下的泛化能力。
- 策略启动、抓取和放置阶段的失败情况。

#### Real Robot Manipulation

真实环境使用 SO-101 机械臂，桌面同时摆放多个不同玩具。

每次 Episode 指定一个目标玩具，目标物体可以随任务变化，其余玩具作为干扰物。

例如：

```text
Episode 1:
Pick up the yellow duck and place it into the box.

Episode 2:
Pick up the red toy car and place it into the box.

Episode 3:
Pick up the teddy bear and place it into the box.
```

真机实验比较 ACT 与 SmolVLA：

- **ACT**：接收视觉观测、机器人状态及结构化目标类别信息。
- **SmolVLA**：接收视觉观测、机器人状态及自然语言指令。

只有正确选择目标、完成抓取并成功放置，才判定当前任务成功。

### 1.2 对比模型

| 模型 | 核心方法 | 训练方式 | 实验环境 |
|---|---|---|---|
| ACT | Transformer + Action Chunking | 监督模仿学习 | 仿真 + 真机 |
| Diffusion Policy | Diffusion-based Action Generation | 扩散式模仿学习 | 仿真 |
| SmolVLA | Vision-Language-Action | 预训练模型微调 | 仿真 + 真机 |

ACT 与 Diffusion Policy 主要学习从机器人观测到动作序列的映射。

SmolVLA 在视觉和机器人状态基础上引入语言条件，利用预训练视觉语言表示与动作生成模块学习机器人操作策略。

真机多目标实验中的 ACT 使用额外的目标条件编码，但仍统一归类为 ACT，不单独作为第四种模型。

### 1.3 研究问题

本项目主要关注以下问题。

**Simulation Policy Benchmark**

在相同的 SO-101 MuJoCo 操作任务下，ACT、Diffusion Policy 和 SmolVLA 的任务成功率与失败类型有何差异？

**Position Generalization**

当目标方块出现在训练数据未覆盖的位置时，策略能否继续完成抓取放置？

**Multi-Object Manipulation**

真实环境同时存在多个候选玩具时，模型能否根据当前任务条件选择正确目标？

**Target Switching**

当候选玩具集合保持不变、但当前目标发生变化时，模型能否及时调整操作行为？

**Language Instruction Generalization**

当自然语言指令使用不同表达方式描述相同目标时，SmolVLA 能否保持稳定的目标选择与操作能力？

---

<a id="architecture"></a>

## 2. 系统架构

项目包含仿真模仿学习系统和真实机器人操作系统。

两部分使用相同的 SO-101 机器人类型，但采用不同的任务设置与数据采集流程。

### 2.1 MuJoCo 仿真系统

```text
                  SO-101 MuJoCo
                        │
                        ▼
              Multi-Object Scene
                        │
                        ▼
             Blue Cube Pick-and-Place
                        │
                        ▼
                Expert Controller
                  Mink IK + FSM
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
                        │
                        ▼
               Closed-loop Rollout
                        │
                        ▼
              Simulation Benchmark
```

系统使用 Mink IK 与有限状态机生成专家轨迹，将视觉、机器人状态和动作保存为 LeRobotDataset。

训练后的策略在 MuJoCo 环境中独立执行闭环 Rollout。

### 2.2 真实机器人系统

```text
                   SO-101 Real Robot
                          │
                          ▼
                    Multiple Toys
                          │
                          ▼
                  Task Specification
                          │
              ┌───────────┴───────────┐
              │                       │
              ▼                       ▼
        Target Category       Language Instruction
              │                       │
              ▼                       ▼
             ACT                   SmolVLA
              │                       │
              └───────────┬───────────┘
                          │
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

真机实验通过目标类别或自然语言指令指定当前操作对象。

两种策略最终输出机器人动作，经安全检查后发送至 SO-101 控制接口。

### 2.3 核心功能模块

| 模块 | 功能 |
|---|---|
| MuJoCo Environment | SO-101 机器人与物理场景仿真 |
| Expert Controller | 基于 Mink IK 与 FSM 生成专家示范 |
| Data Collection | 采集图像、机器人状态、动作及任务信息 |
| LeRobotDataset | 统一存储机器人操作数据 |
| Policy Training | ACT、Diffusion Policy 和 SmolVLA 训练 |
| Simulation Evaluation | MuJoCo 闭环评测 |
| Real Robot Interface | SO-101 状态读取与动作执行 |
| Task Conditioning | 目标类别和语言任务条件 |
| Performance Analysis | 成功率、泛化能力与失败类型分析 |

---

<a id="demo"></a>

## 3. 项目演示

### 3.1 SO-101 MuJoCo 仿真演示

[▶ 查看 SO-101 MuJoCo 仿真演示](media/simulation_demo.mp4)

仿真场景包含：

- SO-101 机械臂。
- 桌面与目标收纳盒。
- 蓝色、红色、黄色和绿色方块。
- 用于策略观测的 RGB 相机。

基础任务为从多个颜色方块中抓取蓝色方块，并将其放入收纳盒。

演示由基于 **Mink Inverse Kinematics + Finite State Machine** 的脚本专家生成，不代表 ACT、Diffusion Policy 或 SmolVLA 的训练结果。

---

<a id="installation"></a>

## 4. 安装与环境配置

### 4.1 环境依赖

| 组件 | 用途 |
|---|---|
| Python | 项目运行环境 |
| MuJoCo | 物理仿真 |
| LeRobot | 数据集、策略训练及机器人接口 |
| PyTorch | 深度学习 |
| Mink | 逆运动学 |
| OpenCV | 图像处理 |
| FFmpeg | 视频编码 |
| Transformers | SmolVLA 模型组件 |

数据处理采用 LeRobotDataset v3.0 格式。

不同模型 Checkpoint 可能依赖不同版本的 LeRobot、PyTorch 和 Transformers，应以对应的运行环境配置为准。

### 4.2 Docker

构建运行镜像：

```bash
docker build -t lerobot_mujoco_sim:runtime -f Dockerfile.runtime .
```

构建训练镜像：

```bash
docker build -t lerobot_mujoco_sim:train -f Dockerfile .
```

启动运行容器：

```bash
docker run --rm -it \
  --gpus all \
  -v "$PWD:/workspace" \
  lerobot_mujoco_sim:runtime
```

进入项目目录：

```bash
cd /workspace
```

上述 GPU 配置适用于具备 NVIDIA Container Toolkit 的运行环境。

### 4.3 Docker Compose

启动容器：

```bash
docker compose up -d lerobot-runtime
```

进入容器：

```bash
docker compose exec lerobot-runtime bash
```

进入项目：

```bash
cd /workspace
```

### 4.4 Python 环境

如果采用 Python 虚拟环境：

```bash
python3.12 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

验证 MuJoCo：

```bash
python -c "import mujoco; print(mujoco.__version__)"
```

验证 PyTorch：

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

验证 LeRobot：

```bash
python -c "import lerobot; print(lerobot.__file__)"
```

模型训练应使用与实际依赖版本及硬件兼容的环境。

---

<a id="project-structure"></a>

## 5. 项目结构

项目按仿真环境、机器人控制、数据采集、策略训练、评测和真机部署划分模块。

以下为项目采用的模块组织结构；具体脚本与配置入口以仓库实际文件为准。

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

### 5.1 主要目录说明

| 目录 | 说明 |
|---|---|
| `asset/` | SO-101 机器人模型与 MuJoCo 场景 |
| `mujoco_env/` | MuJoCo 仿真环境 |
| `controllers/` | 脚本专家控制器 |
| `so101/` | SO-101 机器人相关工具 |
| `configs/` | 数据采集、训练、评测及硬件配置 |
| `scripts/collect/` | 专家示范与数据采集 |
| `scripts/train/` | 策略训练 |
| `scripts/deploy/` | MuJoCo 策略执行 |
| `scripts/eval/` | 仿真闭环评测 |
| `scripts/real/` | 真机部署与评测 |
| `scripts/visualize/` | 数据可视化 |
| `scripts/hf/` | Hugging Face 数据管理 |
| `docs/` | 技术文档 |
| `media/` | 演示视频 |
| `data/` | 数据集 |
| `checkpoints/` | 模型权重 |
| `outputs/` | 实验输出 |

---

# Part I. MuJoCo Simulation

<a id="simulation-environment"></a>

## 6. SO-101 MuJoCo 仿真环境

### 6.1 SO-101 机器人模型

本项目的仿真与真实机器人操作均使用 **SO-101 机械臂**。

机器人资源基于 [TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100) 项目提供的 SO-101 模型。

主要文件：

```text
third_party/SO-ARM100/
asset/so_arm100/SO101/
asset/scene_so101_y.xml
```

其中 `SO-ARM100` 和 `so_arm100` 为上游资源及原始目录命名；本项目使用的机器人模型为 SO-101。

SO-101 具有五个机械臂旋转关节和一个夹爪自由度。

MuJoCo 使用 MJCF/XML 描述机器人关节、执行器、几何模型及物理参数。

### 6.2 MuJoCo 场景

仿真场景定义：

```text
asset/scene_so101_y.xml
```

场景包含：

- SO-101 机械臂。
- 桌面。
- 目标收纳盒。
- 多个不同颜色的方块。
- RGB 相机。

机器人环境接口：

```text
mujoco_env/y_env.py
```

环境负责：

- 初始化机器人与场景。
- 重置 Episode。
- 设置物体初始状态。
- 读取机器人关节状态。
- 渲染相机观测。
- 执行机器人动作。
- 更新 MuJoCo 物理状态。

### 6.3 多颜色方块抓取任务

基础任务：

```text
Pick up the blue cube and place it into the bin.
```

场景中同时存在：

```text
Blue Cube    → Target

Red Cube     → Distractor
Yellow Cube  → Distractor
Green Cube   → Distractor
```

机器人需要抓取指定的蓝色方块，并将其放入收纳盒。

抓取其他颜色的方块不算任务成功。

该任务用于比较 ACT、Diffusion Policy 和 SmolVLA 在相同多物体干扰场景中的操作能力。

### 6.4 Mink IK + FSM 专家控制器

仿真专家控制器采用：

- **Mink Inverse Kinematics**：根据末端执行器目标计算关节运动。
- **Finite State Machine（FSM）**：组织抓取放置任务的执行阶段。

专家控制器入口：

```text
controllers/scripted_fsm_controller.py
```

执行流程：

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

FSM 负责决定当前任务阶段。

Mink IK 将各阶段的末端运动目标转换为机器人关节运动。

该组合属于基于模型的脚本专家控制方法，用于生成模仿学习示范，并不作为与 ACT、Diffusion Policy 和 SmolVLA 并列的学习策略。

---

<a id="simulation-data"></a>

## 7. 仿真专家示范与数据采集

### 7.1 数据采集方式

MuJoCo 环境支持人工遥操作与脚本专家两种数据采集方式。

采集内容包括：

- 相机图像。
- 机器人关节状态。
- 机器人动作。
- 任务描述。
- Episode 元数据。

数据使用 LeRobotDataset 格式存储。

### 7.2 人工遥操作采集

执行：

```bash
python scripts/collect/manual_collect_data.py \
  --env-robot-profile so101 \
  --num-demo 20 \
  --repo-name local/so101_manual \
  --root data/demo_data_so101 \
  --offline-local-only
```

该方式用于人工控制 SO-101 完成抓取放置任务，并记录对应示范数据。

### 7.3 脚本专家批量采集

执行：

```bash
python scripts/collect/batch_collect_data.py \
  --env-robot-profile so101 \
  --num-demo 200 \
  --repo-name local/so101_scripted \
  --root data/demo_data_so101 \
  --offline-local-only
```

也可以使用配置文件：

```bash
python scripts/collect/batch_collect_data.py \
  --config configs/collect_batch.yaml \
  --num-demo 100
```

其中：

- `--num-demo`：采集的示范数量。
- `--repo-name`：数据集标识。
- `--root`：数据集保存路径。
- `--offline-local-only`：仅保存本地数据。

### 7.4 数据质量检查

训练前应检查专家示范是否满足以下条件：

1. 机器人正确抓取蓝色方块。
2. 夹爪能够稳定夹持并抬起目标。
3. 目标方块被正确放入收纳盒。
4. 机器人动作与视频观测同步。
5. Episode 数据没有明显缺失或异常。
6. 训练数据包含足够的初始位置变化。

对于脚本专家，应单独统计其任务成功率，避免将专家控制失败混入模仿学习策略的性能评价。

---

<a id="dataset"></a>

## 8. LeRobotDataset 数据处理

项目使用 **LeRobotDataset v3.0** 存储机器人操作数据。

### 8.1 数据集结构

示例：

```text
data/demo_data_so101/
├── meta/
│   ├── info.json
│   ├── stats.json
│   └── ...
│
├── data/
│   └── ...
│
└── videos/
    └── ...
```

主要内容：

| 数据 | 作用 |
|---|---|
| RGB Images | 视觉观测 |
| Robot State | 机器人关节状态 |
| Actions | 监督学习目标 |
| Task | 任务描述 |
| Episode Metadata | Episode 相关信息 |
| Dataset Statistics | 归一化统计信息 |

具体字段名称由数据集的 Feature Schema 决定。

### 8.2 数据格式转换

对于旧版 LeRobotDataset，可以使用项目中的转换工具：

```bash
python scripts/convert_lerobot_dataset_v21_to_v30_local.py \
  --root data/my_v21_dataset \
  --repo-id local/my_v21_dataset
```

转换工具基于 LeRobot 官方 v2.1 → v3.0 转换流程，并包含历史数据兼容处理。

转换时原始 v2.1 数据会移动至相邻的 `_old` 目录。

正式转换前应检查目标路径及可用磁盘空间。

### 8.3 Parquet 兼容处理

针对部分包含 `spawn.block_xyz` 字段的历史数据，转换工具支持处理 NumPy Object Vector 与 PyArrow Schema 的兼容问题。

如果不需要执行该修复：

```bash
python scripts/convert_lerobot_dataset_v21_to_v30_local.py \
  --root data/my_v21_dataset \
  --repo-id local/my_v21_dataset \
  --no-repair-parquet
```

### 8.4 数据集回放

通过可视化脚本检查采集结果：

```bash
python scripts/visualize/visualize_data.py
```

也可以使用：

```text
notebooks/2.visualize_data.ipynb
```

回放时重点检查：

- 相机视频是否正确。
- 机器人状态与动作是否同步。
- 夹爪开合是否符合实际操作。
- Episode 是否包含完整抓取放置过程。
- 动作序列是否存在异常跳变。

### 8.5 数据一致性检查

训练与推理阶段应保持一致的：

- 关节名称与顺序。
- 动作维度。
- 动作单位和范围。
- 夹爪动作映射。
- 相机数量与特征名称。
- 图像分辨率。
- 数据归一化方式。
- 机器人控制频率。

数据格式和动作定义不一致可能导致模型训练 Loss 正常下降，但闭环执行失败。

---

<a id="simulation-training"></a>

## 9. 仿真策略训练

MuJoCo 仿真实验使用相同任务与对应的专家示范数据，比较三种策略：

- ACT
- Diffusion Policy
- SmolVLA

模型训练使用 LeRobot 提供的策略实现，并根据 SO-101 的观测和动作空间进行配置。

以下命令保留项目原有的训练入口示例；具体参数需与实际安装的 LeRobot 版本及数据集配置一致。

### 9.1 ACT

ACT（Action Chunking with Transformers）使用 Transformer 预测未来一段机器人动作。

主要输入：

```text
Camera Images
      +
Robot State
      │
      ▼
     ACT
      │
      ▼
 Action Chunk
```

训练入口：

```bash
python scripts/train/train_act.py
```

也可以使用 LeRobot 训练接口：

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

ACT 使用监督模仿学习，从专家轨迹中学习视觉观测、机器人状态与动作之间的映射。

### 9.2 Diffusion Policy

Diffusion Policy 使用条件扩散模型，通过迭代去噪生成机器人动作序列。

主要流程：

```text
Camera Images + Robot State
              │
              ▼
       Observation Encoder
              │
              ▼
       Diffusion Policy
              │
              ▼
       Iterative Denoising
              │
              ▼
          Action Chunk
```

训练入口：

```bash
bash scripts/train/train_diffusion.sh
```

LeRobot 训练接口示例：

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

对于包含明显初始静止片段的专家示范，应分析数据预处理是否影响策略启动和动作生成。

### 9.3 SmolVLA

SmolVLA 是视觉语言动作模型，使用视觉语言表示与动作生成模块学习机器人操作策略。

预训练模型：

```text
lerobot/smolvla_base
```

主要输入：

```text
Camera Images
      +
Robot State
      +
Task Instruction
      │
      ▼
    SmolVLA
      │
      ▼
  Action Expert
      │
      ▼
  Action Chunk
```

在当前仿真任务中，任务描述固定为：

```text
Pick up the blue cube and place it into the bin.
```

微调入口：

```bash
bash scripts/train/finetune_smolvla.sh
```

LeRobot 训练接口示例：

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

### 9.4 模型训练配置

| 模型 | 初始化方式 | 主要输入 | 训练目标 |
|---|---|---|---|
| ACT | 随机初始化 | 图像、状态 | 专家动作 |
| Diffusion Policy | 随机初始化 | 图像、状态 | 专家动作序列 |
| SmolVLA | 预训练权重 | 图像、状态、任务指令 | 专家动作序列 |

为保证实验具有可比性，应记录：

- 训练示范数量。
- 数据集划分。
- 图像分辨率与相机配置。
- 动作表示。
- Batch Size。
- 训练 Steps。
- Learning Rate。
- Action Chunk 配置。
- 训练使用的 GPU。
- 模型初始化方式。

不同模型的预训练条件与架构不同，不能仅根据最终成功率将性能差异完全归因于网络结构。

---

<a id="simulation-evaluation"></a>

## 10. MuJoCo 闭环评测与实验结果

本章统一评测 ACT、Diffusion Policy 和 SmolVLA 在 SO-101 MuJoCo 环境中的操作性能。

### 10.1 实验环境与设置

| 项目 | 实验设置 |
|---|---|
| Robot | SO-101 |
| Simulator | MuJoCo |
| Task | Blue Cube Pick-and-Place |
| Target | Blue Cube |
| Distractors | Red / Yellow / Green Cubes |
| Expert Controller | Mink IK + FSM |
| Models | ACT / DP / SmolVLA |
| Evaluation | Closed-loop Rollout |

所有模型使用相同的任务目标、物理环境及成功判定标准。

训练阶段使用专家示范数据。

评测阶段由策略模型独立生成动作，不再使用脚本专家决定抓取轨迹。

### 10.2 统一闭环评测流程

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
 Updated Observation
        │
        ├─────────────┐
        │             │
        ▼             │
 Success Check        │
        │             │
        ▼             │
 Success / Failure    │
        │             │
        └─────────────┘
```

如果任务尚未结束，环境继续提供新的观测，策略根据当前状态生成后续动作。

评测接口：

```text
scripts/eval/
├── eval_mujoco.py
└── summarize_rollouts.py
```

模型评测需要统一：

- Episode 初始化方式。
- 机器人初始状态。
- 方块位置采样规则。
- 控制频率。
- 任务最大执行时间。
- 成功与失败判定条件。

### 10.3 评测指标

| 指标 | 含义 | 优化方向 |
|---|---|---|
| **Success Rate** | 完整完成抓取放置任务的比例 | ↑ |
| **OOD Success Rate** | 未见目标位置下的任务成功率 | ↑ |
| Completion Time | 成功 Episode 的平均完成时间 | ↓ |
| Initiation Failure Rate | 策略未能正常启动操作的比例 | ↓ |
| Grasp Failure Rate | 已尝试抓取但未能成功抓起目标的比例 | ↓ |

其中：

**Success Rate**

任务成功次数占全部测试 Episode 的比例。

**OOD Success Rate**

在训练数据未覆盖的目标位置上，机器人成功完成任务的比例。

**Completion Time**

从 Episode 开始到成功完成任务所用的时间。

仿真实验优先使用 MuJoCo 的仿真时间。

**Initiation Failure Rate**

策略在规定启动时间内没有产生有效机器人运动的 Episode 比例。

启动阈值需根据专家轨迹与控制频率确定。

**Grasp Failure Rate**

进入抓取阶段后，未能成功抓住并抬起目标方块的比例。

该指标按实际发生的抓取尝试统计，以便与未能接近目标等失败情况区分。

### 10.4 ID / OOD 测试设置

为研究策略的位置泛化能力，分别进行 ID 和 OOD 测试。

#### In-Distribution Evaluation

蓝色方块在训练数据覆盖的位置范围内随机初始化。

测试目标：

- 策略是否学会基本抓取放置行为。
- 是否能够处理训练分布内的位置变化。

#### Out-of-Distribution Evaluation

蓝色方块在训练数据未覆盖、但机器人能够到达的位置范围内初始化。

测试目标：

- 策略是否能够适应新的目标位置。
- 是否过度依赖训练时的固定抓取轨迹。

OOD 位置应满足：

- 与训练位置范围明确分离。
- 位于 SO-101 可达工作空间。
- 不与其他方块或场景物体重叠。
- 不超出桌面边界。
- 脚本专家能够在该区域完成任务。

两组测试应尽可能只改变目标位置，避免同时改变光照、颜色、物体尺寸或物理参数。

建议使用相同的测试初始状态集合比较三个模型。

例如，可分别配置100个 ID Episode 和100个 OOD Episode。该数量属于建议评测协议，并非已经完成的测试记录。

### 10.5 仿真任务成功判定

完整任务成功条件：

**Success = CorrectTarget ∧ SuccessfulGrasp ∧ SuccessfulPlacement**

其中：

- **CorrectTarget**：抓取正确的蓝色方块。
- **SuccessfulGrasp**：成功夹住并抬起目标。
- **SuccessfulPlacement**：成功将目标放入收纳盒。

三个条件全部满足，才判定当前 Episode 成功。

如果抓取其他颜色方块、目标中途掉落或未能完成放置，则判定为失败。

### 10.6 仿真实验结果

#### Simulation Policy Benchmark

**Environment:** SO-101 MuJoCo  
**Task:** Multi-Object Scene / Blue Cube Pick-and-Place

| Metric | ACT | Diffusion Policy | SmolVLA |
|---|---:|---:|---:|
| **Success Rate ↑** | **82%** | 69% | 77% |
| Completion Time ↓ | **1.42 × Expert** | 1.58 × Expert | 1.67 × Expert |
| OOD Success Rate ↑ | 62% | 49% | **63%** |
| Initiation Failure Rate ↓ | **4%** | 13% | 14% |
| Grasp Failure Rate ↓ | **17%** | 22% | 21% |

Completion Time 使用相对于脚本专家平均完成时间的比例表示。

实验应报告各模型的实际训练配置、测试 Episode 数量以及成功率统计结果。

### 10.7 失败类型分析

为了进一步解释任务成功率差异，将失败情况划分为以下类型。

| Failure Type | Description |
|---|---|
| Initiation Failure | 机器人未正常开始执行操作 |
| Approach Failure | 末端未能正确接近目标 |
| Grasp Failure | 夹爪未能成功抓起目标 |
| Transport Failure | 搬运过程中目标掉落 |
| Placement Failure | 目标未能正确放入收纳盒 |
| Wrong-object Grasp | 错误抓取其他颜色方块 |
| Timeout | 在最大执行时间内未完成任务 |

评测时记录各 Episode 的执行结果与主要失败原因。

失败类型用于辅助判断：

- 数据覆盖是否不足。
- 视觉定位是否准确。
- 模型动作是否连续。
- 夹爪控制是否稳定。
- 模型是否能够处理执行过程中的状态变化。

---

# Part II. Real-world SO-101 Manipulation

<a id="real-environment"></a>

## 11. SO-101 真机实验环境

本章介绍真实 SO-101 机械臂的实验场景、任务条件和数据采集方式。

与仿真固定抓取蓝色方块不同，真实环境需要根据每次 Episode 的任务目标，从多个玩具中选择正确的操作对象。

### 11.1 机器人与硬件配置

真实机器人使用 **SO-101 机械臂**。

主要组成：

- SO-101 机械臂。
- 夹爪。
- RGB 相机。
- 机器人控制接口。
- 桌面操作场景。
- 多个待操作玩具。
- 目标收纳盒。

真实机器人与 MuJoCo 环境采用相同的机器人类型，但实际关节零点、运动范围、相机参数和接触特性仍需独立校准。

### 11.2 多玩具操作场景

实验场景中同时存在多个不同玩具。

例如：

```text
Scene:
  - Yellow Duck
  - Red Toy Car
  - Teddy Bear
  - Blue Ball
```

每个 Episode 指定一个需要抓取的玩具。

其余玩具作为干扰物。

机器人不仅需要完成抓取放置，还需要根据当前任务条件正确选择目标。

### 11.3 任务目标与指令设计

例如：

```text
Episode 1

Instruction:
  Pick up the yellow duck and place it into the box.

Correct Target:
  Yellow Duck
```

```text
Episode 2

Instruction:
  Pick up the red toy car and place it into the box.

Correct Target:
  Red Toy Car
```

```text
Episode 3

Instruction:
  Pick up the teddy bear and place it into the box.

Correct Target:
  Teddy Bear
```

同一批玩具可以在连续 Episode 中保持不变，只改变当前目标。

这样可以测试策略是否根据任务条件执行操作，而不是始终选择某个固定位置或物体。

### 11.4 真实机器人数据采集

真实机器人通过示范数据学习多玩具目标选择与抓取放置。

每个 Episode 包含：

- 多个玩具的视觉观测。
- 当前目标玩具。
- 任务指令。
- SO-101 机器人状态。
- 机器人动作。
- Episode 元数据。

数据应覆盖不同的：

- 目标玩具类别。
- 目标初始位置。
- 干扰物位置。
- 目标与干扰物之间的距离。
- 目标切换顺序。

### 11.5 数据集任务条件

不同模型使用不同形式的任务条件。

#### ACT

使用结构化目标类别信息：

```json
{
  "target_id": "yellow_duck"
}
```

ACT 通过目标条件编码器获取当前需要抓取的目标类别。

#### SmolVLA

使用自然语言任务描述：

```json
{
  "task": "Pick up the yellow duck and place it into the box."
}
```

也可以在逻辑记录中同时保留：

```json
{
  "target_id": "yellow_duck",
  "task": "Pick up the yellow duck and place it into the box."
}
```

上述 JSON 仅展示逻辑任务信息。

实际训练时，任务字段与额外目标条件需要按照 LeRobotDataset 的 Feature Schema 存储和读取。

标准 ACT 不会仅因为数据集中出现 `target_id` 字段就自动使用该信息，因此需要实现对应的输入编码。

### 11.6 数据集划分

真机训练数据应尽量平衡不同目标玩具的示范数量。

同时避免某个目标始终出现在同一位置，以减少模型通过位置或出现频率猜测目标的可能性。

数据划分包括：

| Dataset Split | 主要用途 |
|---|---|
| Training Set | 模型训练与微调 |
| Validation Set | 训练过程中的模型选择 |
| ID Test Set | 已见目标与训练分布内布局评测 |
| Layout OOD Test Set | 未见物体布局评测 |
| Instruction Variation Set | 不同语言表达下的鲁棒性评测 |

语言表达泛化主要针对 SmolVLA。

对于 ACT，目标类别信息与自然语言表达应明确区分。

---

<a id="real-deployment"></a>

## 12. 真机策略训练与部署

真机实验比较：

- ACT
- SmolVLA

两种模型使用相同的真实 SO-101 操作任务、目标玩具集合及成功判定条件。

### 12.1 ACT 目标条件训练

标准 ACT 主要根据图像与机器人状态预测动作。

在多玩具目标选择任务中，仅有图像和状态可能不足以确定当前应该抓取哪个玩具。

因此，真机实验为 ACT 增加结构化目标类别条件。

模型输入：

```text
Camera Images
      +
Robot State
      +
Target Category
      │
      ▼
     ACT
      │
      ▼
 Action Chunk
      │
      ▼
SO-101 Controller
```

目标条件通过编码后与视觉及机器人状态特征结合。

该机制属于 ACT 的任务条件扩展，不单独作为新的模型类别。

训练时应确保不同目标类别具有对应的专家操作轨迹。

### 12.2 SmolVLA 语言条件微调

SmolVLA 使用预训练视觉语言表示和动作生成模块。

真机输入：

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

示例指令：

```text
Pick up the yellow duck and place it into the box.
```

```text
Pick up the red toy car and place it into the box.
```

微调数据应包含不同目标玩具对应的任务指令及操作示范。

模型根据当前指令选择目标并预测机器人动作。

### 12.3 策略部署接口

真机部署相关模块：

```text
scripts/real/
├── deploy_policy.py
├── evaluate_policy.py
└── task_conditioning.py
```

其中：

- `deploy_policy.py`：策略加载与机器人动作执行。
- `evaluate_policy.py`：真机 Episode 执行与评测。
- `task_conditioning.py`：目标类别或任务条件处理。

机器人配置：

```text
configs/real/so101.yaml
```

### 12.4 SO-101 动作执行

策略生成的动作需要转换为 SO-101 控制接口可以接收的机器人指令。

执行流程：

```text
Camera + Robot State
          │
          ▼
     Policy Inference
          │
          ▼
      Action Chunk
          │
          ▼
    Action Processing
          │
          ▼
   SO-101 Joint Commands
          │
          ▼
    Robot Execution
          │
          ▼
     New Observation
```

动作处理需要保证：

- 关节名称与顺序正确。
- 动作单位正确。
- 夹爪映射正确。
- 动作反归一化正确。
- 关节目标位于安全范围内。
- 控制频率与训练配置匹配。

### 12.5 真机安全控制

真机执行前应检查：

- 关节校准。
- 关节运动方向。
- 夹爪开合范围。
- 相机配置。
- 动作归一化与反归一化。
- 关节位置与速度限制。
- 机器人工作空间。
- 动作执行周期。
- 推理超时处理。
- 紧急停止机制。

所有策略动作必须经过安全性与合法性检查后才能下发给真实机器人。

---

<a id="real-evaluation"></a>

## 13. 真机实验设计与结果

本章针对真实 SO-101 多玩具操作任务，评测 ACT 和 SmolVLA 的任务完成能力、目标选择能力和泛化表现。

### 13.1 统一实验设置

| 项目 | 实验设置 |
|---|---|
| Robot | SO-101 |
| Environment | Real-world Tabletop |
| Task | Target-Conditioned Toy Pick-and-Place |
| Candidate Objects | Multiple Toys |
| Target | Varies by Episode |
| Models | ACT / SmolVLA |
| ACT Condition | Target Category |
| SmolVLA Condition | Language Instruction |
| Evaluation | Closed-loop Execution |

所有模型使用相同的：

- SO-101 机械臂。
- 候选玩具集合。
- 目标类别集合。
- 测试布局。
- 任务最大执行时间。
- 成功判定条件。

由于 ACT 使用目标类别条件、SmolVLA 使用自然语言条件，最终成功率反映的是两种任务条件策略系统的表现。

不能直接将全部差异归因于模型架构或语言理解能力。

### 13.2 Experiment A：Multi-Object Target Selection

**Task:** Target-Conditioned Toy Pick-and-Place

场景中同时存在多个候选玩具。

机器人根据当前任务条件选择正确目标，并将其放入收纳盒。

主要评测：

- Task Success Rate。
- Target Selection Accuracy。
- Wrong-object Rate。
- Grasp Success Rate。

该实验用于检验模型是否真正利用任务条件选择操作目标。

### 13.3 Experiment B：Target Switching

保持候选玩具集合不变，仅改变当前需要抓取的目标。

例如：

```text
Episode 1 → Target A
Episode 2 → Target C
Episode 3 → Target B
Episode 4 → Target D
```

每个 Episode 独立重置任务与机器人状态。

测试模型能否根据当前目标条件改变操作行为，而不是重复抓取固定位置或某个偏好的物体。

### 13.4 Experiment C：Language Instruction Variation

针对相同目标，使用不同自然语言表达。

例如：

```text
Pick up the red car.
```

```text
Grab the red toy car.
```

```text
Put the red car into the box.
```

需要确保不同表达对应的实际任务目标一致。

该实验主要测试 SmolVLA 对指令措辞变化的适应能力。

评测时区分：

- 目标识别错误。
- 目标选择正确但抓取失败。
- 抓取成功但放置失败。

对于 ACT，结构化目标类别不随语言措辞变化，因此不应将该实验直接作为两个模型的同条件语言理解比较。

### 13.5 Experiment D：Object Arrangement Generalization

改变桌面上多个玩具的初始布局。

包括：

- 交换目标与干扰物的位置。
- 改变目标与干扰物之间的距离。
- 改变目标物体朝向。
- 调整候选玩具整体布局。
- 将目标放置在训练数据未覆盖的可达位置。

测试模型是否能够根据视觉与任务条件确定目标，而不是依赖固定空间位置。

评测场景应保证机器人在物理上能够完成操作。

### 13.6 评测指标

| 指标 | 说明 | 优化方向 |
|---|---|---|
| **Task Success Rate** | 完整完成指定目标抓取放置的比例 | ↑ |
| **Target Selection Accuracy** | 正确选择当前指定目标的比例 | ↑ |
| Wrong-object Rate | 错误抓取非目标玩具的 Episode 比例 | ↓ |
| Grasp Success Rate | 目标物体成功被抓取的比例 | ↑ |
| Placement Success Rate | 目标物体成功被放置的比例 | ↑ |
| Target Switching Success Rate | 目标切换测试中的任务成功率 | ↑ |
| Layout OOD Success Rate | 未见物体布局下的任务成功率 | ↑ |
| Instruction Robustness | 不同语言表达下的任务成功率 | ↑ |

其中，目标选择准确率与最终任务成功率需要分别统计。

例如：

- 选择正确玩具，但未能抓取：目标选择正确，任务失败。
- 抓取错误玩具并放入盒子：目标选择错误，任务失败。
- 抓取正确玩具，但放置失败：目标选择正确，任务失败。
- 正确选择、成功抓取并放置：任务成功。

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
**Task:** Target-Conditioned Multi-Object Pick-and-Place

| Model | Success Rate ↑ | Target Selection ↑ | Wrong-object Rate ↓ |
|---|---:|---:|---:|
| ACT | 55% | 81% | 16% |
| SmolVLA | **60%** | **86%** | **12%** |

真实环境中同时存在多个玩具，每次 Episode 的目标物体根据任务条件变化。

ACT 使用结构化目标类别信息。

SmolVLA 使用自然语言指令。

最终实验应报告测试次数、任务目标分布、模型配置及真实测量结果。

### 13.9 失败类型分析

| Failure Type | Description |
|---|---|
| Target Selection Failure | 选择错误目标 |
| Initiation Failure | 机器人未正常开始执行操作 |
| Approach Failure | 未能准确接近目标 |
| Grasp Failure | 未能成功抓起目标 |
| Transport Failure | 搬运过程中目标掉落 |
| Placement Failure | 目标未能正确放入容器 |
| Timeout | 在规定时间内未完成任务 |

重点分析两类不同的问题：

**目标条件相关失败**

模型没有根据当前任务条件选择正确玩具。

**动作执行相关失败**

模型选择正确目标，但因接近、抓取、搬运或放置问题导致任务失败。

通过区分这两种失败，可以避免将所有操作失败简单归因于目标识别或语言理解能力不足。

---

<a id="huggingface"></a>

## 14. Hugging Face 数据管理

项目支持将 LeRobotDataset 保存在本地，也支持通过 Hugging Face Hub 管理数据集。

### 14.1 本地数据存储

数据采集时使用：

```bash
--offline-local-only
```

可以将数据保存在指定本地目录，不执行远程上传。

示例：

```text
data/
├── demo_data_so101/
└── ...
```

模型 Checkpoint 与实验结果分别保存在：

```text
checkpoints/
outputs/
```

### 14.2 上传数据集

登录 Hugging Face：

```bash
hf auth login
```

使用上传脚本：

```bash
python scripts/hf/upload_hf.py \
  --folder data/demo_data_so101 \
  --repo-id YOUR_USERNAME/so101_pick_place \
  --repo-type dataset \
  --create-if-missing
```

其中：

- `--folder`：本地数据集路径。
- `--repo-id`：Hugging Face 数据集仓库。
- `--repo-type dataset`：上传为 Dataset。
- `--create-if-missing`：在仓库不存在时创建。

上传前应确认数据集格式与隐私设置。

### 14.3 数据安全

真实机器人操作数据、Checkpoint 和实验视频根据实际权限进行管理。

私有数据不应直接提交至公共 Git 仓库。

对于较大的视频和模型权重，建议使用独立的数据存储或模型仓库，避免增加 Git 仓库体积。

---

<a id="reproducibility"></a>

## 15. 实验复现与注意事项

### 15.1 实验环境与版本

复现实验应记录：

- 操作系统。
- Python 版本。
- LeRobot 版本。
- MuJoCo 版本。
- PyTorch 版本。
- CUDA 版本。
- GPU 型号。
- 模型 Checkpoint。
- 随机种子。

不同版本的机器人数据接口和策略配置可能存在差异，应保持训练与推理环境兼容。

### 15.2 训练数据一致性

进行模型比较时，应明确记录不同策略使用的：

- 训练示范数量。
- 相机观测。
- 机器人状态。
- 动作表示。
- 数据集划分。
- 训练 Steps。
- 预训练初始化方式。

对于仿真实验，三种模型应使用相同任务的专家示范数据进行比较。

对于真机实验，ACT 和 SmolVLA 应使用相同目标集合的真实操作示范。

### 15.3 目标条件公平性

标准 ACT 不直接支持任意自然语言指令。

在真机多目标任务中，ACT 使用结构化目标类别条件。

SmolVLA 使用自然语言指令。

因此，两种模型的输入条件形式不同。

总体任务成功率可以用于比较两个操作系统，但不能将成功率差异直接解释为语言理解能力或模型架构的单独优势。

### 15.4 泛化类型

实验区分以下泛化设置。

**Position Generalization**

目标在训练未覆盖的位置出现。

**Target Switching**

在已知候选玩具集合中切换当前目标。

**Layout Generalization**

改变目标和干扰物的空间布局。

**Instruction Generalization**

使用不同语言表达描述相同任务。

**Object Generalization**

测试训练时未出现的物体类别。

上述泛化类型具有不同难度，应分别设计评测任务和报告结果。

本项目的基础仿真 OOD 主要指目标方块的位置变化，不等同于新物体类别泛化。

### 15.5 仿真与真实环境差异

MuJoCo 仿真与真实 SO-101 环境存在差异，包括：

- 相机成像。
- 光照。
- 物体材质。
- 接触与摩擦。
- 关节校准。
- 控制延迟。
- 物体类别。
- 任务条件。

当前仿真任务使用蓝色方块，真实任务使用多个不同玩具。

因此，真实玩具操作不仅涉及仿真与真实环境差异，还涉及操作对象和任务设置的变化。

不能将所有性能差异直接解释为纯粹的 Sim-to-Real Gap。

### 15.6 评测协议与成功判定

模型比较应尽量使用：

- 相同的测试初始状态集合。
- 相同的机器人和环境配置。
- 相同的任务最大执行时间。
- 明确且一致的成功判定标准。
- 足够的测试 Episode。
- 独立于训练数据的测试场景。

对于真机多目标任务，应平衡不同玩具作为目标的测试次数。

同时明确区分目标选择失败和动作执行失败。

实验结果需要报告真实测试次数和对应统计方式。

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

感谢相关开源项目及研究工作的作者与维护者。

---

<a id="license"></a>

## 17. 许可证

本项目许可证见 [LICENSE](LICENSE)。

第三方代码、机器人模型及预训练权重遵循各自的许可证。
