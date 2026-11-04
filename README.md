<a id="lerobot--mujoco-act-pipeline"></a>
# LeRobot + MuJoCo + SO-101：ACT 仿真流程

> 中文说明与注释主要由机器翻译并部分校订，阅读时可参考[上游原文](https://github.com/EE5108-DigitalTwins/lerobot_mujoco_sim/tree/912f6f15d9b0052db9719a86c872b659191be716)。部分注释、Notebook 和配置仍为英文；上游演示视频及动画不收录在此仓库。

> **原创作品来源：**
> 该存储库基于[Jeongeun Park](https://github.com/jeongeun980906)的[lerobot-mujoco-tutorial](https://github.com/jeongeun980906/lerobot-mujoco-tutorial)。该分支重点关注 EE5108 迷你项目的**ACT（动作分块 Transformer）**管道。

在 MuJoCo 中收集演示数据，训练 ACT 策略，并将其部署在模拟中。单任务拾放（SO-101 臂，蓝色块 → 收纳盒）。

Mujoco 的LeRobot SO-101 手臂仿真

<a id="table-of-contents"></a>
## 目录

- [安装](#installation)
- [EE5108 项目流程](#ee5108-mini-project-workflow)
- [项目结构](#project-structure)
- [收集数据](#collect-data)
- [数据集从 v2.1 转换为 v3.0](#convert-dataset-from-v21-to-v30)
  - [本地转换](#local-conversion-disk-only)
  - [转换并推送至 Hugging Face Hub](#hugging-face-hub-convert-and-push)
- [回放数据](#playback-data)
- [训练 ACT](#train-act)
- [部署 ACT](#deploy-act)
- [机械臂配置](#arm-profiles--config)
- [上传至 Hugging Face](#upload-to-hugging-face)
- [致谢](#acknowledgements)
- [许可证](#license)

<a id="installation"></a>
## 安装

推荐使用Python 3.12。 Docker 是最简单的选择。

<a id="docker-recommended"></a>
### Docker（推荐）

两个镜像：

- **运行时**（`Dockerfile.runtime`）：数据收集+策略部署（较小，没有Jupyter）。
- **Train/dev**(`Dockerfile`)：与运行时 + Jupyter 相同，用于本地训练。

从`lerobot_mujoco_sim/`构建：

```bash
docker build -t lerobot_mujoco_sim:runtime -f Dockerfile.runtime .
docker build -t lerobot_mujoco_sim:train -f Dockerfile .
```

运行（GPU）：

```bash
docker run --rm -it --gpus all -v "$PWD:/workspace" lerobot_mujoco_sim:runtime
```

容器内部：运行 `cd /workspace`（上述命令将仓库根目录挂载到 `/workspace`）。

使用 docker-compose：

```bash
docker compose up -d lerobot-runtime
docker compose exec lerobot-runtime bash
cd /workspace
```

<a id="so-101--so-100-arm-assets"></a>
## SO-101 / SO-100 机械臂资源

- 随仓库提供：`third_party/SO-ARM100`
- MuJoCo 资源：`asset/so_arm100/SO101/`、`asset/so_arm100/SO100/`
- ACT使用的场景：`asset/scene_so101_y.xml`（SO-101，蓝色块→bin）

<a id="ee5108-mini-project-workflow"></a>
## EE5108 小型项目工作流程

端到端流程：捕获数据→上传→训练ACT（例如Colab）→部署在MuJoCo中。

<a id="1-build-and-start-runtime"></a>
### 1) 构建并启动运行时

从主机上的存储库根目录：

```bash
docker build -t lerobot_mujoco_sim:runtime -f Dockerfile.runtime .
docker run --rm -it --gpus all -v "$PWD:/workspace" lerobot_mujoco_sim:runtime
```

然后在容器内运行 `cd /workspace`。

<a id="2-capture-data-so-101"></a>
### 2）捕获数据（SO-101）

**手动遥控**（推荐第一次）：

```bash
python scripts/collect/manual_collect_data.py \
  --env-robot-profile so101 \
  --num-demo 20 \
  --repo-name <your-dataset-repo> \
  --root data/demo_data_so101 \
  --offline-local-only
```

**脚本化批处理**（Mink FSM，无 teleop）：

```bash
python scripts/collect/batch_collect_data.py \
  --env-robot-profile so101 \
  --num-demo 200 \
  --repo-name <your-dataset-repo> \
  --root data/demo_data_so101 \
  --offline-local-only
```

两者都在`data/demo_data_so101`下写入 LeRobot 风格的数据集（如果目录已存在并且设置了`--offline-local-only`，则为`_fresh_*`变体）。

Docker 和本地安装目标为**LeRobot 0.5.x**，因此新数据集使用**`codebase_version`v3.0**。使用该 API，`LeRobotDataset.add_frame`仅需要一个帧字典，并且每一步都需要一个**`task`**字段（收集器从配置中传递您的`task_name`）。录制后调用**`dataset.finalize()`**，以便刷新 v3 parquet/元数据；收集器脚本在退出时执行此操作，并且收集笔记本在关闭之前以`finalize()`关闭。

如果您有**较旧的磁盘数据集**，其`meta/info.json`仍显示**`v2.1`**，请在使用当前堆栈进行训练或部署之前对其进行转换。参见[数据集从 v2.1 转换为 v3.0](#convert-dataset-from-v21-to-v30)。

<a id="3-upload-dataset-to-hugging-face"></a>
### 3）将数据集上传到Hugging Face

您需要一个 Huggingface 帐户和一个访问令牌来创建数据集。
登录（其中之一）：

```bash
python -c "from huggingface_hub import login; login(token='hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')"
```

上传：

```bash
python scripts/hf/upload_hf.py \
  --folder data/demo_data_so101 \
  --repo-id <your-hf-username>/<your-dataset-repo> \
  --repo-type dataset \
  --create-if-missing \
  --message "Upload SO101 dataset from demo_data_so101"
```

使用`--private`作为私人仓库。

<a id="4-train-act"></a>
### 4) 训练 ACT

- 使用 Google Colab（推荐）并运行笔记本`notebooks/EE5108_training_act.ipynb`。
- 在笔记本中，将`dataset.repo_id`设置为`<your-hf-username>/<your-dataset-repo>`（您在步骤 3 中上传的数据集）。
- Notebook训练ACT检查点并将`deploy_metadata.json`写入检查点目录以方便部署。

<a id="5-deploy-in-mujoco"></a>
### 5）在MuJoCo中部署

复制经过训练的检查点（例如复制到`checkpoints/act_y/`中），然后：

```bash
python scripts/deploy/deploy_act.py --checkpoint checkpoints/act_y
```

配置文件驱动的部署（推荐）：

```bash
python scripts/deploy/deploy_act.py --config configs/deploy_act.yaml
```

仅 CPU：

```bash
python scripts/deploy/deploy_act.py --checkpoint checkpoints/act_y --device cpu
```

<a id="project-structure"></a>
## 项目结构

```
lerobot_mujoco_sim/
├── README.md
├── Dockerfile              # Train/dev image (Jupyter + deps)
├── Dockerfile.runtime      # Runtime image (collect + deploy)
├── docker-compose.yml
├── asset/
│   ├── scene_so101_y.xml   # SO-101 pick-and-place scene
│   ├── so_arm100/          # SO-101/SO-100 MuJoCo assets
│   └── tabletop/
├── configs/
│   ├── collect_batch.yaml
│   ├── collect_manual.yaml
│   └── collect_data.yaml
├── scripts/
│   ├── collect/            # manual_collect_data.py, batch_collect_data.py
│   ├── deploy/             # deploy_act.py
│   ├── train/              # train_act.py
│   ├── visualize/          # visualize_data.py
│   ├── convert_lerobot_dataset_v21_to_v30_local.py  # v2.1 → v3.0 migration wrapper
│   └── hf/                 # upload_hf.py
├── mujoco_env/             # MuJoCo environment
├── controllers/            # Mink FSM for batch collection
├── so101/                  # SO-101 kinematics
└── third_party/SO-ARM100/
```

数据和检查点是本地的（例如`data/`、`checkpoints/`）并且通常不提交。

<a id="collect-data"></a>
## 收集数据

默认任务：挑选蓝色块，放入收纳盒。场景：`asset/scene_so101_y.xml`。

<a id="manual-collection-keyboard"></a>
### 手动采集（键盘）

- **WASD**+**R/F**+ 箭头：联合控制（参见[Keyboard controls](#keyboard-controls-so-101)）。
- **SPACE**：肘节夹具。**Z**：重置并放弃当前剧集。
- 从第一乐章开始录音；成功完成一次拾取和放置以保存剧集。

配置和 YAML：

```bash
python scripts/collect/manual_collect_data.py --config configs/collect_manual.yaml
python scripts/collect/batch_collect_data.py --config configs/collect_batch.yaml --num-demo 100
```

<a id="keyboard-controls-so-101"></a>
### 键盘控制 (SO-101)


| Key   | Action                  |
| ----- | ----------------------- |
| A / D | Shoulder pan            |
| W / S | Shoulder lift           |
| R / F | Elbow flex              |
| ↑ / ↓ | Wrist flex              |
| ← / → | Wrist roll              |
| SPACE | Toggle gripper          |
| Z     | Reset & discard episode |


<a id="scripted-batch"></a>
### 脚本化批处理

`batch_collect_data.py`使用基于 Mink 的 FSM 来生成演示，无需 teleop。与手动采集相同的数据集格式。

<a id="convert-dataset-from-v21-to-v30"></a>
## 将数据集从 v2.1 转换为 v3.0

当`meta/info.json`仍然显示**`"codebase_version": "v2.1"`**（旧版 LeRobot / 磁盘布局）并且您想要**LeRobot 0.5.x**和**v3.0**数据集时，请使用此选项此存储库使用的格式。

**内部实现：**[Hugging Face LeRobot](https://github.com/huggingface/lerobot)`convert_dataset`(`lerobot.scripts.convert_dataset_v21_to_v30`)。此存储库提供了**`scripts/convert_lerobot_dataset_v21_to_v30_local.py`**，一个包装器：

- 运行与上游相同的转换。
- **当 episode 的 Parquet 文件**存储为四个`(3,)`数组的 NumPy“对象向量”时，修复`spawn.block_xyz`（该布局在分片期间破坏了`pyarrow.Schema.from_pandas`）。在转换之前，单元格被重写为嵌套列表。如果您知道您的Parquet 文件已经安全，请使用 `--no-repair-parquet` 跳过修复。

您可以通过两种方式使用该工具：**仅在磁盘**（默认）上，或**磁盘 + 上传到 Hugging Face Hub**。

<a id="prerequisites-on-disk-typical-v21-tree"></a>
### 磁盘先决条件（典型的 v2.1 树）

- `meta/info.json`与`codebase_version`**v2.1**
- `meta/episodes.jsonl`, `meta/tasks.jsonl`, `meta/episodes_stats.jsonl`
- `data/`（如果运行使用编码视频，则为`videos/`）

`meta/stats.json`是**，不需要**（全局统计数据是从`episodes_stats.jsonl`重新计算的）。

<a id="--root-vs---repo-id-read-this-once"></a>
### `--root`与`--repo-id`（读一次）

| Argument | Meaning |
|----------|--------|
| **`--root`** | **Filesystem path** to the dataset: the folder that directly contains `meta/` and `data/` (not the git repo root unless your dataset lives there). |
| **`--repo-id`** | **Not a path.** It is the Hugging Face **dataset repo id** (`user-or-org/dataset-name`) when pushing, or a **placeholder** such as `local/my_run` for local-only conversion (default: `local/<folder_name>` if omitted). |

**Docker:**如果您在`/workspace`挂载此项目，则数据集通常是`/workspace/data/<dataset_name>`、**而不是**`/workspace/lerobot_mujoco_sim/data/...`，除非您显式挂载该嵌套路径。

<a id="environment"></a>
### 环境

使用**Python 3.12**和**`lerobot==0.5.1`**（或另一个包含 v2.1→v3.0 脚本的 0.5.x）：主机上的 venv，此存储库的映像（如果已具有`lerobot`）或 Docker 单行代码下面。

<a id="local-conversion-disk-only"></a>
### 本地转换（仅限磁盘）

Goal: turn the folder at `--root` into a **v3.0** dataset **on your machine**. Nothing is uploaded; **no Hugging Face login** is required.

从仓库根目录：

```bash
pip install "lerobot==0.5.1"   # in a venv if your system Python is PEP 668–managed

python scripts/convert_lerobot_dataset_v21_to_v30_local.py \
  --root /absolute/or/relative/path/to/my_v21_dataset \
  --repo-id local/my_v21_dataset
```

如果省略`--repo-id`，则默认为`local/<folder_name>`。

**文件会发生什么情况：**上游在数据集旁边写入 v3.0 树，将原始 v2.1 树移动到**`{dataset_name}_old`**（在`--root`旁边），然后将转换后的树**移动到**原始`--root`路径。您的训练路径保持不变；只是格式改变了。

**检查：**

```bash
python -c "import json; print(json.load(open(\"path/to/my_v21_dataset/meta/info.json\"))[\"codebase_version\"])"
# expect: v3.0
```

可选：加载`LeRobotDataset`并读取一步以确认。

<a id="hugging-face-hub-convert-and-push"></a>
### Hugging Face Hub：转换并推送

目标：在磁盘上进行相同的**就地**转换，然后**将 v3.0 数据集**上传到 Hub**数据集**存储库，以便其他人（或来自 Colab 的您）可以使用`repo_id=user/dataset`加载它。

**运行**之前

1. **在Hub上创建一个空数据集存储库**（例如[New dataset](https://huggingface.co/new-dataset)），例如`myuser/so101_pick_place_v30`。您需要**写入**访问权限。
2. **验证：**`huggingface-cli login`，或导出`HF_TOKEN`并有权推送到该存储库。

**命令**（真实Hub ID +`--push-to-hub`）：

```bash
python scripts/convert_lerobot_dataset_v21_to_v30_local.py \
  --root /path/to/my_v21_dataset \
  --repo-id myuser/so101_pick_place_v30 \
  --push-to-hub
```

可选：如果您的工作流程使用非默认分支，则`--branch main`（或其他分支）。

**重要提示：**转换**仍重写`--root`**处的文件夹（原始 v2.1 已移至其旁边的`*_old`）。如果必须**保持原始路径中的 v2.1 树**不变，则**将数据集**复制到另一个目录，并将该副本作为`--root`传递。

**仅与上游**相同（此存储库中没有`spawn.block_xyz`修复）：

```bash
python -m lerobot.scripts.convert_dataset_v21_to_v30 \
  --repo-id=myuser/so101_pick_place_v30 \
  --root=/absolute/path/to/my_v21_dataset \
  --push-to-hub=true
```

如果该模块路径丢失，请从 PyPI 安装`lerobot`。更喜欢此存储库的包装器来处理此处收集的数据集，因此Parquet 文件修复会自动运行。

对于磁盘**上已为**v3.0 的数据集，并且您只想上传它，请参阅[上传至 Hugging Face](#upload-to-hugging-face)(`scripts/hf/upload_hf.py`)。

<a id="docker-one-off-local-conversion-example"></a>
### Docker（一次性、本地转换示例）

```bash
docker run --rm -it \
  -v /path/to/lerobot_mujoco_sim:/workspace \
  -w /workspace \
  python:3.12-bookworm bash -lc '
    pip install -q "lerobot==0.5.1" &&
    python scripts/convert_lerobot_dataset_v21_to_v30_local.py \
      --root /workspace/data/my_v21_dataset \
      --repo-id local/my_v21_dataset
  '
```

在容器内添加 Hugging Face 凭据（`huggingface-cli login`或`-e HF_TOKEN=...`）并附加`--push-to-hub --repo-id youruser/your-dataset`以进行Hub推送。

<a id="wrapper-flags-quick-reference"></a>
### 包装标志（快速参考）

- **`--push-to-hub`**— 转换后，推送到名为`--repo-id`的数据集存储库。
- **`--branch`**— Hub分支（上游默认值：`main`）。
- **`--data-file-size-in-mb`**、**`--video-file-size-in-mb`**— Parquet 文件/视频的分片。
- **`--force-conversion`**— 如果 Hub v3 快照已存在，则跳过上游防护（谨慎使用）。
- **`--no-repair-parquet`**— 不要在 parquet 中重写`spawn.block_xyz`。
- **`--skip-preflight`**、**`--no-progress`**、**`--no-logging-setup`**— 高级/CI。

<a id="caveats"></a>
### 注意事项

- 不寻常的数据集布局可能会在旧版 LeRobot 版本中遇到边缘情况；尽可能使用**`lerobot==0.5.1`**。
- **v2.0 → v3.0**通常不是单跳；如果需要，请首先使用旧工具迁移到 v2.1 ([LeRobot discussions](https://github.com/huggingface/lerobot))。

<a id="playback-data"></a>
## 回放数据

重播 MuJoCo 中保存的剧集：

```bash
python scripts/visualize/visualize_data.py
```

或者使用笔记本`notebooks/2.visualize_data.ipynb`。

<a id="train-act"></a>
## 训练 ACT

推荐（学生）：运行 Colab 笔记本`notebooks/EE5108_training_act.ipynb`。

**Hub 令牌：**如果笔记本可能被共享或签入 git，请勿将 Hugging Face 访问令牌粘贴到笔记本源中。优先选择您计算机上的`HF_TOKEN`（或 Colab Secrets）、`huggingface-cli login`或[部署 ACT](#deploy-act)中描述的`token.txt`模式。培训笔记本从环境中读取凭证。

最小本地训练示例（如果您已经设置了环境）：

```bash
python scripts/train/train_act.py
```

检查点写入`checkpoints/act_y/`（包括用于部署的`deploy_metadata.json`）。对于完整的训练（例如 Colab），请使用 LeRobot 管道和 Hugging Face 数据集。

<a id="deploy-act"></a>
## 部署 ACT

使用经过训练的 ACT 检查点在同一 SO-101 MuJoCo 场景中运行策略。

```bash
python scripts/deploy/deploy_act.py --checkpoint checkpoints/act_y
```

`scripts/deploy/deploy_act.py`从检查点的`config.json`构建 ACT 策略，然后加载数据集规范化统计信息（需要正确对操作输出进行非规范化）。

数据集统计信息来自以下位置之一：

- 如果`checkpoints/<your-checkpoint>/deploy_metadata.json`存在，则`deploy_act.py`直接使用它。
- 否则，它会在`--dataset-root`下查找本地数据集元数据。
- 它期望`meta/info.json`（包括`features`）。
- 它期望`meta/stats.json`（标准化统计）。
- 如果这些本地文件丢失，它仅从`--dataset-repo-id`给出的 Hugging Face**数据集**存储库下载`meta/info.json`和`meta/stats.json`。

注释/陷阱：

- 该存储库的部署脚本是**MuJoCo sim-only**。请勿将`lerobot-record`用于此工作流程（它针对物理机器人并需要硬件`--robot.port`）。
- 确保`--dataset-repo-id`指向 Hugging Face *dataset* 存储库（使用`--repo-type dataset`上传），而不是模型/策略存储库。

常用命令（本地数据集统计）：

```bash
python scripts/deploy/deploy_act.py \
  --checkpoint checkpoints/act_y \
  --dataset-root data/demo_data_so101
```

<a id="using-the-yaml-config"></a>
### 使用 YAML 配置

在`configs/deploy_act.yaml`中实时部署的生成边界和运行时默认值。

```bash
python scripts/deploy/deploy_act.py --config configs/deploy_act.yaml
```

CLI 标志仍然会覆盖 YAML 值。选项包括：`--xml-path`、`--device cpu`、`--seed`、`--spawn-x-min`等。参见`--help`。

<a id="important-flags-most-used"></a>
### 重要标志（最常用）

运行`python scripts/deploy/deploy_act.py --help`以获得完整列表，但以下是关键列表：

- `--checkpoint`：检查点文件夹至少包含`config.json`（以及可选的`deploy_metadata.json`）
- `--dataset-root`：本地数据集文件夹（默认在 YAML 中设置）。预计包含`meta/info.json`和`meta/stats.json`。
- `--dataset-repo-id`：仅当本地`meta/`文件不存在时才使用 Hugging Face 数据集存储库 ID
- `--xml-path`：MuJoCo 场景 XML

<a id="if-your-dataset-is-private"></a>
### 如果您的数据集是私有的

在环境中设置`HF_TOKEN`，或者（当使用提供的 docker-compose 运行时时）在存储库根目录的`token.txt`中放置一个令牌，以便容器可以导出它。

<a id="arm-profiles--config"></a>
## 手臂配置文件和配置

配置文件：`so101`（默认）、`so100`、`omy`。默认值来自`configs/collect_data.yaml`（例如`env_robot_profile: so101`、`xml_path: ./asset/scene_so101_y.xml`）。在运行时覆盖：

```bash
python scripts/collect/manual_collect_data.py --env-robot-profile so100
```

生成边界和场景设置位于同一配置中。

<a id="upload-to-hugging-face"></a>
## 上传至 Hugging Face

在**v3.0**布局中收集数据后（或者转换 v2.1 → v3.0 后；请参阅[数据集从 v2.1 转换为 v3.0](#convert-dataset-from-v21-to-v30)）：

```bash
huggingface-cli login   # or set HF_TOKEN
python scripts/hf/upload_hf.py \
  --folder data/demo_data_so101 \
  --repo-id <your-hf-username>/<your-dataset-repo> \
  --repo-type dataset \
  --create-if-missing
```

如果您的磁盘数据集仍然是**v2.1**，请使用转换部分的**Hugging Face Hub**流（`--push-to-hub`和`convert_lerobot_dataset_v21_to_v30_local.py`），而不是使用此脚本上传原始 v2.1 ，除非您有意希望在 Hub 上使用旧格式。

如果将 docker-compose 与`HF_HUB_OFFLINE=1`一起使用，请取消上传设置。

<a id="acknowledgements"></a>
## 致谢

- 原始教程：[lerobot-mujoco-tutorial](https://github.com/jeongeun980906/lerobot-mujoco-tutorial)作者：[Jeongeun Park](https://github.com/jeongeun980906)。
- SO-ARM 资产：[TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100)。
- LeRobot：[huggingface/lerobot](https://github.com/huggingface/lerobot)。
- MuJoCo 解析器灵感：[yet-another-mujoco-tutorial](https://github.com/sjchoi86/yet-another-mujoco-tutorial-v3)。

<a id="license"></a>
## 许可证

参见[LICENSE](LICENSE)。第三方组件（例如SO-ARM100、LeRobot）有自己的许可证。