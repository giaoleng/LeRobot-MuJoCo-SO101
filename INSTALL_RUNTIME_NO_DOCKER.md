<a id="install-without-docker-runtime-profile"></a>
# 不使用 Docker 安装（运行时配置文件）

本指南将项目直接安装在您的主机上，匹配`Dockerfile.runtime`设置（捕获 + 推理，无 Jupyter/dev 堆栈）。

<a id="1-host-requirements"></a>
## 1）主机要求

- 操作系统：Ubuntu 24.04（推荐，与运行时镜像库相同）
- 蟒蛇：3.12
- 用于 EGL 渲染/推理加速的可选 GPU
- Git + 用于 Python 依赖项和 LeRobot 安装的互联网访问

如果您使用不同的发行版/版本，包名称可能会有所不同。

<a id="2-install-system-packages"></a>
## 2）安装系统包

从`lerobot_mujoco_sim/`（或任何地方）运行：

```bash
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
  ca-certificates \
  curl \
  git \
  git-lfs \
  build-essential \
  python3.12 \
  python3.12-venv \
  python3.12-dev \
  python3-pip \
  ffmpeg \
  unzip \
  libgl1 \
  libglib2.0-0 \
  libglfw3 \
  libglew2.2 \
  libosmesa6 \
  libegl1 \
  libx11-6 \
  libxext6 \
  libxrender1 \
  libxrandr2 \
  libxinerama1 \
  libxi6 \
  libxcursor1 \
  libxkbcommon0 \
  libsm6 \
  libxmu6
```

初始化一次 Git LFS：

```bash
git lfs install
```

<a id="3-create-and-activate-virtual-environment"></a>
## 3）创建并激活虚拟环境

在仓库根目录（`lerobot_mujoco_sim/`）中：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

<a id="4-install-runtime-python-dependencies"></a>
## 4) 安装运行时Python依赖项

安装 PyTorch CPU 轮（与运行时配置文件相同）：

```bash
python -m pip install \
  torch==2.7.0 \
  torchvision==0.22.0 \
  --index-url https://download.pytorch.org/whl/cpu
```

安装 MuJoCo + 运行时库：

```bash
python -m pip install \
  mujoco==3.1.6 \
  numpy \
  Pillow \
  glfw \
  termcolor \
  PyYAML \
  mink
```

安装LeRobot（与`Dockerfile.runtime`引脚相同；数据集格式 v3.0）：

```bash
python -m pip install "lerobot==0.5.1"
```

<a id="5-optional-install-hugging-face-cli"></a>
## 5) （可选）安装 Hugging Face CLI

对于登录/上传工作流程：

```bash
curl -LsSf https://hf.co/cli/install.sh | bash
```

然后启动一个新的 shell（或`source ~/.bashrc`）并运行：

```bash
huggingface-cli login
```

<a id="6-project-specific-setup"></a>
## 6) 项目特定设置

如果对象资产存档存在，则将其解压：

```bash
if [ -f asset/objaverse/plate_11.zip ]; then
  unzip -o asset/objaverse/plate_11.zip -d asset/objaverse/
fi
```

设置运行时环境默认值：

```bash
export MUJOCO_GL=egl
```

如果您使用 NVIDIA GPU 和 EGL，请确保主机 GPU 驱动程序已安装并正常工作。

<a id="7-smoke-tests"></a>
## 7) 冒烟测试

检查进口：

```bash
python -c "import mujoco, torch, lerobot; print('ok')"
```

运行项目脚本之一（示例）：

```bash
python scripts/collect/manual_collect_data.py --help
python scripts/deploy/deploy_act.py --help
```

<a id="8-typical-runtime-usage-no-docker"></a>
## 8) 典型运行时使用（无 Docker）

从 repo root 激活 venv：

```bash
export MUJOCO_GL=egl
python scripts/collect/manual_collect_data.py \
  --env-robot-profile so101 \
  --num-demo 20 \
  --repo-name <your-dataset-repo> \
  --root data/demo_data_so101 \
  --offline-local-only
```

部署经过训练的 ACT 检查点：

```bash
python scripts/deploy/deploy_act.py --config configs/deploy_act.yaml
```

<a id="troubleshooting"></a>
## 故障排除

- `ModuleNotFoundError`：确保您的 venv 已激活（`which python`应指向`.venv`）。
- MuJoCo 渲染/上下文错误：确认`MUJOCO_GL=egl`；尝试使用`MUJOCO_GL=osmesa`作为无头 CPU 渲染的后备。
- `evdev`/native 轮子的构建错误：确保安装了`build-essential`和`python3.12-dev`。
- 未找到`huggingface-cli`：重新启动 shell 或将`~/.local/bin`添加到`PATH`。

