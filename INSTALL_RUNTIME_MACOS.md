<a id="install-on-macos-runtime-profile-no-docker"></a>
# 在 macOS 上安装（运行时配置文件，无 Docker）

本指南直接在 macOS（无 Docker）上安装运行时工作流程，改编自 Linux`Dockerfile.runtime`配置文件。

范围：数据收集+推理/部署脚本。
不包括：完整的开发/训练 Docker 工具链。

<a id="1-supported-macos-setup"></a>
## 1) 支持的 macOS 设置

- 推荐 macOS 13+
- 自制软件已安装
- Python 3.12
- 适用于 Apple Silicon (M1/M2/M3) 和 Intel Mac

<a id="2-install-system-prerequisites-homebrew"></a>
## 2）安装系统必备软件（Homebrew）

```bash
brew update
brew install \
  python@3.12 \
  git \
  git-lfs \
  ffmpeg \
  glfw \
  cmake
```

初始化一次 Git LFS：

```bash
git lfs install
```

<a id="3-create-and-activate-a-virtual-environment"></a>
## 3）创建并激活虚拟环境

从仓库根目录（`lerobot_mujoco_sim/`）：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

<a id="4-install-runtime-python-dependencies"></a>
## 4) 安装运行时Python依赖项

与 Linux 运行时 Docker 不同，macOS 应使用 PyTorch 的默认 PyPI 索引。

```bash
python -m pip install \
  torch==2.7.0 \
  torchvision==0.22.0
```

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

<a id="5-configure-mujoco-rendering-on-macos"></a>
## 5) 在 macOS 上配置 MuJoCo 渲染

在 macOS 上使用 GLFW（不是 EGL/OSMesa）：

```bash
export MUJOCO_GL=glfw
```

要坚持它：

```bash
echo 'export MUJOCO_GL=glfw' >> ~/.zshrc
```

<a id="6-optional-hugging-face-cli"></a>
## 6) 可选的Hugging Face CLI

如果您需要数据集上传/登录：

```bash
curl -LsSf https://hf.co/cli/install.sh | bash
```

然后打开一个新的shell并登录：

```bash
huggingface-cli login
```

<a id="7-project-asset-prep"></a>
## 7）项目资产准备

如果存档存在，则解压：

```bash
if [ -f asset/objaverse/plate_11.zip ]; then
  unzip -o asset/objaverse/plate_11.zip -d asset/objaverse/
fi
```

<a id="8-verify-installation"></a>
## 8) 验证安装

```bash
python -c "import mujoco, torch, lerobot; print('ok')"
python scripts/collect/manual_collect_data.py --help
python scripts/deploy/deploy_act.py --help
```

<a id="9-typical-runtime-usage"></a>
## 9) 典型运行时使用

收集演示：

```bash
export MUJOCO_GL=glfw
python scripts/collect/manual_collect_data.py \
  --env-robot-profile so101 \
  --num-demo 20 \
  --repo-name <your-dataset-repo> \
  --root data/demo_data_so101 \
  --offline-local-only
```

部署 ACT：

```bash
python scripts/deploy/deploy_act.py --config configs/deploy_act.yaml
```

<a id="troubleshooting-macos"></a>
## 故障排除 (macOS)

- 如果未找到`python3.12`，请运行`brew info python@3.12`并将brew Python 添加到`PATH`。
- 如果 MuJoCo 上下文创建失败，请在同一 shell 会话中确认`MUJOCO_GL=glfw`。
- 如果旧款 Mac 上出现`opencv-python`滚轮问题，请尝试：
  - `python -m pip install opencv-python-headless`
- 如果`lerobot`安装由于仅 Linux 依赖项（例如`evdev`周围）而失败，则这是上游打包兼容性问题；更喜欢 Linux 运行时，以便与 Docker 严格同等。
- 在 Apple Silicon 上，将所有安装保留在一个 arm64 shell/venv 中（`uname -m`应该是`arm64`）以避免混合拱形轮问题。

