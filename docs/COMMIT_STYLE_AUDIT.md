# 提交风格与首页目录审查

审查日期：2026-11-08。仅检查本仓库；通过GitHub API确认远程main为 `db9166d691233b92f30d5a11c154a0e7117682ee`，与本地审查起点一致。以下路径说明使用该固定版本的 `git log -1 -- <path>` 追溯，代表当前GitHub main文件列表相关提交的可核对依据；不是对未来显示的强制设置。

## 现状

旧标题混用中文、英文、带scope与不带scope形式。docs/scripts/tests部分共同显示同一条跨目录功能提交；这是正常Git历史结果。README标题和现有相对文件链接未发现需要本轮修复的问题。

两仓库都有先前阶段未提交的工程工作。本轮不提交这些修改，不改变算法，不改许可证，不安装hook，不推送。

## 目录职责与建议说明

下表最后一列只在对应真实修改发生时采用。修复bug时应改用fix并描述具体问题，不能仅因目录对应feat就一律使用feat。许可证已存在，不再为其创建“添加许可证”的提交。

| 路径 | 模块职责 | 当前路径最近提交 | 后续提交参考 |
|---|---|---|---|
| `asset` | 机器人MJCF、场景与网格 | `c938858 chore(sim): 导入 SO-101 机械臂模型与 MuJoCo 场景` | `feat: 完善SO-101机械臂模型` |
| `configs` | 采集、仿真与部署配置 | `ec0095a chore(config): 整理仿真与模型训练配置` | `chore: 统一仿真与部署配置` |
| `controllers` | Mink IK与有限状态控制器 | `0e3de26 chore(control): 整理机械臂运动控制器` | `feat: 完善关节运动控制` |
| `docs` | 算法、安装与部署文档 | `22bcf18 feat: add ACT, Diffusion Policy and SmolVLA support` | `docs: 更新算法与部署文档` |
| `media` | 截图和仿真演示 | `f833e54 docs(demo): 添加 SO-101 MuJoCo 抓取放置仿真演示` | `docs: 添加MuJoCo仿真演示` |
| `mujoco_env` | MuJoCo环境、状态与动作交互 | `d30123e chore(env): 整理 MuJoCo 仿真环境接口` | `feat: 完善MuJoCo环境交互` |
| `notebooks` | 数据采集与模型训练Notebook | `de9a449 docs(notebook): 整理 ACT 训练 Notebook` | `docs: 整理模型训练Notebook` |
| `scripts` | 采集、训练、推理与评测入口 | `22bcf18 feat: add ACT, Diffusion Policy and SmolVLA support` | `feat: 完善策略训练与推理脚本` |
| `so101` | 几何FK/IK与仿真辅助接口 | `2a275d2 chore(core): 整理 SO-101 核心模块与第三方依赖` | `feat: 完善SO-101运动学接口` |
| `tests` | 评测聚合及控制接口回归 | `22bcf18 feat: add ACT, Diffusion Policy and SmolVLA support` | `test: 完善机器人与评测接口测试` |
| `third_party/SO-ARM100` | 机器人模型与参考资产 | `2a275d2 chore(core): 整理 SO-101 核心模块与第三方依赖` | `chore: 整理机器人模型依赖` |
| `.dockerignore` | 容器构建上下文排除规则 | `105aadd chore: 整理项目基础配置与许可证` | `build: 更新容器构建忽略规则` |
| `.gitignore` | 忽略数据、权重与运行输出 | `105aadd chore: 整理项目基础配置与许可证` | `chore: 更新文件忽略规则` |
| `AGENTS.md` | 开发与提交边界 | `105aadd chore: 整理项目基础配置与许可证` | `docs: 规范机器人项目开发流程` |
| `Dockerfile` | 开发环境镜像 | `047488b build: 整理 Docker 与容器运行配置` | `build: 完善机器人开发镜像` |
| `Dockerfile.runtime` | 推理与采集运行镜像 | `047488b build: 整理 Docker 与容器运行配置` | `build: 完善机器人运行镜像` |
| `docker-compose.yml` | 容器服务与设备挂载 | `047488b build: 整理 Docker 与容器运行配置` | `build: 完善机器人容器配置` |
| `INSTALL_RUNTIME_MACOS.md` | macOS运行环境安装说明 | `92a3f03 docs: 整理项目安装说明与中文教程` | `docs: 更新macOS运行环境指南` |
| `INSTALL_RUNTIME_NO_DOCKER.md` | Linux本机依赖安装说明 | `92a3f03 docs: 整理项目安装说明与中文教程` | `docs: 更新Linux运行环境指南` |
| `LICENSE` | 项目许可 | `105aadd chore: 整理项目基础配置与许可证` | `chore: 更新项目许可证` |
| `README.md` | 项目首页与运行指引 | `db9166d Remove simulated data disclaimer from README` | `docs: 完善项目使用指南` |
| `README.txt` | 兼容文本版使用说明 | `22bcf18 feat: add ACT, Diffusion Policy and SmolVLA support` | `docs: 同步项目文本使用说明` |

## 本轮安全实施

- 新增根目录 `CONTRIBUTING.md`：统一七种类型、中文标题、原子提交、暂存检查和历史约束。
- 新增根目录 `.gitmessage`：提供中文技术标题编辑模板，并在此工作副本设置local commit.template；不影响全局配置。
- 新增本文：记录实际路径说明、职责、未来用语和历史整理边界。

## GitHub首页预期

本轮没有push，GitHub当前页面不会变化。日后正常推送这些本地提交后，新增CONTRIBUTING.md和.gitmessage显示规范配置相关提交，docs显示本轮审查文档提交。README保持原提交；没有为刷新它而修改正文。

其他未提交修改的目录继续显示旧提交标题。不会承诺每个目录立即全部变为新风格。目录自身的真实后续提交会逐步更新显示；同一个跨目录提交也可能仍让若干路径显示同一标题。

## 是否需要改写历史

采用新规范不需要历史改写，因此本轮选择只新增真实修改的提交。若要求本页上表所有旧标题立即变更，则需要改写这些旧提交及其后代，这属于另一项任务；此处只给出方案，不执行，也不默认请求执行。

待单独审批的具体方案：

1. 以本页固定main SHA为旧基线，使用上表的路径提交建立“完整旧SHA—旧标题—新标题”映射；只映射要修改的标题，不改补丁内容。
2. 先保存本地未提交工作，在独立工作副本操作；为原main建立备份引用和Git bundle，并记录远程当前HEAD。
3. 从最早目标提交开始保留merge拓扑，仅替换message，保留tree和作者信息；逐提交核对tree SHA、文件数量、许可证、最终tree不变。
4. 列出所有被重建提交及新旧SHA；原签名、PR引用、协作者分支可能受影响，必须提前协调。
5. 先交付本地候选分支和核对结果，再决定远程迁移方式。已有共享分支上替换历史会涉及非快进更新，与本轮禁止强推的限制冲突；本轮不进行该操作。

不采用空提交、空白修改、先删后加、无意义重命名来规避上述限制。

## 既有工程修改的后续分组建议

| 实际工作 | 建议标题 |
|---|---|
| 控制目标采集与时钟修复 | `fix: 修复采集动作与控制时钟` |
| 真机部署与类别条件入口 | `feat: 增加SO-101真机策略接口` |
| ACT评测JSONL与汇总 | `feat: 增加ACT逐回合评测记录` |
| 参考硬件配置 | `chore: 补充SO-101参考配置` |
| 离线接口测试 | `test: 覆盖控制与真机配置接口` |

这些分组尚未提交；需先按依赖和真实diff审阅，不能把功能、配置、测试强行拆成不可用的中间状态。
