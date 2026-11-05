# 本地 Codex 集成任务（适用于本仓库，不要在本机训练）

目标：将当前以 ACT 为主的 SO-101 MuJoCo 项目扩展为 ACT / Diffusion Policy / SmolVLA 三策略实验仓库，同时保留原来的专家脚本、EE5108、数据转换、回放及 ACT 使用流程。用户不允许把私有数据、权重、轨迹或实验日志上传到公共服务。

1. 首先完整检查 README.md、现有 tracked 文件、scripts/、controllers/、mujoco_env/、LeRobot 版本、数据集 features 与现有 ACT 评测方式；不要猜测已有 API。保持 Git 工作区现有未跟踪文件，不能删除或覆盖。
2. 阅读本增量包 scripts/train/ 和 docs/，把 DP 和 SmolVLA 训练入口融合到实际仓库，不重写官方模型结构。训练执行只在 BitaHub RTX 3090，允许本地 shell -n、单测、CPU 轻量静态检查，禁止本地加载大模型、下载大权重、训练或大量 rollouts。
3. 基于真实 mujoco_env/y_env.py 的 reset/step/camera/action/success API，逐步实现 `scripts/eval/eval_mujoco.py`，支持 --policy act|diffusion|smolvla、指定本地 checkpoint、seed、episode 数、ID/OOD split、相同成功判定与 JSONL 保存。必须显式完成每个策略的图像 key、状态向量、动作维度与归一化映射，不能用通用假输出伪装成功。
4. 重写 README 首页简介和目录，将单任务解释升级为三策略对比平台；保留旧章节内容，增加 ACT/DP/SmolVLA 训练、仿真评测、结果表（未测填待测）、脚本专家演示与真机部署规划。说明脚本专家视频不代表训练模型。
5. `docs/REAL_DEPLOYMENT.md` 仅建立 SO-100 部署计划；在真实相机/关节标定/夹爪映射/安全限速已验证之前，不写可以直接驱动真机的高风险命令。
6. 默认所有模型与数据不公开：`--policy.push_to_hub=false`、WANDB_MODE=disabled，补齐 .gitignore 的 private 数据、weights、outputs、rollouts 条目，且避免误忽略已有被追踪代码。不得自动 git push、force push、删除、重置或提交，先展示 `git diff --stat` 和检查结果供用户审核。
7. 最少测试：Python 语法检查、bash -n、评测汇总单测、无需模型权重的 mock rollout 接口测试；若接口缺失报告阻塞点，不要宣布完整训练与评测已通过。

交付：修改文件清单、尚未验证的接口清单、测试输出、Git diff 摘要，以及在 BitaHub 执行的精确命令（命令后立即说明需要反馈的输出）。
