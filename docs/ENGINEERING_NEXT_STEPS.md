# 工程修复与验证操作单（2026-11-08 历史记录）

> **状态说明（2026-11-09）：** 本文记录 2026-11-08 当轮代码修复、离线检查及当时的建议验证流程，并非整个项目当前进度。项目维护者确认仿真与真机实验均已完成；最新状态和已报告结果以 [项目状态](PROJECT_STATE.md) 与 [README](../README.md) 为准。实验原始产物未上传 GitHub。

以下保留当轮技术验收内容，命令不代表最新实际运行配置。

> **2026-11-09 后续代码更新：** 新增 `scripts/eval/eval_mujoco.py` 统一支持 ACT/DP/SmolVLA 评测，`scripts/eval/act_rollout.py` 增加 `policy_name` 参数，新增 `tests/test_unified_eval.py` 无模型测试。请以 README 10.5.1 节的现行命令为准。本节下面“多策略统一接口：具体后续实现方案”是 11 月 8 日的历史规划，而非当前功能缺失声明。

## 已处理

- 采集动作改为裁剪后的绝对目标关节角与期望夹爪比例；不再把物理未推进时的反馈当作目标。
- 动作shape/NaN/Inf提前检查；批量采集FPS与控制频率统一，seed按完成尝试递增，手动采集按配置FPS工作。
- ACT部署接入 `scripts/eval/act_rollout.py`，支持 `--rollout-jsonl`、`--eval-split` 和 `--timeout-clock wall|sim`。
- 每个已完成Episode自动记录seed、success/timeout、控制步数、墙钟/仿真时间、select_action耗时、checkpoint、代码SHA/工作区状态、关键源码哈希、包版本、布局范围与chunk配置。
- 新文件采用独占创建，拒绝覆盖旧评测；窗口关闭或执行错误记录completed=false，统计工具不把它计入成功率分母。
- 聚合器拒绝非法类型、负时间与NaN；缺失阶段检测时不填猜测值。

已运行：15项unittest通过，30个Python AST、28个XML解析和主场景7个XML本地引用检查通过；git diff --check通过。新评测调度、记录、汇总使用fake物理/策略验证真实函数链；尚未启动真实MuJoCo/模型。

## 1. 运行环境验收

在已有环境记录Python、mujoco、mink、torch、torchvision、lerobot的安装版本，以及Git SHA。先按仓库Docker/runtime已有配置检查，不同时升级依赖。以下命令都从仓库根目录执行；不要在现有base环境中临时混装。

先完成 `python scripts/verify_source.py` 和 `python -m unittest discover -s tests -v`。再在MuJoCo环境加载 `asset/scene_so101_y.xml`，只做reset和短step，检查：

1. 关节/actuator顺序为五臂关节+夹爪，限位和单位正确。
2. topview/sideview图像尺寸、颜色、方向与数据一致。
3. 默认位置、夹爪开合、接触数值无NaN；20Hz控制对应正确仿真时钟。
4. seed固定时初始布局重现；换seed时布局变化；可选其他场景的资产缺口单独处理。

若加载失败，先定位具体XML/资产或API报错，不更换整套依赖碰运气。

## 2. 新采集数据做小规模读回

先用现有batch入口采集1个Episode，独立输出目录，保持FPS20/control_hz20；若改频率，两项一起改。

验收时取同一控制周期的四项：obs_t、动作目标a_t、实际ctrl、obs_(t+1)。SO101 action前5维应与 `env.compute_q` 一致，末维为夹爪0开1闭比例；不要拿动作目标与当时反馈完全相等作为正确性条件。

检查state为末端xyz+RPY6维，action为五关节弧度+夹爪6维，时间戳差约1/FPS；图像与state同刻，成功轨迹最后确实放入盒子并松爪。旧数据先抽查，不自动改写；没有原控制命令时重新采集比猜测还原更可靠。

在meta中记录schema/采集代码版本。继续采集或合并旧数据前，先确认动作语义一致，避免同字段名下混入两种监督目标。

## 3. ACT最小训练与恢复

优先使用已有可配置shell入口，先打印：

```bash
DATASET_ROOT=/datasets/so101 DATASET_REPO_ID=local/so101 \
  bash scripts/train/train_act.sh --dry-run
```

已有依赖与训练授权后，先将TRAIN_STEPS设为少量步骤，batch按现有单卡容量选择。检查数据读取、loss有限、checkpoint完整、checkpoint与数据统计绑定。先不要扩GPU或跑多模型。

需特别核对当前LeRobot版本：ACT构造器是否接受dataset_stats、from_pretrained的参数、归一化是否由processor承担、select_action的输入键与输出是否反归一化。缺依赖时本轮无法确认这些API；不要同时叠加旧Normalize和新processor，造成二次归一化。

如果Python旧训练入口 `train_act.py` 与当前LeRobot版本不兼容，先把错误定位到上述具体接口，再统一训练/部署使用同一版本与processor方式，避免各自打兼容补丁。

## 4. 接通真实ACT评测与汇总

先在依赖环境跑1次功能检查。用实际checkpoint和dataset路径替换示例；JSONL必须是新路径：

```bash
python scripts/deploy/deploy_act.py --config configs/deploy_act.yaml \
  --checkpoint /checkpoints/act --dataset-root /datasets/so101 \
  --dataset-repo-id local/so101 --num-trials 1 --seed 7 \
  --timeout-clock sim --trial-timeout-s 60 \
  --eval-split smoke --rollout-jsonl outputs/eval/act_smoke.jsonl
```

```bash
python scripts/eval/summarize_rollouts.py outputs/eval/act_smoke.jsonl \
  --csv outputs/eval/act_smoke.csv
```

验收：每个结束的Episode一行；success+timeout=completed；窗口关闭的未完成Episode不计入分母。所有行checkpoint、频率、chunk、代码版本匹配；CSV成功数与逐行手算一致。`completion_time_s`依据所选timeout-clock，另保留wall_time_s与simulation_time_s。正式比较统一使用同一时钟。

`inference_ms`实际测量每次policy.select_action调用并同步输出；ACT缓存chunk时某些调用不做模型前向，因此还需同时报告chunk_size/n_action_steps。它不包含图像获取和预处理耗时，也不等于全闭环延迟。

自动记录的action_emitted只表示命令已发出，不能替代机械臂开始运动的检测。当前不自动填started/grasped/placed；对应阶段失败率在缺检测时保持空值。

## 5. 多策略统一接口：具体后续实现方案

先验证ACT完整链，再扩DP/SmolVLA：

1. 从实际checkpoint的config读取policy type、相机键、state/action形状、chunk与归一化配置；建立显式 `build_observation(raw_obs, policy_spec)`。
2. 每个策略提供同一接口：reset()、predict(observation)->绝对关节目标5维+夹爪比例。把processor/反归一化放在适配层，环境不重复处理。
3. 用各自本地checkpoint先单次预测；验证输出6维、有限、单位正确、夹爪映射一致。缺失或错误元数据直接报错，不尝试猜测。
4. 接入本轮的评测调度与JSONL记录，policy字段改成实际模型；保留seed、布局、控制频率、时钟与终止条件。
5. 使用完全相同的ID/OOD初态列表。每模型固定训练数据划分和预算；训练seed与测试布局seed分开。先少量Episode排接口，再确定正式数量。

本轮不能代替你完成各模型真实checkpoint/API验证。不要仅凭训练shell能打印命令就扩大正式训练。

## 6. 实验分析与面试证据

| 待补部分 | 具体方法 | 验收产物 |
|---|---|---|
| Expert稳定性 | 固定seed列表，统计成功/IK失败/超时；保存失败阶段 | 原始逐Episode表、关键视频 |
| ACT泛化 | 保存训练布局范围与测试初态，ID/OOD分开且均可达 | 按split汇总的成功数、Wilson区间 |
| 动作平滑性 | 计算相邻目标delta_q与实际关节速度分布 | 频率说明、P95/最大值、失败相关性 |
| 阶段失败检测 | 启动=关节位移阈值持续成立；抓取=物体抬起且接触；放置=盒内+释放 | 阈值/持续时间与手工视频标注交叉验证 |
| FK/IK一致性 | 固定多个可达关节配置；FK对比MuJoCo site位姿；IK回代FK | 位置/方向误差、不可达样本与限位行为 |
| 控制数值稳定性 | 固定场景逐步测试20Hz与其他批准频率，检查穿透/接触力/NaN | 运行配置与失败片段，不凭视频观感下结论 |

## 7. 真机相关后续顺序

需要实际硬件、标定与明确的实验资源：先记录关节名/顺序/单位/零位；验证相机和夹爪映射；比较低幅度目标命令与反馈；建立最大变化量、速度限幅和停止接口，再接策略。第一阶段先做单关节与空载检查，之后才做物体交互。每阶段保留命令与反馈日志。算法代码检查不能替代这些硬件验证。
