# SO-101 真机部署与评测

> **状态说明（2026-11-09）：** 项目维护者确认 SO-101 真机操作实验已经完成，ACT 与 SmolVLA 的汇总结果见 [README 第 13.8 节](../README.md#138-实验结果)。原始日志、权重、标定与测试视频未上传 GitHub。下文为 **2026-11-08 代码接口补全阶段** 的技术说明和参考配置，记录的是当轮开发与静态检查范围，不能据此判断整个项目目前尚未开展真机实验。

## 2026-11-08 工程交付记录

新增入口：

| 文件 | 职责 |
|---|---|
| `scripts/real/deploy_policy.py` | 加载本地ACT/SmolVLA/Diffusion策略，预处理、推理、动作限制、真机执行 |
| `scripts/real/evaluate_policy.py` | 人工复位与判定、多目标切换、逐Episode JSONL |
| `scripts/real/task_conditioning.py` | 共享目标词表、ACT one-hot条件、训练数据包装与manifest核对 |
| `scripts/real/runtime.py` | 配置/资源检查、LeRobot适配、限位/速率保护与资源释放 |
| `scripts/real/train_target_act.py` | 训练带类别条件的ACT，保存策略、processor、词表和数据feature快照 |
| `configs/real/so101.yaml` | 串口、校准、相机、checkpoint、类别词表、控制参数 |
| `configs/real/requirements.txt` | 真机独立环境的依赖入口 |

本轮只做Python AST、YAML和文件/接口静态核对；未连接硬件、加载权重或运行训练。新增离线回归用例在 `tests/test_real_interfaces.py`，留待后续运行。

## 配置与依赖

接口版本为LeRobot 0.5.1。目标训练/部署环境使用 `configs/real/requirements.txt`；本轮不安装。包中包含Feetech和SmolVLA可选依赖，Torch等依赖由LeRobot声明。模型、VLM配置和tokenizer资源需已在本地；部署显式启用离线加载，不在执行中下载。

`configs/real/so101.yaml` 已填好一套常见双USB相机参考配置，当前适用于没有设备时的离线代码检查：

| 参数 | 默认值 |
|---|---|
| 串口 / 机器人ID | `/dev/ttyACM0` / `so101_follower` |
| front / wrist相机 | 索引0 / 1，均640×480、30FPS |
| 标定目录 | `./calibration/so101`，预期文件`so101_follower.json` |
| ACT checkpoint | `./checkpoints/real_act/pretrained_model` |
| 真机数据目录 | `./data/real_so101` |
| 策略控制频率 | 20Hz |

五个臂关节的参考范围取自仓库 `asset/so_arm100/SO101/so101_body.xml`，由弧度转换为角度；夹爪仍单独使用真机驱动0..100百分比。它们不能还原租用设备的零位偏移或工作空间限制。`robot.hardware_confirmed=false` 让参考配置保持可读、可静态检查，同时禁止直接启动硬件。

没有设备时无需补造标定JSON、权重文件或相机序列号。以后若拿到实际设备及已有训练产物，再核对以下配置，并将hardware_confirmed设为true：

1. robot.port、id、calibration_dir。目录必须已有 `<id>.json`，入口不自动标定。标定来自同一台机械臂，连接后还核对电机标定状态。
2. cameras中的硬件名称、设备索引、尺寸与FPS。相机名应对应 `observation.images.<name>`，与训练元数据一致。
3. policy.checkpoint和dataset_root。checkpoint目录应为LeRobot的pretrained_model，含config.json、model.safetensors，以及saved模式下两个processor JSON及其资产。
4. 按本机标定/工作空间核对五臂关节joint_limits；当前已填模型参考范围。gripper限位是标定后的0..100百分比。
5. 选择policy.type和conditioning.mode；ACT多目标使用category，SmolVLA使用language，单任务ACT可使用none。

真机state和action均使用六个命名的 `.pos` 关节。五臂关节是degrees，夹爪是标定百分比，不能将MuJoCo的弧度/夹爪闭合比例或末端xyz+RPY直接作为此入口的真机输入。该差异通过feature名字和动作语义显式检查，不进行猜测转换。

默认计划检查，不导入Torch/LeRobot，也不连接硬件：

```bash
python scripts/real/deploy_policy.py --config configs/real/so101.yaml --dry-run
python scripts/real/evaluate_policy.py --config configs/real/so101.yaml --episodes 2 --dry-run
```

只检查已有本地文件，再加 `--check-files`。它核对checkpoint类型、六关节feature、相机映射、类别manifest和processor文件；不加载模型。SmolVLA的processor还可能依赖本地tokenizer/VLM缓存，执行环境需提前备齐。

## ACT类别条件

采用ACT已有的 `observation.environment_state` 分支，输入为按照target_order排列的one-hot；训练和部署使用同一TaskConditioner，类别变换会改变ENV输入token。

训练数据应是已标定的真机关节数据，帧的task文本必须能与配置中的目标指令一一匹配。训练包装器根据task生成one-hot，ENV采用IDENTITY归一化，其他观测/动作由LeRobot processor处理。

仅打印训练配置：

```bash
python scripts/real/train_target_act.py --config configs/real/so101.yaml \
  --output checkpoints/real_act/pretrained_model --steps 5
```

后续你准备好环境、数据和计算预算后，显式加 `--execute` 才训练。它不初始化机械臂，不推送模型；初始ResNet不额外下载权重。输出保存策略、前后processor、task_conditioning.json、dataset_info.json、training_recipe.json。部署优先读取checkpoint里的feature快照，避免训练时新增ENV特征未写回原数据目录的问题。

已有类别条件checkpoint必须提供一致的task_conditioning.json和ENV输入维数。普通ACT checkpoint不能仅靠传target_id就变成类别条件策略。使用 `--export-conditioning /path/task_conditioning.json` 可以写出当前映射供训练准备，但不会修改模型结构或权重。

SmolVLA通过task传入语言条件，例如：

```bash
python scripts/real/deploy_policy.py --config configs/real/so101.yaml \
  --policy-type smolvla --conditioning-mode language \
  --checkpoint checkpoints/real_smolvla/pretrained_model --target-id red_car --dry-run
```

## 后续部署

下面命令才会连接机械臂，仅在完成本机配置后由你执行：

```bash
python scripts/real/deploy_policy.py --config configs/real/so101.yaml \
  --target-id yellow_duck --duration-s 10 \
  --output outputs/real/deploy_round1.jsonl --execute
```

对每次动作检查有限值、完整关节键、标定限位、相对步长与最大速度。推理耗时或观测年龄超过配置时，本步不发送动作。控制循环中发现stop_file时退出；该文件由操作者创建且不自动删除。它是软件停止入口，不能替代实体急停；阻塞的驱动/推理调用期间无法立即处理文件变化。

每轮退出或异常均尝试释放串口和相机。disable_torque_on_disconnect默认true，机械臂退出后可能失去保持力，操作者应先准备支撑或安全停放位置；可根据硬件操作规程配置。

## 后续评测与汇总

```bash
python scripts/real/evaluate_policy.py --config configs/real/so101.yaml \
  --targets yellow_duck red_car --episodes 4 --split id \
  --output outputs/real/eval_round1.jsonl --execute
```

每集开始前由操作者复位并确认，结束后输入结果JSON；脚本不通过“动作已发出”或“时间已到”推断成功。运行结束后已断开机器人，再等待判定。目标按targets顺序循环；每集重置policy/processor状态，避免上一任务的动作chunk污染下一任务。

结果例：`{"success":true,"correct_target":true,"grasped":true,"placed":true}`。失败可以只提供success=false，已知阶段再补充；脚本检查布尔类型和完整阶段信息的逻辑一致性。中断记录completed=false，现有聚合器不将其计入成功率分母。

```bash
python scripts/eval/summarize_rollouts.py outputs/real/eval_round1.jsonl \
  --csv outputs/real/eval_round1.csv
```

部署日志kind=deployment不含人工成绩，用于排查执行过程；请只将evaluate_policy产生的评测日志传给成功率汇总器。日志文件独占创建，拒绝覆盖已有记录。

## 静态接口依据

此次查阅固定版本 `huggingface/lerobot` v0.5.1，commit `1396b9fab7aecddd10006c33c47a487ffdcb54b4`，沿用其公开接口而未复制驱动实现：

- [SO follower配置与驱动](https://github.com/huggingface/lerobot/tree/1396b9fab7aecddd10006c33c47a487ffdcb54b4/src/lerobot/robots/so_follower)
- [ACT训练与推理接口示例](https://github.com/huggingface/lerobot/tree/1396b9fab7aecddd10006c33c47a487ffdcb54b4/examples/tutorial/act)
- [SmolVLA推理示例](https://github.com/huggingface/lerobot/blob/1396b9fab7aecddd10006c33c47a487ffdcb54b4/examples/tutorial/smolvla/using_smolvla_example.py)
- [processor工厂](https://github.com/huggingface/lerobot/blob/1396b9fab7aecddd10006c33c47a487ffdcb54b4/src/lerobot/policies/factory.py)

以上是本轮接口基线；未来版本变化应先核对API再迁移。

---

## 早期设计记录

以下保留原设计约束，当前文件与命令以本文前半部分为准。

# 早期 SO-100 方案草稿（历史记录，非当前 SO-101 项目状态）

目标：使用仿真训练/微调的 SmolVLA 进行 SO-100 真机验证。

## 部署前必须验证

1. SO-101 MuJoCo 与 SO-100 真实机器人的关节顺序、限位、角度单位和动作范围映射。
2. 夹爪开/合方向、位置标定及动作裁剪。
3. 图像分辨率、相机数量与 camera key 与训练一致；检查 SmolVLA 的 rename map。
4. 模型输出动作与 LeRobot SO-100 驱动器目标控制量是否完全一致。
5. 有物理急停、最大关节变化量、速度限制；先无物体低速测试，再逐步接触。
6. 避免将仿真中允许的穿透、过大接触力或危险指令发送到真实硬件。

在验证以上接口之前不提供可直接驱动机械臂的脚本，以免误将错误关节单位或夹爪映射用于真机。

## 实验结论范围

若只部署 SmolVLA 到真机，能够报告其真机成功率，但不能声称它在真机上优于 ACT 和 Diffusion Policy。
