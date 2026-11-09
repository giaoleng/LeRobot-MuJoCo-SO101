# README 首页替换稿（人工合并，不覆盖现有 README）

> 此文件提供可以复制到现有 README.md 的简介与新增章节。保留原有安装、EE5108、数据采集、转换、回放、ACT 操作、致谢和许可证说明。

# LeRobot + MuJoCo + SO-101

**ACT · Diffusion Policy · SmolVLA | SO-101 仿真训练与闭环评测 | SO-100 真机部署规划**

基于 LeRobot、MuJoCo 和 SO-101 机械臂构建机器人模仿学习实验平台。任务为控制机械臂抓取蓝色方块并放入收纳盒，覆盖专家示范采集、LeRobotDataset 转换、策略训练和闭环评测。计划比较 ACT、Diffusion Policy（DP）及 SmolVLA，并探索将 SmolVLA 迁移到 SO-100 真机。

**开发状态：** 当前已完成 MuJoCo 场景和脚本专家演示。ACT / DP / SmolVLA 的训练、闭环评测及 SmolVLA 真机部署应按实际验收进度逐项标记，不预报成功率。

## 策略训练与评测

| 策略 | 实现 | 训练方式 | 入口 | 状态 |
| --- | --- | --- | --- | --- |
| ACT | LeRobot | 从头训练 | 现有 ACT 训练流程 | 待验证/按实际更新 |
| Diffusion Policy | LeRobot `diffusion` | 从头训练 | `scripts/train/train_diffusion.sh` | 入口已编写，待远程训练 |
| SmolVLA | LeRobot `smolvla_base` | 预训练权重微调 | `scripts/train/finetune_smolvla.sh` | 入口已编写，待远程训练 |

三种策略使用同一份 LeRobotDataset、同一组测试 Seeds 和同一套成功判定规则。训练和模型权重保留在私有远程存储，不默认上传 Hub；详情见 [三策略实验方案](POLICY_COMPARISON.md)。

## 仿真演示

[查看已录制的脚本专家 MuJoCo 演示](../media/simulation_demo.mp4)（**不是 ACT/DP/SmolVLA 策略演示**）。

## 真机部署规划

仅计划将 SmolVLA 部署至 SO-100 真机；需要先完成关节校准、动作范围、夹爪、摄像头输入、急停和限速验证，详情见 [真机部署注意事项](REAL_DEPLOYMENT.md)。

## 目录补充

保留原目录表，并新增：

- [策略训练与评测](#策略训练与评测)
- [仿真演示](#仿真演示)
- [真机部署规划](#真机部署规划)
