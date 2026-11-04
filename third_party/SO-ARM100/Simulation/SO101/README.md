<a id="so101-robot---urdf-and-mujoco-description"></a>
# SO101 机器人 - URDF 和 MuJoCo 描述

该存储库包含 SO101 机器人的 URDF 和 MuJoCo (MJCF) 文件。

<a id="overview"></a>
## 概述

- 机器人模型文件是使用[onshape-to-robot](https://github.com/Rhoban/onshape-to-robot)插件从 Onshape 中设计的 CAD 模型生成的。
- 生成的 URDF 已修改为允许网格具有相对路径而不是`package://...`。
- 由于模拟和规划期间存在问题的碰撞行为，基础碰撞网格被删除。

<a id="calibration-methods"></a>
## 校准方法

MuJoCo 文件`scene.xml`支持两个不同校准的 SO101 机器人文件：

- **新校准（默认）**：每个关节的虚拟零位设置为其关节范围的**中间**。使用 ->`so101_new_calib.xml`。
- **旧校准**：每个关节的虚拟零设置为机器人**水平完全伸展的配置**。使用 ->`so101_old_calib.xml`。

要在校准方法之间切换，请修改`scene.xml`中包含的机器人文件。

<a id="motor-parameters"></a>
## 电机参数

机器人中使用的 STS3215 电机的电机属性改编自[Open Duck Mini project](https://github.com/apirrone/Open_Duck_Mini)。

<a id="gripper-note"></a>
## 抓手笔记

在LeRobot中，夹具表示为**线性关节**，其中：

* `0`= 全关闭
* `100`= 全开

此映射为**，尚未在当前 URDF 和 MuJoCo 文件中反映**。

---

请随意提出问题或贡献改进！
