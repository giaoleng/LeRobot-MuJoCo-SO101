import sys
from pathlib import Path
import time
import mujoco
import mujoco.viewer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from so101.mujoco_utils import set_initial_pose, send_position_command, move_to_pose, hold_position
from so101.forward_kinematics import get_forward_kinematics
#from so101_forward_kinematics import read_arm_translations
import numpy as np

xml_path = PROJECT_ROOT / 'asset' / 'so_arm100' / 'SO101' / 'scene.xml'
body_xml_path = PROJECT_ROOT / 'asset' / 'so_arm100' / 'SO101' / 'so101_body.xml'
# translation_dict = read_arm_translations(body_xml_path)

m = mujoco.MjModel.from_xml_path(str(xml_path))
d = mujoco.MjData(m)


def show_cubes(viewer, starting_config, final_config, halfwidth=0.013):
    starting_object_position, starting_object_orientation = get_forward_kinematics(starting_config)
    final_object_position, final_object_orientation = get_forward_kinematics(final_config)

    mujoco.mjv_initGeom(
        viewer.user_scn.geoms[0],
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[halfwidth, halfwidth, halfwidth],
        pos=starting_object_position,
        mat=starting_object_orientation.flatten(),
        rgba=[1, 0, 0, 0.2]
    )
    mujoco.mjv_initGeom(
        viewer.user_scn.geoms[1],
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[halfwidth, halfwidth, halfwidth],
        pos=final_object_position,
        mat=final_object_orientation.flatten(),
        rgba=[0, 1, 0, 0.2]
    )
    viewer.user_scn.ngeom = 2
    viewer.sync()


# ---联合配置---

# 1. 开始：拾取位置上方，夹具打开
starting_configuration = {
    'shoulder_pan': -45.0,
    'shoulder_lift': 45.0,
    'elbow_flex': -45.0,
    'wrist_flex': 90.0,
    'wrist_roll': 0.0,
    'gripper': 50
}

# 2. 在拾取位置关闭夹具
starting_configuration_closed = {
    'shoulder_pan': -45.0,
    'shoulder_lift': 45.0,
    'elbow_flex': -45.0,
    'wrist_flex': 90.0,
    'wrist_roll': 0.0,
    'gripper': 5
}

# 3. 在摆动之前抬起桌子以清理桌子
lift_configuration = {
    'shoulder_pan': -45.0,
    'shoulder_lift': 10.0,
    'elbow_flex': -10.0,
    'wrist_flex': 90.0,
    'wrist_roll': 0.0,
    'gripper': 5
}

# 4. 摆动至放置位置（仍举起）
swing_configuration = {
    'shoulder_pan': 45.0,
    'shoulder_lift': 10.0,
    'elbow_flex': -10.0,
    'wrist_flex': 90.0,
    'wrist_roll': 0.0,
    'gripper': 5
}

# 5. 降低至放置位置，夹具仍关闭
final_configuration_closed = {
    'shoulder_pan': 45.0,
    'shoulder_lift': 45.0,
    'elbow_flex': -45.0,
    'wrist_flex': 90.0,
    'wrist_roll': 0.0,
    'gripper': 5
}

# 6. 打开夹具以释放
final_configuration = {
    'shoulder_pan': 45.0,
    'shoulder_lift': 45.0,
    'elbow_flex': -45.0,
    'wrist_flex': 90.0,
    'wrist_roll': 0.0,
    'gripper': 50
}


# ---运行模拟---

set_initial_pose(d, starting_configuration)
send_position_command(d, starting_configuration)

with mujoco.viewer.launch_passive(m, d) as viewer:

    # 显示拾取和放置目标立方体
    show_cubes(viewer, starting_configuration, final_configuration)

    # 1. 保持在拾取位置，夹具打开
    print("Step 1: At pick location, gripper open")
    hold_position(m, d, viewer, 2.0)

    # 2. 关闭夹具以抓取物体
    print("Step 2: Closing gripper")
    move_to_pose(m, d, viewer, starting_configuration_closed, 1.0)
    hold_position(m, d, viewer, 1.0)

    # 3. 抬起以清理桌子
    print("Step 3: Lifting")
    move_to_pose(m, d, viewer, lift_configuration, 1.0)
    hold_position(m, d, viewer, 0.5)

    # 4. 摆动至放置位置
    print("Step 4: Swinging across")
    move_to_pose(m, d, viewer, swing_configuration, 1.5)
    hold_position(m, d, viewer, 0.5)

    # 5. 降低放置位置
    print("Step 5: Lowering to place location")
    move_to_pose(m, d, viewer, final_configuration_closed, 1.0)
    hold_position(m, d, viewer, 1.0)

    # 6. 打开夹具以释放物体
    print("Step 6: Opening gripper")
    move_to_pose(m, d, viewer, final_configuration, 1.0)
    hold_position(m, d, viewer, 2.0)

    print("Pick and place complete.")