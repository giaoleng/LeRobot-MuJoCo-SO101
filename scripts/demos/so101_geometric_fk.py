import sys
from pathlib import Path
import time
import mujoco
import mujoco.viewer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from so101.mujoco_utils import set_initial_pose, send_position_command, move_to_pose, hold_position
from so101.forward_kinematics import get_forward_kinematics
import numpy as np

xml_path = PROJECT_ROOT / 'asset' / 'so_arm100' / 'SO101' / 'scene.xml'
m = mujoco.MjModel.from_xml_path(str(xml_path))
d = mujoco.MjData(m)

def show_cylinder(viewer, position, rotation, radius=0.0245, halfheight=0.05, rgba=[1, 0, 0, 1]):
    # 添加与 z 轴对齐的圆柱体
    mujoco.mjv_initGeom(
        viewer.user_scn.geoms[0],
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,   # 气缸类型
        size=[radius, halfheight, 0],                  # [半径，半高，忽略]
        pos=position,                         # 中心位置
        mat=rotation.flatten(),              # orientation matrix (identity = z-up)
        rgba=rgba                           # 颜色
    )
    viewer.user_scn.ngeom = 1
    viewer.sync()
    return

def show_tcp_frame(viewer, position, rotation, axis_length=0.05, radius=0.004):
    """Draw the TCP frame as three axis cylinders: R=x, G=y, B=z."""
    axes = [
        (np.array([1,0,0]), [1,0,0,1]),  # x = red
        (np.array([0,1,0]), [0,1,0,1]),  # y = green
        (np.array([0,0,1]), [0,0,1,1]),  # z = blue
    ]
    for i, (local_axis, rgba) in enumerate(axes):
        # 该轴在世界坐标系中的方向
        world_axis = rotation @ local_axis

        # Centre of the axis cylinder = TCP position + half-length along axis
        centre = position + world_axis * (axis_length / 2)

        # 构建一个旋转矩阵，其 z 轴指向 world_axis
        z = world_axis
        # Pick an arbitrary perpendicular for x
        x = np.array([1,0,0]) if abs(z[0]) < 0.9 else np.array([0,1,0])
        x = x - np.dot(x, z) * z
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        mat = np.column_stack([x, y, z])

        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[i],
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[radius, axis_length / 2, 0],
            pos=centre,
            mat=mat.flatten(),
            rgba=rgba
        )
    viewer.user_scn.ngeom = 3
    viewer.sync()


# 测试配置 - 用于根据图中的真实位置验证 FK 函数
test_configuration = {
    'shoulder_pan': -45.0,   # in radians for mujoco! 
    'shoulder_lift': 45.0,
    'elbow_flex': -45.00,
    'wrist_flex': 90.0,
    'wrist_roll': 0.0,
    'gripper': 10          
}

# for debug - shows the zero position of the robot and the TCP frame
test_configuration = {
    'shoulder_pan': 0.0,   # in radians for mujoco! 
    'shoulder_lift': 0.0,
    'elbow_flex': 0.0,
    'wrist_flex': 0.0,
    'wrist_roll': 0.0,
    'gripper': 0.0          
}

body_xml_path = PROJECT_ROOT / 'asset' / 'so_arm100' / 'SO101' / 'so101_body.xml'

set_initial_pose(d, test_configuration)
send_position_command(d, test_configuration)

object_position, object_orientation = get_forward_kinematics(test_configuration)

print("TCP position:", object_position)
print("TCP orientation (z-axis = cylinder axis):")
print(object_orientation)
print("z-column (cylinder axis in world):", object_orientation[:, 2])

for i in range(m.nbody):
    name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i)
    print(f"body {i}: {name}")

with mujoco.viewer.launch_passive(m, d) as viewer:
  # 30 秒后自动关闭查看器。
  start = time.time()
  
  # Add a cylinder as a site for visualization
  # show_cylindrical（查看器，对象位置，对象方向）
  # 然后调用它而不是 show_cylindrical：
  show_tcp_frame(viewer, object_position, object_orientation)

  # Hold Starting Position for 2 seconds
  hold_position(m, d, viewer, 120.0)


