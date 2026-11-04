import numpy as np
from .forward_kinematics import get_forward_kinematics, get_gw1, get_g12, get_g23, get_g34

# ---------------------------------------------------------------------------
# 机器人常数
# ---------------------------------------------------------------------------
BASE_X = 0.0388353   # Shoulder_pan 关节的世界 x 偏移量
BASE_Y = 0.0

L1 = np.sqrt(0.11257**2 + 0.028**2)   # 上臂：~0.1160 m
L2 = np.sqrt(0.1349**2  + 0.0052**2)  # 前臂：~0.1350 m
BETA1 = np.arctan2(0.028,  0.11257)   # 固定连杆 1 偏移角度：~13.97 度
BETA2 = np.arctan2(0.0052, 0.1349)    # 固定连杆 2 偏移角度：~2.21 度
ZERO_INTERIOR = np.pi/2 - BETA1 - BETA2  # interior elbow angle at theta3=0: ~73.82 deg

TCP_Z_OFFSET = 0.100  # g5t：在 z 世界中，TCP 远低于手腕滚动
G45_LOCAL    = np.array([0.0, -0.0611, 0.0181])  # 手腕弯曲框架中的 g45 位移
DEFAULT_WRIST_ROLL_DEG = 0.0

# 解析FK/IK模型在机器人的局部场景框架中表达，而
# the tabletop MuJoCo scene places the robot on a table at z=0.805 m.
# 公共 IK 输入是绝对桌面世界坐标。
TABLETOP_WORLD_OFFSET = np.array([0.0, 0.0, 0.805], dtype=np.float64)

SHOULDER_LIFT_MIN_DEG = -100.0
SHOULDER_LIFT_MAX_DEG = 100.0
ELBOW_FLEX_MIN_DEG = -95.0
ELBOW_FLEX_MAX_DEG = 95.0


# ---------------------------------------------------------------------------
# 1b：theta1（肩盘）
# ---------------------------------------------------------------------------

def solve_theta1(target_position):
    """Solve shoulder_pan by projecting target onto the xy-plane.

    theta1 = -atan2(y_des, x_des) from base offset.
    Negated for the Rz(180)@Rx(180) convention in get_gw1.
    """
    x = target_position[0] - BASE_X
    y = target_position[1] - BASE_Y
    return np.rad2deg(-np.arctan2(y, x))


def world_to_analytic_frame(target_position):
    """Convert absolute tabletop world coordinates to the analytic IK frame."""
    return np.asarray(target_position, dtype=np.float64) - TABLETOP_WORLD_OFFSET


def analytic_to_world_frame(position):
    """Convert analytic IK/FK coordinates back to tabletop world coordinates."""
    return np.asarray(position, dtype=np.float64) + TABLETOP_WORLD_OFFSET


# ---------------------------------------------------------------------------
# 1c：所需的手腕弯曲位置
# ---------------------------------------------------------------------------

def get_wrist_flex_position(tcp_pos, theta1_deg):
    """Back-compute desired wrist_flex (joint 4) position from TCP target.

    For a vertical grasp with simplified g5t (eye, [0,0,-0.100]):
      wrist_roll = tcp_pos + [0, 0, 0.100]   (TCP is 0.1m below wrist_roll)
      wrist_flex = wrist_roll - R_wf @ g45_local

    R_wf is the wrist_flex frame rotation, which for a vertical grasp
    (theta4=90) depends only on theta1 — independent of theta2 and theta3.
    """
    tcp_pos = np.array(tcp_pos)

    # warn_roll 位于 TCP 正上方 TCP_Z_OFFSET
    wrist_roll = tcp_pos + np.array([0., 0., TCP_Z_OFFSET])

    # wrist_flex rotation (theta2=theta3=0 reference, theta4=90 for vertical grasp)
    R_wf = (get_gw1(theta1_deg) @ get_g12(0) @ get_g23(0) @ get_g34(90))[0:3, 0:3]

    # wrist_flex position = wrist_roll minus g45 offset expressed in world frame
    wrist_flex = wrist_roll - R_wf @ G45_LOCAL
    return wrist_flex


# ---------------------------------------------------------------------------
# 1d：theta2 和 theta3（肩抬和肘屈）
# ---------------------------------------------------------------------------

def solve_theta2_theta3(theta1_deg, wrist_target):
    """2-link planar IK for shoulder_lift and elbow_flex.

    Projects shoulder->wrist onto the arm plane (reach_dir, world_z)
    and applies the law of cosines.

    Returns NaN for both angles if target is unreachable.
    """
    theta1_rad = np.deg2rad(theta1_deg)

    # 肩部位置（无论theta2、theta3都是固定的）
    shoulder = (get_gw1(theta1_deg) @ get_g12(0.0))[0:3, 3]

    # 手臂平面内的伸展方向 — 仅源自 theta1
    reach_3d = np.array([np.cos(-theta1_rad), np.sin(-theta1_rad), 0.0])

    # 将肩膀->手腕投影到 (reach_dir, world_z)
    delta   = np.array(wrist_target) - shoulder
    r_reach = np.dot(delta, reach_3d)
    r_up    = delta[2]
    r       = np.sqrt(r_reach**2 + r_up**2)

    if r > L1 + L2 or r < abs(L1 - L2):
        return float('nan'), float('nan')

    # 余弦定律——肘部内角
    cos_elbow = np.clip((r**2 - L1**2 - L2**2) / (2*L1*L2), -1, 1)
    elbow_angle = np.arccos(cos_elbow)

    # 余弦定律 — 东西向三角形的肩角
    cos_alpha = np.clip((L1**2 + r**2 - L2**2) / (2*L1*r), -1, 1)
    alpha = np.arccos(cos_alpha)

    # 肩部->手腕与垂直方向的仰角
    psi = np.arctan2(r_reach, r_up)

    theta2 = psi - alpha - BETA1
    theta3 = elbow_angle - ZERO_INTERIOR

    return np.rad2deg(theta2), np.rad2deg(theta3)


# ---------------------------------------------------------------------------
# 1e：theta4（手腕弯曲）
# ---------------------------------------------------------------------------

def solve_theta4(theta1_deg, theta2_deg, theta3_deg):
    """Solve wrist_flex to align the gripper z-axis with world z (vertical grasp).

    After theta1-3 are known, the wrist_flex frame orientation is determined
    by the accumulated rotation R_before = gw1 @ g12 @ g23.
    theta4 rotates about the wrist_flex local z-axis via g34 (R_fixed=Rz(-90)).

    For wrist_roll z = world z we need:
        R_before @ Rz(-90 + theta4) @ [0,1,0] = R_before.T @ [0,0,1]

    Solving:   theta4 = 90 + atan2(-target[0], target[1])
    where target = R_before.T @ [0,0,1]
    """
    R_before = (get_gw1(theta1_deg) @ get_g12(theta2_deg) @ get_g23(theta3_deg))[0:3, 0:3]
    target = R_before.T @ np.array([0., 0., 1.])
    theta4 = np.pi/2 + np.arctan2(-target[0], target[1])
    return np.rad2deg(theta4)


def _tcp_error(theta1_deg, theta2_deg, theta3_deg, target_position):
    theta4_deg = solve_theta4(theta1_deg, theta2_deg, theta3_deg)
    fk_pos, _ = get_forward_kinematics(
        {
            'shoulder_pan': theta1_deg,
            'shoulder_lift': theta2_deg,
            'elbow_flex': theta3_deg,
            'wrist_flex': theta4_deg,
            'wrist_roll': DEFAULT_WRIST_ROLL_DEG,
        }
    )
    err = np.asarray(fk_pos, dtype=np.float64) - np.asarray(target_position, dtype=np.float64)
    return float(np.linalg.norm(err)), float(theta4_deg)


def solve_theta2_theta3_from_tcp(theta1_deg, target_position):
    """Solve shoulder/elbow by minimizing TCP position error directly.

    This uses absolute world-frame TCP coordinates and assumes a top-down grasp
    with fixed `wrist_roll=0`.
    """
    target_position = np.asarray(target_position, dtype=np.float64)

    best_err = float('inf')
    best_pair = (float('nan'), float('nan'))

    coarse_t2 = np.linspace(-20.0, 80.0, 21)
    coarse_t3 = np.linspace(-90.0, 20.0, 23)
    for theta2_deg in coarse_t2:
        for theta3_deg in coarse_t3:
            err, _ = _tcp_error(theta1_deg, theta2_deg, theta3_deg, target_position)
            if err < best_err:
                best_err = err
                best_pair = (theta2_deg, theta3_deg)

    theta2_deg, theta3_deg = best_pair
    for step in (10.0, 5.0, 2.0, 1.0, 0.5):
        improved = True
        while improved:
            improved = False
            candidates = [
                (theta2_deg, theta3_deg),
                (theta2_deg + step, theta3_deg),
                (theta2_deg - step, theta3_deg),
                (theta2_deg, theta3_deg + step),
                (theta2_deg, theta3_deg - step),
                (theta2_deg + step, theta3_deg + step),
                (theta2_deg + step, theta3_deg - step),
                (theta2_deg - step, theta3_deg + step),
                (theta2_deg - step, theta3_deg - step),
            ]
            for cand_t2, cand_t3 in candidates:
                cand_t2 = float(np.clip(cand_t2, SHOULDER_LIFT_MIN_DEG, SHOULDER_LIFT_MAX_DEG))
                cand_t3 = float(np.clip(cand_t3, ELBOW_FLEX_MIN_DEG, ELBOW_FLEX_MAX_DEG))
                err, _ = _tcp_error(theta1_deg, cand_t2, cand_t3, target_position)
                if err + 1e-9 < best_err:
                    best_err = err
                    theta2_deg = cand_t2
                    theta3_deg = cand_t3
                    improved = True

    if best_err > 0.02:
        return float('nan'), float('nan')

    return float(theta2_deg), float(theta3_deg)


# ---------------------------------------------------------------------------
# 全逆运动学
# ---------------------------------------------------------------------------

def get_inverse_kinematics(target_position, target_orientation=None):
    """Geometric IK for the SO-101 arm assuming a vertical grasp.

    Parameters
    ----------
    target_position    : array-like (3,) — desired TCP [x, y, z]
    target_orientation : np.ndarray (3,3), optional

    Returns
    -------
    joint_config : dict (angles in degrees)
    """
    target_position = world_to_analytic_frame(target_position)

    joint_config = {
        'shoulder_pan':  0.0,
        'shoulder_lift': 0.0,
        'elbow_flex':    0.0,
        'wrist_flex':    0.0,
        'wrist_roll':    0.0,
        'gripper':       0.0,
    }

    # 1b: θ1
    theta1 = solve_theta1(target_position)
    joint_config['shoulder_pan'] = theta1

    # 1d：theta2、theta3 直接在绝对世界坐标系中通过 TCP 求解
    theta2, theta3 = solve_theta2_theta3_from_tcp(theta1, target_position)
    joint_config['shoulder_lift'] = theta2
    joint_config['elbow_flex']    = theta3

    # 1e: θ4
    theta4 = solve_theta4(theta1, theta2, theta3)
    joint_config['wrist_flex'] = theta4

    # 1f: θ5
    joint_config['wrist_roll'] = DEFAULT_WRIST_ROLL_DEG

    return joint_config


# ---------------------------------------------------------------------------
# 确认
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    from .forward_kinematics import get_forward_kinematics

    known_config = {
        'shoulder_pan':  -45.0,
        'shoulder_lift':  45.0,
        'elbow_flex':    -45.0,
        'wrist_flex':     90.0,
        'wrist_roll':      0.0,
    }
    tcp_pos, _ = get_forward_kinematics(known_config)
    print(f"FK TCP position: {np.round(tcp_pos, 4)}")

    wrist = get_wrist_flex_position(tcp_pos, -45.0)
    print(f"Wrist flex pos:  {np.round(wrist, 4)}  (expect [0.2389, 0.1742, 0.1816])")

    solved = get_inverse_kinematics(tcp_pos)
    print(f"\nSolved:")
    print(f"  theta1: {solved['shoulder_pan']:.4f}   (expect -45.0)")
    print(f"  theta2: {solved['shoulder_lift']:.4f}   (expect  45.0)")
    print(f"  theta3: {solved['elbow_flex']:.4f}   (expect -45.0)")
    print(f"  theta4: {solved['wrist_flex']:.4f}   (expect  90.0)")
    print(f"  theta5: {solved['wrist_roll']:.4f}   (expect -45.0)")

    # 附加配置
    print("\nAdditional verification:")
    for t2, t3 in [(30,-30),(60,-20),(50,-60)]:
        cfg = {'shoulder_pan':-45,'shoulder_lift':t2,'elbow_flex':t3,
               'wrist_flex':90,'wrist_roll':0}
        tcp,_ = get_forward_kinematics(cfg)
        sol = get_inverse_kinematics(tcp)
        print(f"  true t2={t2:3d}, t3={t3:3d} -> solved t2={sol['shoulder_lift']:.1f}, t3={sol['elbow_flex']:.1f}")