# scripted_fsm_controller_with_mink.py
#
# Robust scripted controller for MuJoCo dataset generation using mink IK.
# Designed for long unattended demo collection (10k+ episodes).
#
# 特征
# - 基于 mink 的 IK 解算器，具有 QP 优化
# - 声明性阶段表
# - IK重试逻辑
# - 超时保护
# - 掌握验证
# - 干净的状态处理

import numpy as np
import mujoco
import mink
from typing import Optional, Tuple, Callable

# -------------------------------------------------------------
# 抓取校准（与之前相同）
# -------------------------------------------------------------

GRASP_Z_OFFSET = -0.012       # final Z offset for grasp
GRASP_X_OFFSET = -0.005       # 小 XY 调整至块中心
GRASP_Y_OFFSET = 0.0
APPROACH_Z_OFFSET = 0.075     # 块/框上方的悬停高度
PLACE_Z_OFFSET = 0.040        # 放入盒子时的深度
# 下降到块上时每刻的最大 Z 步数（米/刻）。
DESCEND_Z_STEP = 0.0035
# 下降到箱中时每个刻度的最大 Z 步（避免过冲/振荡）
PLACE_DESCEND_Z_STEP = 0.0028
# 当 place_in_box 在此距离内时，缩小 dq 以避免过冲/振荡
PLACE_DAMP_DIST = 0.04
FSM_DEBUG = True
FSM_DEBUG_EVERY = 5
# Max joint delta per control step (rad/tick); larger = faster motion
MAX_DQ_STEP = 1.0

# 方向策略：仅允许改变偏航（Z 旋转）。
LOCK_XY_ROTATION = True       # 保持横滚/俯仰固定，自由偏航

# -------------------------------------------------------------
# 相表 - 添加新的旋转相
# -------------------------------------------------------------

def quat_normalize_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / n


def quat_wxyz_to_rotmat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = quat_normalize_wxyz(q)
    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ], dtype=np.float64)


def rpy_to_quat_wxyz(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = np.cos(roll * 0.5), np.sin(roll * 0.5)
    cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
    cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)

    q = np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ], dtype=np.float64)
    return quat_normalize_wxyz(q)


def quat_wxyz_to_rpy(q: np.ndarray) -> Tuple[float, float, float]:
    r = quat_wxyz_to_rotmat(q)
    roll = np.arctan2(r[2, 1], r[2, 2])
    pitch = np.arcsin(np.clip(-r[2, 0], -1.0, 1.0))
    yaw = np.arctan2(r[1, 0], r[0, 0])
    return float(roll), float(pitch), float(yaw)

# Top-down quaternion for gripperframe site (wxyz).
# The gripperframe site has a 90-deg Y rotation baked in (quat="0.707107 0 0.707107 0")
# so the site's X-axis = gripper body -Z (approach direction).
# 对于自上而下的方法，我们需要站点 X 轴 -> 世界 -Z，
# i.e. R(q)*[1,0,0] = [0,0,-1], solved by q = [0.707107, 0, 0.707107, 0].
TOP_DOWN_QUAT = np.array([0.707107, 0.0, 0.707107, 0.0], dtype=np.float64)
DEFAULT_EE_SITE_NAME = "gripperframe"


def resolve_ee_site_name(env, preferred: Optional[str] = None) -> str:
    """Resolve a valid end-effector site name for the current model."""
    candidates = []
    if preferred:
        candidates.append(preferred)
    candidates.extend([DEFAULT_EE_SITE_NAME, "tip"])

    seen = set()
    for name in candidates:
        if name in seen or name is None:
            continue
        seen.add(name)
        try:
            env.model.site(name).id
            return name
        except KeyError:
            continue

    valid_names = [env.model.site(i).name for i in range(env.model.nsite)]
    raise KeyError(
        f"No valid end-effector site found. Tried {candidates}. "
        f"Available sites: {valid_names}"
    )

def quat_angle_between(q1: np.ndarray, q2: np.ndarray) -> float:
    """Compute angle between two quaternions (wxyz format)."""
    q1 = quat_normalize_wxyz(q1)
    q2 = quat_normalize_wxyz(q2)
    dot = np.abs(np.clip(np.dot(q1, q2), -1.0, 1.0))
    return 2.0 * np.arccos(dot)


def topdown_axis_error(q_wxyz: np.ndarray) -> float:
    """Yaw-invariant top-down error for ee site X-axis vs world -Z (radians)."""
    rot = quat_wxyz_to_rotmat(q_wxyz)
    ee_x_axis = rot[:, 0]
    desired_down = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    cosang = np.clip(np.dot(ee_x_axis, desired_down), -1.0, 1.0)
    return float(np.arccos(cosang))

PHASES = [
    # 0) 在剧集开始时，从存放位置移动到安全的“回家”姿势
    dict(
        name="stow_to_home",
        target=None,          # 由 fsm["home_xyz"] 填充
        gripper=1.0,
        tol=0.04,
        timeout=250,
    ),
    # 1) 在夹具朝下的情况下移动到块上方，仅偏航旋转
    dict(
        name="above_block",
        target=lambda cube, box: [
            cube[0] + GRASP_X_OFFSET,
            cube[1] + GRASP_Y_OFFSET,
            cube[2] + APPROACH_Z_OFFSET,
        ],
        # 在夹具关闭的情况下开始剧集，这样我们就清楚
        # 请参阅下降阶段开始时的打开手势。
        gripper=1.0,
        tol=0.02,
        timeout=150,
    ),
    # 2) 笔直下降到方块上
    dict(
        name="descend",
        target=lambda cube, box: [
            cube[0] + GRASP_X_OFFSET,
            cube[1] + GRASP_Y_OFFSET,
            cube[2] + GRASP_Z_OFFSET,
        ],
        # 当我们下降时，钳口会张开，以便在关闭之前清除所有障碍物。
        gripper=0.0,
        tol=0.015,
        timeout=150,
    ),
    # 3) 将夹具闭合到位
    dict(
        name="grip",
        target=None,          # 保持当前姿势
        gripper=1.0,
        tol=None,             # 纯粹基于时间
        timeout=15,
    ),
    # 4) 从桌子上垂直提起
    dict(
        name="lift",
        target=lambda cube, box: [
            cube[0] + GRASP_X_OFFSET,
            cube[1] + GRASP_Y_OFFSET,
            cube[2] + APPROACH_Z_OFFSET,
        ],
        gripper=1.0,
        tol=0.02,
        timeout=150,
    ),
    # 5）移动到盒子上方的位置
    dict(
        name="move_to_box",
        target=lambda cube, box: [
            box[0],
            box[1],
            box[2] + APPROACH_Z_OFFSET,
        ],
        gripper=1.0,
        tol=0.02,
        timeout=200,
    ),
    # 6) 放入盒子中并释放
    dict(
        name="place_in_box",
        target=lambda cube, box: [
            box[0],
            box[1],
            box[2] + PLACE_Z_OFFSET,
        ],
        gripper=0.0,          # 在底部时打开
        tol=0.02,
        timeout=150,
    ),
    # 7）返回原点（初始）姿势
    dict(
        name="return_home",
        target=None,          # 由 fsm["home_xyz"] 填充
        gripper=0.0,
        tol=0.03,
        timeout=250,
    ),
]

# -------------------------------------------------------------
# MINK IK 解算器设置（不变）
# -------------------------------------------------------------

class MinkIKSolver:
    """Wrapper for mink IK solver with SO-101 arm configuration."""
    
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, 
                 arm_joint_names: list, ee_site_name: str = DEFAULT_EE_SITE_NAME):
        """
        Initialize mink solver for SO-101 arm.
        
        Args:
            model: MuJoCo model
            data: MuJoCo data
            arm_joint_names: List of joint names for the arm
            ee_site_name: Name of the end-effector site
        """
        self.model = model
        self.data = data
        self.arm_joint_names = arm_joint_names
        self.ee_site_name = ee_site_name
        self.lock_xy_rotation = LOCK_XY_ROTATION
        self.ee_site_id = self.model.site(self.ee_site_name).id
        
        # 创建水貂配置
        self.configuration = mink.Configuration(model)
        
        # 设置任务
        self.tasks = []
        
        # 末端执行器任务
        self.ee_task = mink.FrameTask(
            frame_name=ee_site_name,
            frame_type="site",
            position_cost=8.0,
            orientation_cost=1.0,
            lm_damping=1.0,
            gain=1.0,
        )
        self.tasks.append(self.ee_task)
        
        # 保持关节靠近首选位置的姿势任务
        self.posture_task = mink.PostureTask(
            model=model,
            cost=1e-2,               # 低优先级
            gain=1.0,                 # 反馈增益
            lm_damping=1.0,            # 减震
        )
        self.tasks.append(self.posture_task)
        
        # 设置限制
        self.limits = []
        
        # 关节位置限制（来自模型）
        self.limits.append(mink.ConfigurationLimit(model=model))
        
        # Velocity limits (tune these for SO-101)
        max_vel = 3.0  # 弧度/秒
        self.limits.append(mink.VelocityLimit(
            model,
            {name: max_vel for name in arm_joint_names}
        ))
        
        # 求解器参数
        self.solver = "daqp"  # or "quadprog" if installed
        self.damping = 1e-6
        self.max_iters = 20
        self.pos_threshold = 1e-4
        self.dt = model.opt.timestep
        
        # 存储首选姿势（当前配置）
        self._update_preferred_posture()

    def _get_current_site_yaw(self) -> float:
        rot = self.data.site_xmat[self.ee_site_id].reshape(3, 3)
        return float(np.arctan2(rot[1, 0], rot[0, 0]))

    def _lock_roll_pitch_allow_yaw(self, reference_quat_wxyz: np.ndarray) -> np.ndarray:
        ref_roll, ref_pitch, _ = quat_wxyz_to_rpy(reference_quat_wxyz)
        current_yaw = self._get_current_site_yaw()
        return rpy_to_quat_wxyz(ref_roll, ref_pitch, current_yaw)
        
    def _update_preferred_posture(self):
        """Update the posture task target to current configuration."""
        self.configuration.update(self.data.qpos)
        self.posture_task.set_target_from_configuration(self.configuration)
        
    def solve(self, target_xyz: np.ndarray, target_quat: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """
        Solve IK for target position and orientation.
        
        Args:
            target_xyz: Target position [x, y, z]
            target_quat: Target orientation as [w, x, y, z] (optional)
            
        Returns:
            Joint positions if solution found, None otherwise
        """
        try:
            # 从当前状态更新配置
            self.configuration.update(self.data.qpos)
            
            # 设置末端执行器目标
            if target_quat is not None:
                target_quat_arr = np.array(target_quat, dtype=np.float64)
                if self.lock_xy_rotation:
                    target_quat_arr = self._lock_roll_pitch_allow_yaw(target_quat_arr)

                # 同时使用位置和方向
                # target_quat 是 [w, x, y, z] ，符合 SO3 的 wxyz 约定
                target_se3 = mink.SE3.from_rotation_and_translation(
                    rotation=mink.SO3(wxyz=target_quat_arr),
                    translation=np.array(target_xyz, dtype=np.float64),
                )
            else:
                # 未提供目标四：在跟踪位置的同时保持当前方向
                current_transform = self.configuration.get_transform_frame_to_world(
                    self.ee_site_name, "site"
                )
                target_se3 = mink.SE3.from_rotation_and_translation(
                    rotation=current_transform.rotation(),
                    translation=np.array(target_xyz, dtype=np.float64),
                )
            
            self.ee_task.set_target(target_se3)
            
            # 更新姿势任务（可选 - 取消注释以保持当前姿势）
            # self._update_preferred_posture()
            
            # 求解 IK
            vel = mink.solve_ik(
                configuration=self.configuration,
                tasks=self.tasks,
                dt=self.dt,
                solver=self.solver,
                damping=self.damping,
                limits=self.limits,
            )
            
            # 整合解决方案
            self.configuration.integrate_inplace(vel, self.dt)
            
            # 提取关节位置
            q_solution = []
            for name in self.arm_joint_names:
                joint_id = self.model.joint(name).id
                qpos_addr = self.model.jnt_qposadr[joint_id]
                q_solution.append(self.configuration.q[qpos_addr])
            
            return np.array(q_solution)
            
        except Exception as e:
            print(f"[MINK IK] Solver failed: {e}")
            return None


# -------------------------------------------------------------
# FSM STATE（使用 Mink 解算器更新）
# -------------------------------------------------------------

def make_fsm():
    return dict(
        phase=0,
        tick=0,
        pick_xyz=None,
        bin_xyz=None,
        home_xyz=None,
        approach_z=None,
        box_approach_z=None,
        ik_failures=0,
        ee_site_name=DEFAULT_EE_SITE_NAME,
        mink_solver=None,  # 将存储水貂解算器实例
    )

# -------------------------------------------------------------
# 平滑曲线（与之前相同）
# -------------------------------------------------------------

def smoothstep(x):
    return x * x * (3 - 2 * x)

def next_ee_waypoint(env, target_xyz, get_ee_xyz_fn, speed: float = 6.0):
    """
    Simple first-order trajectory towards target in Cartesian space.
    """
    ee_xyz = get_ee_xyz_fn(env)
    direction = target_xyz - ee_xyz
    dist = float(np.linalg.norm(direction))
    if dist < 1e-6:
        return target_xyz.copy()
    step = min(1.0, speed / max(dist, 1e-6))
    return ee_xyz + step * direction

# -------------------------------------------------------------
# IK 规划（使用水貂）
# -------------------------------------------------------------

def plan_ik_with_mink(fsm: dict, target_xyz: np.ndarray, 
                      use_top_down: bool = True) -> Optional[np.ndarray]:
    """
    Plan IK using mink solver.
    
    Args:
        fsm: FSM state dictionary containing mink_solver
        target_xyz: Target position
        use_top_down: Whether to enforce top-down orientation
        
    Returns:
        Joint positions if solution found, None otherwise
    """
    solver = fsm.get("mink_solver")
    if solver is None:
        return None
    
    # Try different Z offsets for retry logic
    for dz in [0.0, -0.01, -0.02, 0.01, -0.03, 0.02]:
        trial_xyz = target_xyz.copy()
        trial_xyz[2] += dz

        if use_top_down:
            q = solver.solve(trial_xyz, TOP_DOWN_QUAT)
            if q is not None:
                return q
        else:
            q = solver.solve(trial_xyz)
            if q is not None:
                return q
    
    return None

def get_current_orientation(env, ee_site_name: str = DEFAULT_EE_SITE_NAME) -> np.ndarray:
    """Get current end-effector orientation as quaternion (wxyz)."""
    resolved_site_name = resolve_ee_site_name(env, ee_site_name)
    site_id = env.model.site(resolved_site_name).id
    # MuJoCo 将位点方向存储为 xmat 中的单位四元数（作为旋转矩阵）
    # 我们需要将旋转矩阵转换为四元数
    rotmat = env.data.site_xmat[site_id].reshape(3, 3)
    
    # 将旋转矩阵转换为四元数 (wxyz)
    # 使用标准算法
    trace = rotmat[0,0] + rotmat[1,1] + rotmat[2,2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (rotmat[2,1] - rotmat[1,2]) * s
        y = (rotmat[0,2] - rotmat[2,0]) * s
        z = (rotmat[1,0] - rotmat[0,1]) * s
    else:
        if rotmat[0,0] > rotmat[1,1] and rotmat[0,0] > rotmat[2,2]:
            s = 2.0 * np.sqrt(1.0 + rotmat[0,0] - rotmat[1,1] - rotmat[2,2])
            w = (rotmat[2,1] - rotmat[1,2]) / s
            x = 0.25 * s
            y = (rotmat[0,1] + rotmat[1,0]) / s
            z = (rotmat[0,2] + rotmat[2,0]) / s
        elif rotmat[1,1] > rotmat[2,2]:
            s = 2.0 * np.sqrt(1.0 + rotmat[1,1] - rotmat[0,0] - rotmat[2,2])
            w = (rotmat[0,2] - rotmat[2,0]) / s
            x = (rotmat[0,1] + rotmat[1,0]) / s
            y = 0.25 * s
            z = (rotmat[1,2] + rotmat[2,1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + rotmat[2,2] - rotmat[0,0] - rotmat[1,1])
            w = (rotmat[1,0] - rotmat[0,1]) / s
            x = (rotmat[0,2] + rotmat[2,0]) / s
            y = (rotmat[1,2] + rotmat[2,1]) / s
            z = 0.25 * s
    
    return quat_normalize_wxyz(np.array([w, x, y, z]))

# -------------------------------------------------------------
# 掌握验证（与之前相同）
# -------------------------------------------------------------

def cube_lifted(env, cube_xyz_initial, cube_fn, min_z_delta=0.01):
    cube_now = cube_fn(env)
    return cube_now[2] > cube_xyz_initial[2] + min_z_delta

# -------------------------------------------------------------
# 初始化功能
# -------------------------------------------------------------

def init_mink_for_so101(env, arm_joint_names: list, ee_site_name: str = DEFAULT_EE_SITE_NAME):
    """
    Initialize mink solver for SO-101 arm and add to FSM.
    
    Args:
        env: Environment with model and data attributes
        arm_joint_names: List of joint names for the arm (e.g., 
                        ["joint1", "joint2", "joint3", "joint4", "joint5"])
        ee_site_name: Name of the end-effector site in the model
    """
    # 创建水貂求解器
    solver = MinkIKSolver(
        model=env.model,
        data=env.data,
        arm_joint_names=arm_joint_names,
        ee_site_name=ee_site_name
    )
    
    return solver

# -------------------------------------------------------------
# MAIN FSM STEP (modified for rotation-before-descent)
# -------------------------------------------------------------

def fsm_step(
    env,
    fsm,
    get_ee_xyz_fn,
    cube_pos_fn,
):
    """
    FSM step function using mink for IK.
    
    Args:
        env: Environment with model and data
        fsm: FSM state dictionary
        get_ee_xyz_fn: Function to get current end-effector position
        cube_pos_fn: Function to get current cube position
    """
    ee_xyz = get_ee_xyz_fn(env)
    cube_xyz_live = cube_pos_fn(env)
    _, bin_xyz_live = env.get_obj_pose()

    phase_cfg = PHASES[fsm["phase"]]

    if phase_cfg["name"] != "descend":
        fsm["descend_xy_recover"] = False

    # 在每个进近阶段条目上捕获拾取位置，以便重试跟踪
    # 当前立方体位置（尝试失败后立方体可能会漂移）。
    if fsm["pick_xyz"] is None or (phase_cfg["name"] == "approach_cube" and fsm["tick"] == 0):
        fsm["pick_xyz"] = cube_xyz_live.copy()
    if fsm["bin_xyz"] is None:
        fsm["bin_xyz"] = np.asarray(bin_xyz_live, dtype=np.float32)

    cube_xyz = fsm["pick_xyz"]
    # 仅在预抓取定位阶段跟踪实时立方体。
    # 对于旋转/下降，保持锁定的拾取目标，这样我们就不会追逐移动的物体
    # 立方体已经被偶然的接触所推动。
    if phase_cfg["name"] in {"approach_cube", "align_cube"}:
        cube_xyz = cube_xyz_live.copy()
        fsm["pick_xyz"] = cube_xyz.copy()
    bin_xyz = fsm["bin_xyz"]
    target_fn = phase_cfg["target"]
    gripper = phase_cfg["gripper"]

    if FSM_DEBUG and fsm["tick"] == 0:
        print(
            f"[FSM DEBUG] ENTER phase={fsm['phase']} name={phase_cfg['name']} "
            f"gripper={gripper:.2f}"
        )

    # -------------------------------------------------
    # 更新目标位置
    # -------------------------------------------------
    should_refresh_target = False
    if fsm["planned_phase"] != fsm["phase"]:
        should_refresh_target = True
    elif phase_cfg["name"] in REFRESH_PHASES and fsm["tick"] > 0:
        should_refresh_target = (fsm["tick"] % TARGET_REFRESH_EVERY) == 0

    if should_refresh_target:
        prev_target = fsm.get("phase_target_xyz")
        if target_fn is None:
            new_target = None
        else:
            new_target = np.asarray(target_fn(cube_xyz, bin_xyz), dtype=np.float32)

        fsm["phase_target_xyz"] = new_target
        fsm["planned_phase"] = fsm["phase"]

        # IMPORTANT: if target changes mid-phase, reset trajectory state so
        # `next_ee_waypoint`不会继续集成到过时的 traj_target。
        target_changed = False
        if (prev_target is None) != (new_target is None):
            target_changed = True
        elif prev_target is not None and new_target is not None:
            target_changed = float(np.linalg.norm(new_target - prev_target)) > 1e-4

        if target_changed:
            fsm["traj_start"] = None
            fsm["traj_target"] = None
            fsm["traj_progress"] = 0.0

    if target_fn is None:
        target_xyz = ee_xyz.copy()
    else:
        target_xyz = fsm["phase_target_xyz"]

    completion_target_xyz = target_xyz.copy() if target_xyz is not None else None

    align_phase_idx = 1

    # 如果显式旋转阶段被禁用，则完全跳过它。
    if phase_cfg["name"] == "rotate_to_topdown" and not ENABLE_ROTATE_PHASE:
        if FSM_DEBUG and fsm["tick"] == 0:
            print("[FSM DEBUG] rotate_to_topdown disabled -> skipping to descend")
        fsm["phase"] = min(fsm["phase"] + 1, len(PHASES) - 1)
        fsm["planned_phase"] = None
        fsm["phase_target_xyz"] = None
        fsm["tick"] = 0
        fsm["traj_start"] = None
        fsm["traj_target"] = None
        fsm["traj_progress"] = 0.0
        fsm["descend_xy_recover"] = False
        action = np.zeros(env.n_arm_joints + 1, dtype=np.float32)
        action[-1] = PHASES[fsm["phase"]]["gripper"]
        return action

    # 如果旋转偏离接近锚点太远，请在之前重新运行对齐
    # 尝试更多的方向校正。
    if phase_cfg["name"] == "rotate_to_topdown":
        rotate_pos_err = float(np.linalg.norm(ee_xyz - target_xyz))
        if fsm["tick"] >= 5 and rotate_pos_err > DESCEND_REALIGN_XY_ERR:
            if FSM_DEBUG:
                print(
                    f"[FSM DEBUG] rotate drift -> align: pos_err={rotate_pos_err:.4f}>"
                    f"{DESCEND_REALIGN_XY_ERR:.4f}"
                )
            fsm["phase"] = align_phase_idx
            fsm["planned_phase"] = None
            fsm["phase_target_xyz"] = None
            fsm["tick"] = 0
            fsm["traj_start"] = None
            fsm["traj_target"] = None
            fsm["traj_progress"] = 0.0
            fsm["descend_xy_recover"] = True
            action = np.zeros(env.n_arm_joints + 1, dtype=np.float32)
            action[-1] = gripper
            return action

    # Never continue descent when lateral alignment is poor; go back to align.
    if phase_cfg["name"] == "descend":
        descend_xy_err = float(np.linalg.norm((ee_xyz - target_xyz)[:2]))
        if descend_xy_err > DESCEND_XY_RECOVER_ENTER:
            if FSM_DEBUG:
                print(
                    f"[FSM DEBUG] descend xy recover -> align: xy_err={descend_xy_err:.4f}>"
                    f"{DESCEND_XY_RECOVER_ENTER:.4f}"
                )
            fsm["phase"] = align_phase_idx
            fsm["planned_phase"] = None
            fsm["phase_target_xyz"] = None
            fsm["tick"] = 0
            fsm["traj_start"] = None
            fsm["traj_target"] = None
            fsm["traj_progress"] = 0.0
            fsm["descend_xy_recover"] = True
            action = np.zeros(env.n_arm_joints + 1, dtype=np.float32)
            action[-1] = gripper
            return action

    # 如果之前的下降/上升失败，请先垂直上升至安全悬停高度
    # 在当前 XY 处，以避免在横向移动之前在桌面上/附近拖动。
    if fsm.get("recover_lift_pending") and phase_cfg["name"] == "approach_cube":
        safe_hover_z = float(cube_xyz[2] + APPROACH_Z_OFFSET)
        if ee_xyz[2] < safe_hover_z - RECOVERY_Z_MARGIN:
            target_xyz = np.array([ee_xyz[0], ee_xyz[1], safe_hover_z], dtype=np.float32)
            if FSM_DEBUG and (fsm["tick"] == 0 or (fsm["tick"] % FSM_DEBUG_EVERY) == 0):
                print(
                    f"[FSM DEBUG] recovery-lift active: ee_z={ee_xyz[2]:.4f} -> safe_hover_z={safe_hover_z:.4f}"
                )
        else:
            fsm["recover_lift_pending"] = False
            fsm["traj_start"] = None
            fsm["traj_target"] = None
            fsm["traj_progress"] = 0.0

    # 简单方法政策：
    # 1) 在 XY 方向上移动立方体，同时保持安全运输高度，
    # 2) 然后才稳定到立方体上方的悬停高度。
    if phase_cfg["name"] == "approach_cube":
        hover_z = float(cube_xyz[2] + APPROACH_Z_OFFSET)
        transit_z = max(float(ee_xyz[2]), hover_z)
        final_hover_target = np.array([target_xyz[0], target_xyz[1], hover_z], dtype=np.float32)
        completion_target_xyz = final_hover_target.copy()
        xy_to_hover = float(np.linalg.norm(ee_xyz[:2] - final_hover_target[:2]))

        if xy_to_hover > 0.02:
            staged_target = np.array([final_hover_target[0], final_hover_target[1], transit_z], dtype=np.float32)
        else:
            staged_target = final_hover_target

        if float(np.linalg.norm(staged_target - target_xyz)) > 1e-4:
            fsm["traj_start"] = None
            fsm["traj_target"] = None
            fsm["traj_progress"] = 0.0

        target_xyz = staged_target

    # 保持 XY 对齐不接近，并在下降和旋转过程中重复使用。
    if phase_cfg["name"] == "approach_cube":
        fsm["approach_xy"] = target_xyz[:2].copy()
    elif phase_cfg["name"] in ["descend", "rotate_to_topdown"] and fsm.get("approach_xy") is not None:
        target_xyz = target_xyz.copy()
        target_xyz[:2] = fsm["approach_xy"]

    # -------------------------------------------------
    # 使用 mink 计算路点并求解 IK
    # -------------------------------------------------
    waypoint = next_ee_waypoint(env, target_xyz, get_ee_xyz_fn, speed=6.0)

    # 在恢复提升期间，在当前 XY 处严格垂直移动。
    if fsm.get("recover_lift_pending") and phase_cfg["name"] == "approach_cube":
        waypoint = target_xyz.copy()

    wp_err = waypoint - ee_xyz
    wp_dist = float(np.linalg.norm(wp_err))
    
    # 根据阶段确定定向策略
    if phase_cfg["name"] == "rotate_to_topdown":
        # 在旋转阶段，重点是在保持位置的同时实现自上而下的方向
        use_top_down = True
    elif phase_cfg["name"] == "descend":
        # 在下降阶段，保持自上而下的方向
        use_top_down = ENFORCE_TOPDOWN_ON_DESCEND
    else:
        # 其他阶段：保持当前方向（无方向约束）
        use_top_down = False

    # 对于下降阶段，使用直接目标而不是增量步进
    if phase_cfg["name"] == "descend":
        # 直接瞄准最终位置 - 这更干净并避免振荡
        waypoint = target_xyz.copy()
        
        # But limit downward speed for safety
        current_z = ee_xyz[2]
        target_z = target_xyz[2]
        if current_z - target_z > DESCEND_Z_STEP:
            waypoint[2] = current_z - DESCEND_Z_STEP

    q_target = plan_ik_with_mink(fsm, waypoint, use_top_down=use_top_down)

    ee_site_name = fsm.get("ee_site_name")
    if not ee_site_name:
        solver = fsm.get("mink_solver")
        ee_site_name = getattr(solver, "ee_site_name", DEFAULT_EE_SITE_NAME)

    if q_target is None:
        # 如果 mink 失败，则留在当前位置
        print(f"[FSM] Mink IK failed for phase {phase_cfg['name']}, holding position")
        q_target = env.get_joint_state()[:env.n_arm_joints].copy()
        fsm["ik_failures"] += 1
    else:
        fsm["ik_failures"] = 0

    # -------------------------------------------------
    # 生成动作
    # -------------------------------------------------
    q_current = env.get_joint_state()[:env.n_arm_joints]
    dq = np.clip(q_target - q_current, -MAX_DQ_STEP, MAX_DQ_STEP)

    action = np.zeros(env.n_arm_joints + 1, dtype=np.float32)
    action[:env.n_arm_joints] = dq
    action[-1] = gripper

    # -------------------------------------------------
    # 竣工检查
    # -------------------------------------------------
    done = False
    done_reason = ""
    xy_err = np.nan
    z_err = np.nan
    dist = np.nan
    
    if phase_cfg["name"] == "rotate_to_topdown":
        # Special completion check for rotation phase
        current_quat = get_current_orientation(env, ee_site_name)
        angle_diff = quat_angle_between(current_quat, TOP_DOWN_QUAT)
        axis_err = topdown_axis_error(current_quat)
        
        # 还要检查位置公差以确保我们仍然处于接近高度
        err = ee_xyz - completion_target_xyz
        xy_err = np.linalg.norm(err[:2])
        z_err = abs(err[2])
        dist = np.linalg.norm(err)
        pos_err = dist
        
        done = (axis_err < ROTATION_THRESHOLD) and (pos_err < 0.025)
        done_reason = (
            f"axis_err={axis_err:.4f}<{ROTATION_THRESHOLD:.4f} "
            f"and pos_err={pos_err:.4f}<0.025"
        )
        
        if FSM_DEBUG and (fsm["tick"] % FSM_DEBUG_EVERY) == 0:
            print(
                f"[FSM DEBUG] rotation progress: axis_err={axis_err:.4f}, "
                f"quat_err={angle_diff:.4f}, pos_err={pos_err:.4f}"
            )
            
    elif phase_cfg["tol"] is not None:
        err = ee_xyz - completion_target_xyz
        xy_err = np.linalg.norm(err[:2])
        z_err = abs(err[2])
        dist = np.linalg.norm(err)

        if phase_cfg["name"] == "approach_cube":
            done = (xy_err < 0.015) and (z_err < 0.030)
            done_reason = f"xy_err={xy_err:.4f}<0.015 and z_err={z_err:.4f}<0.030"
        elif phase_cfg["name"] == "align_cube":
            done = (xy_err < 0.012) and (z_err < 0.020)
            done_reason = f"xy_err={xy_err:.4f}<0.012 and z_err={z_err:.4f}<0.020"
        elif phase_cfg["name"] == "descend":
            pos_ok = (xy_err < 0.012) and (z_err < 0.010)
            done = pos_ok
            done_reason = f"xy_err={xy_err:.4f}<0.012, z_err={z_err:.4f}<0.010"
        else:
            done = dist < phase_cfg["tol"]
            done_reason = f"dist={dist:.4f}<tol={phase_cfg['tol']:.4f}"
    else:
        done = fsm["tick"] >= phase_cfg["timeout"]
        done_reason = f"tick={fsm['tick']}>=timeout={phase_cfg['timeout']}"

    if FSM_DEBUG and (fsm["tick"] == 0 or (fsm["tick"] % FSM_DEBUG_EVERY) == 0):
        print(
            f"[FSM DEBUG] phase={phase_cfg['name']} tick={fsm['tick']} "
            f"ee={np.round(ee_xyz, 4)} target={np.round(target_xyz, 4)} "
            f"waypoint={np.round(waypoint, 4)} wp_dist={wp_dist:.4f} "
            f"xy_err={xy_err:.4f} z_err={z_err:.4f} dist={dist:.4f} done={done}"
        )

    timed_out = fsm["tick"] > phase_cfg["timeout"]
    allow_timeout_advance = False
    if timed_out:
        print(f"[FSM] timeout phase {phase_cfg['name']}")
        if FSM_DEBUG:
            print(
                f"[FSM DEBUG] TIMEOUT phase={phase_cfg['name']} tick={fsm['tick']} "
                f"ee={np.round(ee_xyz, 4)} target={np.round(target_xyz, 4)}"
            )

        # 如果对齐非常接近，但在残余手腕/方向运动上停滞，
        # 接受并继续轮换阶段而不是重新启动。
        if phase_cfg["name"] == "align_cube" and np.isfinite(xy_err) and np.isfinite(z_err):
            if (xy_err < 0.014) and (z_err < 0.025):
                done = True
                allow_timeout_advance = True
                done_reason = f"align timeout fallback: xy_err={xy_err:.4f}<0.014 and z_err={z_err:.4f}<0.025"
            else:
                done = False
                done_reason = f"timeout tick={fsm['tick']}>{phase_cfg['timeout']}"
        elif phase_cfg["name"] == "rotate_to_topdown":
            # 如果旋转超时但我们足够接近，则无论如何都会继续下降
            current_quat = get_current_orientation(env, ee_site_name)
            axis_err = topdown_axis_error(current_quat)
            if axis_err < ROTATION_THRESHOLD * 2.0:  # 2x 阈值以内
                done = True
                allow_timeout_advance = True
                done_reason = f"rotate timeout fallback: axis_err={axis_err:.4f}"
            else:
                done = False
                done_reason = f"timeout tick={fsm['tick']}>{phase_cfg['timeout']}"
        else:
            done = False
            done_reason = f"timeout tick={fsm['tick']}>{phase_cfg['timeout']}"

    # Critical phases must not advance on timeout; restart pick attempt instead.
    if timed_out and (not allow_timeout_advance) and phase_cfg["name"] in {"approach_cube", "align_cube", "rotate_to_topdown", "descend", "lift"}:
        if phase_cfg["name"] in {"rotate_to_topdown", "descend"}:
            print(f"[FSM] {phase_cfg['name']} timeout -> return to align")
            fsm["phase"] = 1
            fsm["planned_phase"] = None
            fsm["phase_target_xyz"] = None
            fsm["tick"] = 0
            fsm["traj_start"] = None
            fsm["traj_target"] = None
            fsm["traj_progress"] = 0.0
            fsm["descend_xy_recover"] = True
            return action

        print(f"[FSM] {phase_cfg['name']} timeout -> restart from approach")
        if phase_cfg["name"] in {"descend", "lift"}:
            fsm["recover_lift_pending"] = True
        fsm["phase"] = 0
        fsm["planned_phase"] = None
        fsm["phase_target_xyz"] = None
        fsm["pick_xyz"] = None
        fsm["tick"] = 0
        fsm["traj_start"] = None
        fsm["traj_target"] = None
        fsm["traj_progress"] = 0.0
        fsm["approach_xy"] = None
        fsm["descend_xy_recover"] = False
        fsm["lift_fail_streak"] = 0
        return action

    # -------------------------------------------------
    # 掌握验证
    # -------------------------------------------------
    if phase_cfg["name"] == "lift":
        LIFT_CHECK_DELAY = 12
        LIFT_MIN_Z_DELTA = 0.004
        LIFT_FAIL_STREAK_MAX = 10

        if fsm["tick"] > LIFT_CHECK_DELAY:
            if cube_lifted(env, cube_xyz, cube_pos_fn, min_z_delta=LIFT_MIN_Z_DELTA):
                fsm["lift_fail_streak"] = 0
            else:
                fsm["lift_fail_streak"] += 1

            if fsm["lift_fail_streak"] >= LIFT_FAIL_STREAK_MAX:
                print("[FSM] grasp failed, restarting")
                fsm["recover_lift_pending"] = True
                fsm["phase"] = 0
                fsm["planned_phase"] = None
                fsm["phase_target_xyz"] = None
                fsm["pick_xyz"] = None
                fsm["tick"] = 0
                fsm["traj_start"] = None
                fsm["traj_target"] = None
                fsm["traj_progress"] = 0.0
                fsm["descend_xy_recover"] = False
                fsm["lift_fail_streak"] = 0
                return action

    # -------------------------------------------------
    # 相变
    # -------------------------------------------------
    if done and fsm["phase"] < len(PHASES) - 1:
        if FSM_DEBUG:
            print(
                f"[FSM DEBUG] COMPLETE phase={phase_cfg['name']} "
                f"reason=({done_reason}) -> next={PHASES[fsm['phase'] + 1]['name']}"
            )
        fsm["phase"] += 1
        if fsm["phase"] < len(PHASES) and PHASES[fsm["phase"]]["name"] == "rotate_to_topdown" and not ENABLE_ROTATE_PHASE:
            if FSM_DEBUG:
                print("[FSM DEBUG] skipping rotate_to_topdown phase")
            fsm["phase"] += 1
        fsm["planned_phase"] = None
        fsm["phase_target_xyz"] = None
        if PHASES[fsm["phase"]]["name"] not in ["descend", "rotate_to_topdown"]:
            fsm["approach_xy"] = None
        if PHASES[fsm["phase"]]["name"] != "descend":
            fsm["descend_xy_recover"] = False
        fsm["lift_fail_streak"] = 0
        fsm["tick"] = 0
        fsm["traj_start"] = None
        fsm["traj_progress"] = 0.0
    else:
        if FSM_DEBUG and done and fsm["phase"] == len(PHASES) - 1:
            print(
                f"[FSM DEBUG] COMPLETE terminal phase={phase_cfg['name']} reason=({done_reason})"
            )
        fsm["tick"] += 1

    return action


# -------------------------------------------------------------
# 使用示例
# -------------------------------------------------------------

def setup_so101_controller(env, arm_joint_names=None, ee_site_name=DEFAULT_EE_SITE_NAME):
    """
    Setup function to initialize FSM with mink for SO-101.
    
    Args:
        env: Environment with model and data
        arm_joint_names: List of joint names (if None, will try to detect)
        ee_site_name: Name of end-effector site
        
    Returns:
        fsm: Initialized FSM state
    """
    # 如果未提供联名，请尝试检测它们
    if arm_joint_names is None:
        # SO-101 arm joint names (matches y_env.py robot_profile='so101')
        arm_joint_names = [
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
        ]
    
    # 创建有限状态机
    fsm = make_fsm()

    resolved_ee_site_name = resolve_ee_site_name(env, ee_site_name)
    fsm["ee_site_name"] = resolved_ee_site_name

    # Compute a safe "home" Cartesian pose (used for unstow + return_home).
    # For SO101 resets, the arm starts stowed; we want a neutral pose over the workspace
    # 在尝试拾放之前。
    cube_xyz, bin_xyz = env.get_obj_pose()
    cube_xyz = np.asarray(cube_xyz, dtype=np.float32)
    bin_xyz = np.asarray(bin_xyz, dtype=np.float32)
    home_xy = 0.5 * (cube_xyz[:2] + bin_xyz[:2])
    home_z = float(max(cube_xyz[2], bin_xyz[2]) + APPROACH_Z_OFFSET + 0.05)
    fsm["home_xyz"] = np.asarray([home_xy[0], home_xy[1], home_z], dtype=np.float32)

    # 初始化并附加 mink 解算器
    fsm["mink_solver"] = init_mink_for_so101(
        env=env,
        arm_joint_names=arm_joint_names,
        ee_site_name=resolved_ee_site_name
    )

    return fsm


# -------------------------------------------------------------
# 简单的 7 相 FSM 步骤
# -------------------------------------------------------------

def _phase_target_xyz(fsm, phase_cfg, cube_xyz, bin_xyz):
    """
    Compute Cartesian target for the current phase.
    """
    name = phase_cfg["name"]
    if phase_cfg["target"] is None:
        if name in {"stow_to_home", "return_home"}:
            # 返回录制的起始姿势
            return np.asarray(fsm["home_xyz"], dtype=np.float32)
        # 对于纯粹的计时阶段（“抓地力”），请留在我们现在的位置
        return None

    return np.asarray(phase_cfg["target"](cube_xyz, bin_xyz), dtype=np.float32)


def fsm_step(
    env,
    fsm,
    get_ee_xyz_fn,
    cube_pos_fn,
):
    """
    Very simple 7‑phase pick-and-place:
      1) above_block     – hover above cube, gripper open
      2) descend         – move straight down
      3) grip            – close gripper
      4) lift            – move straight up
      5) move_to_box     – move above box
      6) place_in_box    – lower and open gripper
      7) return_home     – go back to initial pose
    """
    ee_xyz = get_ee_xyz_fn(env)
    cube_xyz_live = cube_pos_fn(env)
    _, bin_xyz_live = env.get_obj_pose()

    # 在剧集开始时缓存一次拾取和盒子位置
    if fsm["pick_xyz"] is None:
        fsm["pick_xyz"] = cube_xyz_live.copy()
    if fsm["bin_xyz"] is None:
        fsm["bin_xyz"] = np.asarray(bin_xyz_live, dtype=np.float32)

    cube_xyz = fsm["pick_xyz"]
    bin_xyz = fsm["bin_xyz"]

    phase_cfg = PHASES[fsm["phase"]]
    phase_name = phase_cfg["name"]

    if FSM_DEBUG and (fsm["tick"] == 0 or (fsm["tick"] % FSM_DEBUG_EVERY) == 0):
        print(
            f"[FSM] phase={phase_name} idx={fsm['phase']} tick={fsm['tick']} "
            f"cube={np.round(cube_xyz, 3)} box={np.round(bin_xyz, 3)}"
        )

    # ------------------------------------------------------------------
    # Determine Cartesian target for this phase
    # ------------------------------------------------------------------
    target_xyz = _phase_target_xyz(fsm, phase_cfg, cube_xyz, bin_xyz)

    # 对于接近块阶段，我们想要一条成形的路径，而不是一条直线
    # XYZ 线。该动议是：
    # 1) 纯 XY 处于固定高的 Approach_z 直到越过立方体，
    # 2）然后纯Z下降到最终悬停高度。
    if phase_name == "above_block" and target_xyz is not None:
        ee_z = float(ee_xyz[2])
        cube_hover_z = float(cube_xyz[2] + APPROACH_Z_OFFSET)

        # 第一次进入此阶段时初始化固定的进场高度。
        if fsm.get("approach_z") is None or fsm["tick"] == 0:
            margin = 0.05  # 立方体悬停高度上方的额外间隙
            fsm["approach_z"] = max(ee_z, cube_hover_z + margin)

        approach_z = float(fsm["approach_z"])

        # 根据当前姿势选择阶段
        lateral_vec = target_xyz[:2] - ee_xyz[:2]
        lateral_dist = float(np.linalg.norm(lateral_vec))

        VERT_STEP = 0.035
        XY_STEP = 0.05

        if lateral_dist > 0.02:
            # 第 1 阶段：仅在 XY 中以固定的逼近_z 移动。
            if lateral_dist > 1e-6:
                direction_xy = lateral_vec / lateral_dist
            else:
                direction_xy = np.zeros(2, dtype=np.float32)
            step = min(XY_STEP, lateral_dist)
            new_xy = ee_xyz[:2] + step * direction_xy
            target_xyz = np.array(
                [new_xy[0], new_xy[1], approach_z],
                dtype=np.float32,
            )
        else:
            # Stage 2: we're roughly above the cube; descend vertically to hover.
            if ee_z > cube_hover_z + 1e-3:
                new_z = max(ee_z - VERT_STEP, cube_hover_z)
            else:
                new_z = cube_hover_z
            target_xyz = np.array(
                [target_xyz[0], target_xyz[1], new_z],
                dtype=np.float32,
            )

    # 对于移动到盒子阶段，我们想要相同形状的路径：
    # 1) 纯 XY 处于固定的高 box_approach_z 直到越过盒子，
    # 2）然后纯Z下降到盒子悬停高度。
    if phase_name == "move_to_box" and target_xyz is not None:
        ee_z = float(ee_xyz[2])
        box_hover_z = float(bin_xyz[2] + APPROACH_Z_OFFSET)

        if fsm.get("box_approach_z") is None or fsm["tick"] == 0:
            margin = 0.05
            fsm["box_approach_z"] = max(ee_z, box_hover_z + margin)

        box_approach_z = float(fsm["box_approach_z"])

        lateral_vec = target_xyz[:2] - ee_xyz[:2]
        lateral_dist = float(np.linalg.norm(lateral_vec))

        VERT_STEP = 0.035
        XY_STEP = 0.055

        if lateral_dist > 0.02:
            # 第 1 阶段：仅在固定 box_approach_z 处沿 XY 移动。
            if lateral_dist > 1e-6:
                direction_xy = lateral_vec / lateral_dist
            else:
                direction_xy = np.zeros(2, dtype=np.float32)
            step = min(XY_STEP, lateral_dist)
            new_xy = ee_xyz[:2] + step * direction_xy
            target_xyz = np.array(
                [new_xy[0], new_xy[1], box_approach_z],
                dtype=np.float32,
            )
        else:
            # Stage 2: we're roughly above the box; keep hover Z (no further descent here).
            target_xyz = np.array(
                [target_xyz[0], target_xyz[1], box_hover_z],
                dtype=np.float32,
            )

    # Gripper command for this phase
    gripper_cmd = float(phase_cfg["gripper"])
    # 在 place_in_box 中：保持夹具关闭，直到手臂停在目标处，然后释放
    if phase_name == "place_in_box" and target_xyz is not None and phase_cfg.get("tol") is not None:
        place_dist = float(np.linalg.norm(ee_xyz - target_xyz))
        if place_dist >= phase_cfg["tol"]:
            gripper_cmd = 1.0  # 仍在移动 – 保持关闭
        # 否则在目标处使用一次phase_cfg["gripper"] (0.0)

    # ------------------------------------------------------------------
    # IK+动作
    # ------------------------------------------------------------------
    if target_xyz is None:
        # 无运动阶段 - 保持关节，仅驱动夹具
        q_current = env.get_joint_state()[:env.n_arm_joints]
        dq = np.zeros_like(q_current, dtype=np.float32)
        waypoint = ee_xyz.copy()
    else:
        # 对于接近阶段，我们已经调整了 target_xyz 的形状以进入
        # pure-Z / pure-XY segments; use it directly as the waypoint so the
        # 路径遵循该形状而不是直线。
        if phase_name == "above_block":
            waypoint = target_xyz.copy()
        elif phase_name == "place_in_box":
            # 具有有限 Z 步长的直接目标以避免过冲/振荡
            # 在放下方块之前（与下降阶段相同的想法）。
            waypoint = np.array(target_xyz, dtype=np.float32)
            current_z = float(ee_xyz[2])
            target_z = float(target_xyz[2])
            if current_z - target_z > PLACE_DESCEND_Z_STEP:
                waypoint[2] = current_z - PLACE_DESCEND_Z_STEP
        else:
            waypoint = next_ee_waypoint(env, target_xyz, get_ee_xyz_fn, speed=6.0)

        q_target = plan_ik_with_mink(fsm, waypoint, use_top_down=True)

        if q_target is None:
            print(f"[FSM] Mink IK failed in phase '{phase_name}', holding position.")
            q_target = env.get_joint_state()[:env.n_arm_joints].copy()
            fsm["ik_failures"] += 1
        else:
            fsm["ik_failures"] = 0

        q_current = env.get_joint_state()[:env.n_arm_joints]
        dq = np.clip(q_target - q_current, -MAX_DQ_STEP, MAX_DQ_STEP)
        # 在 place_in_box 中，靠近目标时阻尼运动以避免掉落时振荡
        if phase_name == "place_in_box" and target_xyz is not None:
            place_dist = float(np.linalg.norm(ee_xyz - target_xyz))
            if place_dist < PLACE_DAMP_DIST:
                scale = place_dist / PLACE_DAMP_DIST
                dq = dq * scale

    action = np.zeros(env.n_arm_joints + 1, dtype=np.float32)
    action[:env.n_arm_joints] = dq
    action[-1] = gripper_cmd

    # ------------------------------------------------------------------
    # 调试：监控 Z 高度以了解接近行为
    # ------------------------------------------------------------------
    if FSM_DEBUG and (fsm["tick"] % FSM_DEBUG_EVERY) == 0:
        cube_z = float(cube_xyz[2])
        ee_z = float(ee_xyz[2])
        waypoint_z = float(waypoint[2]) if target_xyz is not None else ee_z
        print(
            f"[FSM Z] phase={phase_name} tick={fsm['tick']} "
            f"ee_z={ee_z:.4f} waypoint_z={waypoint_z:.4f} cube_z={cube_z:.4f}"
        )

    # ------------------------------------------------------------------
    # 阶段完成
    # ------------------------------------------------------------------
    done = False
    if phase_cfg["tol"] is None:
        # 纯粹基于时间的阶段（例如“握力”）
        done = fsm["tick"] >= phase_cfg["timeout"]
    else:
        err = ee_xyz - (target_xyz if target_xyz is not None else ee_xyz)
        dist = float(np.linalg.norm(err))
        done = dist < phase_cfg["tol"] or fsm["tick"] >= phase_cfg["timeout"]

        if FSM_DEBUG and (fsm["tick"] % FSM_DEBUG_EVERY) == 0:
            print(
                f"[FSM] phase={phase_name} dist={dist:.4f} "
                f"tol={phase_cfg['tol']:.4f} done={done}"
            )

    # 推进阶段
    if done and fsm["phase"] < len(PHASES) - 1:
        fsm["phase"] += 1
        fsm["tick"] = 0
    else:
        fsm["tick"] += 1

    return action