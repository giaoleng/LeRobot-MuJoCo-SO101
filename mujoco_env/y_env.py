import sys
import random
import numpy as np
import xml.etree.ElementTree as ET
import mujoco
from mujoco_env.mujoco_parser import MuJoCoParserClass
from mujoco_env.utils import prettify, rotation_matrix, add_title_to_img
from mujoco_env.ik import solve_ik
from mujoco_env.transforms import rpy2r, r2rpy, r2quat, quat2r
import os
import copy
import glfw

class SimpleEnv:
    def __init__(self, 
                 xml_path,
                action_type='eef_pose', 
                state_type='joint_angle',
                robot_profile='so101',
                seed = None,
                pick_body_name='body_obj_mug_5',
                place_body_name='body_obj_plate_11',
                spawn_x_range=(0.20, 0.3),
                spawn_y_range=(0.02, 0.1),
                spawn_z_range=(0.815, 0.815),
                spawn_min_dist=0.2,
                spawn_xy_margin=0.0,
                spawn_fallback_min_dist=0.1):
        """
        args:
            xml_path: str, path to the xml file
            action_type: str, type of action space, 'eef_pose','delta_joint_angle' or 'joint_angle'
            state_type: str, type of state space, 'joint_angle' or 'ee_pose'
            robot_profile: str, robot kinematic profile ('omy', 'so100', 'so101')
            seed: int, seed for random number generator
            pick_body_name: str, body name for the pick object in the MuJoCo model
            place_body_name: str, body name for the place/target object in the MuJoCo model
            spawn_x_range: tuple[float, float], x sampling range for objects
            spawn_y_range: tuple[float, float], y sampling range for objects
            spawn_z_range: tuple[float, float], z sampling range for objects
            spawn_min_dist: float, minimum distance between sampled objects
            spawn_xy_margin: float, xy margin applied to ranges during sampling
            spawn_fallback_min_dist: float, retry distance when initial sampling fails
        """
        # 加载 xml 文件
        self.env = MuJoCoParserClass(name='Tabletop',rel_xml_path=xml_path)
        self.action_type = action_type
        self.state_type = state_type
        self.pick_body_name = pick_body_name
        self.place_body_name = place_body_name
        self.spawn_x_range = spawn_x_range
        self.spawn_y_range = spawn_y_range
        self.spawn_z_range = spawn_z_range
        self.spawn_min_dist = spawn_min_dist
        self.spawn_xy_margin = spawn_xy_margin
        self.spawn_fallback_min_dist = spawn_fallback_min_dist
        self.spawn_bin_exclusion_margin = 0.015
        self.robot_profile = robot_profile
        # 取放门控：仅在目标对象完成后才计数成功
        # 显然至少有一次被从桌子上举起来。
        self._target_was_lifted = False
        self._target_rest_height = None
        self._lift_height_delta = 0.035

        if robot_profile == 'omy':
            self.joint_names = ['joint1',
                        'joint2',
                        'joint3',
                        'joint4',
                        'joint5',
                        'joint6',]
            self.tcp_body_name = 'tcp_link'
            self.gripper_joint_name = 'rh_r1'
            self.gripper_actuator_scales = np.array([1.0, 0.8, 1.0, 0.8], dtype=np.float32)
            self.gripper_max_rad = 1.0   # 保留现有的 OMY 行为
            self.gripper_continuous = False
            # 放置后，OMY 臂必须缩回到此 Z（米）以上，以
            # signal episode success; prevents false positives during the pick phase.
            self.success_height_threshold = 0.9
        elif robot_profile in ['so100', 'so101']:
            self.joint_names = [
                'shoulder_pan',
                'shoulder_lift',
                'elbow_flex',
                'wrist_flex',
                'wrist_roll',
            ]
            self.tcp_body_name = 'gripper'
            self.gripper_joint_name = 'gripper'
            self.gripper_actuator_scales = np.array([1.0], dtype=np.float32)
            # 夹爪关节范围：[-0.17453, 1.74533] rad。  行动常态化
            # to [0, 1] where 0 = fully open and 1 = fully closed.
            self.gripper_max_rad = 1.74533
            self.gripper_continuous = True
            # SO100/SO101：相对于垃圾箱检查手臂缩回高度
            # check_success() 中的几何图形而不是绝对值。
            # 此处设置为 None 以选择 SO100/SO101 代码路径。
            self.success_height_threshold = None
            # Conservative XY reach limit for table sampling relative to robot base.
            self.spawn_reach_radius = 0.50
        else:
            raise ValueError(f"Unsupported robot_profile: {robot_profile}")

        self.n_arm_joints = len(self.joint_names)
        self.n_gripper_actuators = len(self.gripper_actuator_scales)
        self._prev_key_state = {}
        self._teleop_debug_counter = 0
        # Disable gripper pad debug overlays by default; they are useful for
        # 调试接触几何形状，但在数据收集过程中视觉上有噪音。
        self.show_gripper_pad_debug = False
        # 鼠标慢跑状态（鼠标中键拖动）
        self._mouse_last_x = None
        self._mouse_last_y = None
        # Mocap IK 跟踪状态
        self._mocap_body_id = -1
        self._mocap_id = -1
        self._last_mocap_pos = None
        self._last_mocap_quat = None
        self.joint_mins = np.array([self.env.model.joint(name).range[0] for name in self.joint_names], dtype=np.float32)
        self.joint_maxs = np.array([self.env.model.joint(name).range[1] for name in self.joint_names], dtype=np.float32)

        available_joint_names = set(self.env.joint_names)
        missing_joints = [jname for jname in self.joint_names if jname not in available_joint_names]
        if missing_joints:
            raise ValueError(
                f"robot_profile='{self.robot_profile}' is incompatible with xml_path='{xml_path}'. "
                f"Missing joints: {missing_joints}. "
                f"Available joints: {self.env.joint_names}"
            )

        if self.gripper_continuous:
            gripper_joint = self.env.model.joint(self.gripper_joint_name)
            gripper_range = np.asarray(gripper_joint.range, dtype=np.float32)
            # SO100/SO101 下颌运动学相对于任务空间 API 是相反的：
            # 较大的原始关节值使下巴张得更宽，而呼叫者则期望
            # 0 = open and 1 = closed.
            self.gripper_closed_rad = float(gripper_range[0])
            self.gripper_open_rad = float(gripper_range[1])
            self.gripper_motion_span = max(
                self.gripper_open_rad - self.gripper_closed_rad, 1e-6
            )
        else:
            self.gripper_closed_rad = float(self.gripper_max_rad)
            self.gripper_open_rad = 0.0
            self.gripper_motion_span = max(self.gripper_closed_rad - self.gripper_open_rad, 1e-6)

        self._gripper_pad_debug_geoms = [
            ("fixed_jaw_pad", [1.0, 0.15, 0.15, 0.45]),
            ("moving_jaw_pad", [0.15, 0.55, 1.0, 0.45]),
            ("gripper_palm_pad", [1.0, 0.8, 0.15, 0.30]),
        ]

        # 将在每次重置时使用采样的生成配置进行填充
        # 所有可移动块对象（不包括固定箱/板）。
        self.spawn_obj_names = []
        self.spawn_obj_xyzs = None

        self.init_viewer()
        self.reset(seed)

    @property
    def model(self):
        """Expose the underlying MuJoCo model directly."""
        return self.env.model

    @property
    def data(self):
        """Expose the underlying MuJoCo data directly."""
        return self.env.data

    def _key_is_down(self, key):
        if self.env.is_key_pressed_repeat(key=key):
            return True

        viewer = getattr(self.env, 'viewer', None)
        window = getattr(viewer, 'window', None)
        if window is None:
            return False

        try:
            return glfw.get_key(window, key) == glfw.PRESS
        except Exception:
            return False

    def _key_is_pressed_once(self, key):
        is_down = self._key_is_down(key)
        was_down = self._prev_key_state.get(key, False)
        self._prev_key_state[key] = is_down
        return is_down and not was_down

    def _make_gripper_ctrl(self, gripper_fraction):
        """
        Convert the normalized task-space command (0=open, 1=closed) to raw
        actuator targets.
        """
        gripper_fraction = float(np.clip(gripper_fraction, 0.0, 1.0))
        if self.gripper_continuous:
            gripper_target = self.gripper_open_rad - gripper_fraction * self.gripper_motion_span
        else:
            gripper_target = gripper_fraction * self.gripper_max_rad
        gripper_cmd = np.array(
            [gripper_target] * self.n_gripper_actuators,
            dtype=np.float32,
        )
        gripper_cmd *= self.gripper_actuator_scales
        return gripper_cmd

    def _get_bin_exclusion_bounds(self):
        """
        Return a rectangular XY exclusion zone for the fixed bin.
        """
        try:
            body_names = self.env.get_body_names(prefix='body_obj_')
            if 'body_obj_bin' not in body_names:
                return None
            p_bin = self.env.get_p_body('body_obj_bin')
        except Exception:
            return None

        # obj_bin.xml外部足迹半尺寸在 x 和 y 方向上为 0.06m。
        half_extent = 0.06 + self.spawn_bin_exclusion_margin
        return (
            p_bin[0] - half_extent,
            p_bin[0] + half_extent,
            p_bin[1] - half_extent,
            p_bin[1] + half_extent,
        )

    def _is_within_reach_xy(self, x, y):
        """
        Check whether an XY point is inside the SO100/SO101 reachable table area.
        """
        if self.robot_profile not in ['so100', 'so101']:
            return True
        try:
            p_base = self.env.get_p_body('base')
        except Exception:
            p_base = np.zeros(3, dtype=np.float32)
        d_xy = np.linalg.norm(np.array([x - p_base[0], y - p_base[1]], dtype=np.float32))
        return bool(d_xy <= self.spawn_reach_radius)

    def _sample_object_xyzs(self, n_obj, min_dist):
        """
        Sample object positions while respecting reachability and bin exclusion.
        """
        x_min, x_max = self.spawn_x_range
        y_min, y_max = self.spawn_y_range
        z_min, z_max = self.spawn_z_range
        m = self.spawn_xy_margin
        x_lo, x_hi = x_min + m, x_max - m
        y_lo, y_hi = y_min + m, y_max - m
        if x_lo > x_hi or y_lo > y_hi:
            raise ValueError("spawn_xy_margin is too large for the configured spawn range.")

        bin_bounds = self._get_bin_exclusion_bounds()
        xyzs = []
        max_attempts = 5000
        attempts = 0
        while len(xyzs) < n_obj and attempts < max_attempts:
            attempts += 1
            x = np.random.uniform(x_lo, x_hi)
            y = np.random.uniform(y_lo, y_hi)
            z = np.random.uniform(z_min, z_max)
            cand = np.array([x, y, z], dtype=np.float32)

            if not self._is_within_reach_xy(x, y):
                continue

            if bin_bounds is not None:
                bx0, bx1, by0, by1 = bin_bounds
                if (bx0 <= x <= bx1) and (by0 <= y <= by1):
                    continue

            if any(np.linalg.norm(cand - other) < min_dist for other in xyzs):
                continue

            xyzs.append(cand)

        if len(xyzs) != n_obj:
            raise ValueError(
                f"Could not sample {n_obj} objects within constrained spawn region "
                f"after {max_attempts} attempts."
            )
        return np.stack(xyzs, axis=0)

    def _sample_object_xyzs_with_retries(self, n_obj):
        """
        Sample object positions with progressively relaxed spacing constraints.
        """
        candidate_min_dists = []
        for value in [
            self.spawn_min_dist,
            self.spawn_fallback_min_dist,
            0.08,
            0.06,
        ]:
            value = float(value)
            if value < 0:
                continue
            if not any(np.isclose(value, existing) for existing in candidate_min_dists):
                candidate_min_dists.append(value)

        last_exc = None
        for idx, min_dist in enumerate(candidate_min_dists):
            try:
                if idx > 0:
                    print(f"[SimpleEnv.reset] retrying with min_dist={min_dist}")
                return self._sample_object_xyzs(n_obj=n_obj, min_dist=min_dist)
            except ValueError as exc:
                last_exc = exc
                print(f"[SimpleEnv.reset] object sampling failed: {exc}")

        raise last_exc

    def _teleop_debug_status(self):
        key_map = {
            'W': glfw.KEY_W,
            'A': glfw.KEY_A,
            'S': glfw.KEY_S,
            'D': glfw.KEY_D,
            'R': glfw.KEY_R,
            'F': glfw.KEY_F,
            'Q': glfw.KEY_Q,
            'E': glfw.KEY_E,
            'LEFT': glfw.KEY_LEFT,
            'RIGHT': glfw.KEY_RIGHT,
            'UP': glfw.KEY_UP,
            'DOWN': glfw.KEY_DOWN,
            'SPACE': glfw.KEY_SPACE,
            'Z': glfw.KEY_Z,
        }
        active_keys = [name for name, key in key_map.items() if self._key_is_down(key)]

        viewer = getattr(self.env, 'viewer', None)
        window = getattr(viewer, 'window', None)
        focused = False
        if window is not None:
            try:
                focused = bool(glfw.get_window_attrib(window, glfw.FOCUSED))
            except Exception:
                focused = False

        return focused, active_keys

    def _get_mouse_delta(self):
        """Poll middle mouse button and return (dx, dy, ctrl_held) cursor pixel deltas.
        Resets tracking when button is released so there is no position jump on re-engage."""
        viewer = getattr(self.env, 'viewer', None)
        window = getattr(viewer, 'window', None)
        if window is None:
            self._mouse_last_x = None
            self._mouse_last_y = None
            return 0.0, 0.0, False
        try:
            middle_held = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS
            xpos, ypos = glfw.get_cursor_pos(window)
            ctrl_held = (
                glfw.get_key(window, glfw.KEY_LEFT_CONTROL) == glfw.PRESS or
                glfw.get_key(window, glfw.KEY_RIGHT_CONTROL) == glfw.PRESS
            )
        except Exception:
            self._mouse_last_x = None
            self._mouse_last_y = None
            return 0.0, 0.0, False
        if not middle_held:
            self._mouse_last_x = None
            self._mouse_last_y = None
            return 0.0, 0.0, ctrl_held
        if self._mouse_last_x is None:
            # 第一帧保持 — 锚定位置，还没有增量
            self._mouse_last_x = xpos
            self._mouse_last_y = ypos
            return 0.0, 0.0, ctrl_held
        dx = xpos - self._mouse_last_x
        dy = ypos - self._mouse_last_y
        self._mouse_last_x = xpos
        self._mouse_last_y = ypos
        return dx, dy, ctrl_held

    def init_viewer(self):
        '''
        Initialize the viewer
        '''
        self.env.reset()
        try:
            p_base = self.env.get_p_body('base')
            p_tcp = self.env.get_p_body(self.tcp_body_name)
            # 启动靠近前工作区的自由相机，而不是
            # 默认远缩小视图。
            lookat = 0.5 * (p_base + p_tcp)
        except Exception:
            lookat = np.array([0.25, 0.0, 0.95], dtype=np.float32)
        self.env.init_viewer(
            distance=0.9,
            azimuth=180,
            elevation=-20,
            lookat=lookat,
            transparent=False,
            black_sky=True,
            use_rgb_overlay=False,
            loc_rgb_overlay='top right',
        )
        # 将 pert.select 预先分配给动作捕捉目标主体，以便
        # Ctrl + 右键拖动可立即移动它，无需单击选择。
        self._init_mocap_pert()

    def _init_mocap_pert(self):
        '''Find the 'target' mocap body and assign it as the permanent pert selection.'''
        try:
            body_id = mujoco.mj_name2id(
                self.env.model, mujoco.mjtObj.mjOBJ_BODY, 'target')
            self._mocap_body_id = body_id
            self._mocap_id = (
                int(self.env.model.body_mocapid[body_id])
                if body_id >= 0 else -1
            )
            if body_id >= 0 and self._mocap_id >= 0:
                viewer = getattr(self.env, 'viewer', None)
                if viewer is not None and hasattr(viewer, 'pert'):
                    # 只设置select——mjv_initPerturb被_mouse_button_callback调用
                    # 当用户实际开始 Ctrl+拖动时，xpos 才有效。
                    viewer.pert.select = body_id
                print(f'[mocap] target body_id={body_id} mocap_id={self._mocap_id} — Ctrl+right-drag to move arm')
            else:
                print('[mocap] no target body found — mocap drag disabled')
        except Exception as e:
            print(f'[mocap init] {e}')
            self._mocap_body_id = -1
            self._mocap_id = -1
    def reset(self, seed = None):
        '''
        Reset the environment
        Move the robot to a initial position, set the object positions based on the seed
        '''
        if seed is not None:
            np.random.seed(seed=seed)
        
        # 设置机器人特定的初始配置和起始位置
        if self.robot_profile == 'so101':
            # SO101 stowed: joint 1 = 0, joint 2 = clockwise limit, joint 3 = anti-clockwise limit,
            # wrist = 90°. Uses model limits for j2/j3 so stowed stays correct if model changes.
            q_zero = np.array(
                [
                    0.0,
                    self.joint_mins[1],   # Shoulder_lift 处于顺时针极限
                    self.joint_maxs[2],   # 逆时针极限处的肘部弯曲
                    np.pi / 2,            # 手腕弯曲 90°
                    0.0,                  # 手腕滚动
                ],
                dtype=np.float32,
            )
            self.env.forward(q=q_zero, joint_names=self.joint_names, increase_tick=False)
        else:
            # Use IK for Cartesian / eef_pose controlled robots
            if self.robot_profile == 'so100':
                q_init = np.zeros(self.n_arm_joints, dtype=np.float32)
                p_trgt = np.array([0.25, -0.35, 0.95])
                R_trgt = rpy2r(np.deg2rad([90, 0, 90]))
            else:  # 奥米
                q_init = np.zeros(self.n_arm_joints, dtype=np.float32)
                p_trgt = np.array([0.3, 0.0, 1.0])
                R_trgt = rpy2r(np.deg2rad([90, 0, 90]))

            q_zero,ik_err_stack,ik_info = solve_ik(
                env = self.env,
                joint_names_for_ik = self.joint_names,
                body_name_trgt     = self.tcp_body_name,
                q_init       = q_init,
                p_trgt       = p_trgt,
                R_trgt       = R_trgt,
            )
            self.env.forward(q=q_zero,joint_names=self.joint_names,increase_tick=False)

        # 将动作捕捉目标球体同步到当前 TCP 位置和方向
        try:
            if self._mocap_id >= 0:
                mujoco.mj_forward(self.env.model, self.env.data)
                p_tcp, R_tcp = self.env.get_pR_body(body_name=self.tcp_body_name)
                self.env.data.mocap_pos[self._mocap_id] = p_tcp
                # Convert rotation matrix to quaternion for mocap_quat (MuJoCo: [w,x,y,z])
                q_tcp = r2quat(R_tcp)
                self.env.data.mocap_quat[self._mocap_id] = q_tcp
                self._last_mocap_pos = p_tcp.copy()
                self._last_mocap_quat = q_tcp.copy()
        except Exception:
            pass

        # Legacy block: also sync via body_id lookup if _init_mocap_pert not yet called
        try:
            target_body_id = mujoco.mj_name2id(
                self.env.model, mujoco.mjtObj.mjOBJ_BODY, 'target')
            if target_body_id >= 0:
                mocap_id = self.env.model.body_mocapid[target_body_id]
                if mocap_id >= 0 and self._mocap_id < 0:
                    self.env.forward(increase_tick=False)
                    p_tcp, _ = self.env.get_pR_body(body_name=self.tcp_body_name)
                    self.env.data.mocap_pos[mocap_id] = p_tcp
        except Exception:
            pass

        # 随机拾取对象位置。  板/靶体固定在
        # 场景 XML（无自由接头），这里从未涉及过。
        all_obj_names = self.env.get_body_names(prefix='body_obj_')
        obj_names = [n for n in all_obj_names if n != self.place_body_name]
        n_obj = len(obj_names)
        obj_xyzs = self._sample_object_xyzs_with_retries(n_obj=n_obj)

        # 记录并应用所有可移动块的初始生成配置。
        self.spawn_obj_names = list(obj_names)
        self.spawn_obj_xyzs = obj_xyzs.copy()
        for obj_idx in range(n_obj):
            self.env.set_p_base_body(body_name=obj_names[obj_idx], p=obj_xyzs[obj_idx, :])
            self.env.set_R_base_body(body_name=obj_names[obj_idx], R=np.eye(3, 3))

        # 清除残余动态，使新生成的物体开始静止
        self.env.data.qvel[:] = 0.0
        self.env.data.qacc[:] = 0.0
        if self.env.data.ctrl is not None and self.env.data.ctrl.size > 0:
            self.env.data.ctrl[:] = 0.0
            # 重置时将夹具初始化为关闭状态，以便编写脚本
            # 明显地在下降阶段开始打开。
            self.env.data.ctrl[-self.n_gripper_actuators:] = self._make_gripper_ctrl(1.0)
        self.env.forward(increase_tick=False)

        # 在开始远程操作之前让物体在重力作用下沉降
        settle_ctrl = np.zeros(self.env.n_ctrl, dtype=np.float32)
        if self.env.data.ctrl is not None and self.env.data.ctrl.size > 0:
            self.env.data.ctrl[:] = 0.0
            # 当物体在重力作用下沉降时，保持夹具关闭。
            settle_ctrl[-self.n_gripper_actuators:] = self._make_gripper_ctrl(1.0)
            self.env.data.ctrl[:] = settle_ctrl
        for _ in range(20):
            self.env.step(settle_ctrl)

        # 设置机器人的初始位姿
        self.last_q = copy.deepcopy(q_zero)
        self.prev_q = copy.deepcopy(q_zero)
        # 将 close-gripper 命令附加到关节配置中。
        self.q = np.concatenate([q_zero, self._make_gripper_ctrl(1.0)])
        self.p0, self.R0 = self.env.get_pR_body(body_name=self.tcp_body_name)
        mug_init_pose, plate_init_pose = self.get_obj_pose()
        self.obj_init_pose = np.concatenate([mug_init_pose, plate_init_pose],dtype=np.float32)
        for _ in range(100):
            self.step_env()
        print("DONE INITIALIZATION")
        # gripper_state is a float in [0, 1]: 0 = fully open, 1 = fully closed.
        # 在夹具关闭的情况下开始剧集，以便第一个脚本/teleop
        # 操作不会立即打开它。
        self.gripper_state = 1.0
        self._target_was_lifted = False
        # Record post-settling baseline height for the target object.  Lift
        # 检测是相对于该基线（不是绝对 z）进行测量的。
        self._target_rest_height = float(self.env.get_p_body(self.pick_body_name)[2])
        self.past_chars = []

    def step(self, action):
        '''
        Take a step in the environment
        args:
            action: np.array of shape (7,), action to take
        returns:
            state: np.array, state of the environment after taking the action
                - ee_pose: [px,py,pz,r,p,y]
                - joint_angle: [j1,j2,j3,j4,j5,j6]

        '''
        action = np.asarray(action, dtype=np.float32)
        expected_dim = 7 if self.action_type == 'eef_pose' else self.n_arm_joints + 1
        if action.shape != (expected_dim,) or not np.all(np.isfinite(action)):
            raise ValueError(f"Expected finite action of shape ({expected_dim},), got {action.shape}")
        prev_q = copy.deepcopy(self.last_q)
        if self.action_type == 'eef_pose':
            q = self.env.get_qpos_joints(joint_names=self.joint_names)
            self.p0 += action[:3]
            self.R0 = self.R0.dot(rpy2r(action[3:6]))
            
            # 根据机器人轮廓调整 IK 参数
            if self.robot_profile in ('so100', 'so101'):
                ik_stepsize = 0.3
                ik_eps = 0.5
                max_ik_tick = 50
            else:  # 奥米
                ik_stepsize = 1.0
                ik_eps = 1e-2
                max_ik_tick = 50
            
            q ,ik_err_stack,ik_info = solve_ik(
                env                = self.env,
                joint_names_for_ik = self.joint_names,
                body_name_trgt     = self.tcp_body_name,
                q_init             = q,
                p_trgt             = self.p0,
                R_trgt             = self.R0,
                max_ik_tick        = max_ik_tick,
                ik_stepsize        = ik_stepsize,
                ik_eps             = ik_eps,
                ik_th              = np.radians(5.0),
                render             = False,
                verbose_warning    = False,
                restore_state      = False,
            )
        elif self.action_type == 'delta_joint_angle':
            q = action[:self.n_arm_joints] + prev_q
        elif self.action_type == 'joint_angle':
            q = action[:self.n_arm_joints]
        else:
            raise ValueError('action_type not recognized')

        q = np.clip(q, self.joint_mins, self.joint_maxs)
        self.prev_q = prev_q
        self.last_q = copy.deepcopy(q)
        
        # Map the task-space convention 0=open, 1=closed to the underlying
        # 执行器命令。 SO100/SO101 使用相反的原始接头方向：
        # 更大的角度使下巴张得更宽。
        gripper_cmd = self._make_gripper_ctrl(action[-1])
        self.compute_q = q
        self._commanded_gripper_fraction = float(np.clip(action[-1], 0.0, 1.0))
        q = np.concatenate([q, gripper_cmd])

        self.q = q
        if self.state_type == 'joint_angle':
            return self.get_joint_state()
        elif self.state_type == 'ee_pose':
            return self.get_ee_pose()
        elif self.state_type == 'delta_q' or self.action_type == 'delta_joint_angle':
            dq =  self.get_delta_q()
            return dq
        else:
            raise ValueError('state_type not recognized')

    def get_action_target(self):
        """返回最近一次 step 的监督目标：绝对关节弧度 + 夹爪闭合比例。

        不读取尚未执行控制命令的关节反馈；必须在 step(action) 后调用。
        """
        return np.concatenate([
            self.compute_q,
            [self._commanded_gripper_fraction],
        ]).astype(np.float32)

    def step_env(self):
        # 确保扰动目标保持锁定到动作捕捉，防止意外
        # 选择可能导致自由接头损坏的块或夹具
        if self._mocap_body_id >= 0:
            viewer = getattr(self.env, 'viewer', None)
            if viewer is not None and hasattr(viewer, 'pert'):
                if viewer.pert.select != self._mocap_body_id:
                    viewer.pert.select = self._mocap_body_id
        
        self.env.step(self.q)

    def grab_image(self):
        '''
        grab images from the environment
        returns:
            rgb_top:   np.array, rgb image from the topview camera
            rgb_front: np.array, rgb image from the frontview camera
        '''
        # 主流：自顶向下视图
        self.rgb_agent = self.env.get_fixed_cam_rgb(cam_name='topview')
        # 辅助流：前视摄像头（使用 MuJoCo“侧视”手柄）
        self.rgb_ego = self.env.get_fixed_cam_rgb(cam_name='sideview')
        # Keep rgb_side in sync with the secondary stream for overlays in teleop.
        self.rgb_side = self.rgb_ego.copy()
        return self.rgb_agent, self.rgb_ego
        

    def render(self, teleop=False):
        '''
        Render the environment
        '''
        self.env.plot_time()
        p_current, R_current = self.env.get_pR_body(body_name=self.tcp_body_name)
        R_current = R_current @ np.array([[1,0,0],[0,0,1],[0,1,0 ]])
        self.env.plot_sphere(p=p_current, r=0.02, rgba=[0.95,0.05,0.05,0.5])
        self.env.plot_capsule(p=p_current, R=R_current, r=0.01, h=0.2, rgba=[0.05,0.95,0.05,0.5])
        if self.show_gripper_pad_debug:
            self._plot_gripper_contact_pads()

        # 俯视相机始终显示在右上角。
        rgb_top_view = add_title_to_img(self.rgb_agent, text='Top View', shape=(640,480))
        self.env.viewer_rgb_overlay(rgb_top_view, loc='top right')

        # 前视摄像头：
        # - 遥操作（数据采集）时，显示在左上角。
        # - 在数据集回放期间，将其显示在右下角。
        if self.rgb_ego is not None:
            if teleop:
                rgb_front_view = add_title_to_img(self.rgb_ego, text='Front View', shape=(640,480))
                self.env.viewer_rgb_overlay(rgb_front_view, loc='top left')
            else:
                rgb_front_view = add_title_to_img(self.rgb_ego, text='Front View', shape=(640,480))
                self.env.viewer_rgb_overlay(rgb_front_view, loc='bottom right')
            self.env.viewer_text_overlay(text1='Key Pressed',text2='%s'%(self.env.get_key_pressed_list(as_text=True)))
            self.env.viewer_text_overlay(text1='Key Repeated',text2='%s'%(self.env.get_key_repeated_list(as_text=True)))
            focused, active_keys = self._teleop_debug_status()
            self.env.viewer_text_overlay(text1='Window Focus', text2='YES' if focused else 'NO (click viewer)')
            self.env.viewer_text_overlay(text1='Active Keys (raw)', text2='%s' % active_keys)
        self.env.render()

    def _plot_gripper_contact_pads(self):
        """
        Draw translucent boxes over the gripper collision pads so they are easy
        to inspect in the viewer.
        """
        for geom_name, rgba in self._gripper_pad_debug_geoms:
            geom_id = mujoco.mj_name2id(
                self.env.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name
            )
            if geom_id < 0:
                continue
            p = self.env.data.geom_xpos[geom_id].copy()
            R = self.env.data.geom_xmat[geom_id].reshape(3, 3).copy()
            size = self.env.model.geom_size[geom_id].copy()
            self.env.plot_box(
                p=p,
                R=R,
                xlen=2.0 * size[0],
                ylen=2.0 * size[1],
                zlen=2.0 * size[2],
                rgba=rgba,
                label=geom_name,
            )

    def get_joint_state(self):
        '''
        Get the joint state of the robot
        returns:
            q: np.array, joint angles of the robot + normalised gripper position
            [j1,j2,j3,j4,j5,j6,gripper]  where gripper ∈ [0,1] (0=open, 1=closed)
        '''
        qpos = self.env.get_qpos_joints(joint_names=self.joint_names)
        gripper_val = self._get_gripper_fraction()
        return np.concatenate([qpos, [gripper_val]], dtype=np.float32)
    
    def teleop_robot(self):
        '''
        Teleoperate the robot using keyboard or middle mouse button.
        returns:
            action: np.array, action to take
            done: bool, True if the user wants to reset the teleoperation

        Mouse (middle button drag):
            Drag left/right : +/- Y  (world frame)
            Drag up/down    : +/- Z  (world frame)
            Ctrl + drag L/R : +/- X  (world frame)

        Keyboard:
            W/S : +/- X (Forward/Backward)
            A/D : +/- Y (Left/Right)
            R/F : +/- Z (Up/Down)
            Q/E : Yaw left/right
            UP/DOWN     : Pitch up/down
            LEFT/RIGHT  : Roll left/right
            SPACE : gripper open/close
            Z     : reset
        '''
        # 关节空间控制（SO101默认）
        if self.action_type == 'delta_joint_angle':
            return self._teleop_joint_space()

        # Mocap / IK 笛卡尔控制 — 拖动红色目标球体
        if self.action_type == 'eef_pose':
            return self._teleop_mocap()

        # 后备：仅键盘笛卡尔控制
        dpos = np.zeros(3)
        drot = np.eye(3)

        # 鼠标点动：中键拖动
        # 无Ctrl：dx -> Y，-dy -> Z
        # 使用 Ctrl：dx -> X，-dy -> Z
        _mouse_pos_scale = 0.00015  # 每像素米
        _mdx, _mdy, _mctrl = self._get_mouse_delta()
        if abs(_mdx) > 0.2 or abs(_mdy) > 0.2:
            if _mctrl:
                dpos[0] += _mdx * _mouse_pos_scale
            else:
                dpos[1] += _mdx * _mouse_pos_scale
            dpos[2] -= _mdy * _mouse_pos_scale

        # 键盘：WASD/RF/QE/箭头
        if self._key_is_down(glfw.KEY_W):
            dpos += np.array([0.007,0.0,0.0])  # 前进 (+X)
        if self._key_is_down(glfw.KEY_S):
            dpos += np.array([-0.007,0.0,0.0])  # 向后 (-X)
        if self._key_is_down(glfw.KEY_A):
            dpos += np.array([0.0,-0.007,0.0])  # 左 (-Y)
        if self._key_is_down(glfw.KEY_D):
            dpos += np.array([0.0,0.007,0.0])  # 右 (+Y)
        if self._key_is_down(glfw.KEY_R):
            dpos += np.array([0.0,0.0,0.007])  # Up (+Z)
        if self._key_is_down(glfw.KEY_F):
            dpos += np.array([0.0,0.0,-0.007])  # 向下 (-Z)
        if self._key_is_down(glfw.KEY_Q):
            drot = rotation_matrix(angle=0.1 * 0.3, direction=[0.0, 0.0, 1.0])[:3, :3]  # 左偏航
        if self._key_is_down(glfw.KEY_E):
            drot = rotation_matrix(angle=-0.1 * 0.3, direction=[0.0, 0.0, 1.0])[:3, :3]  # 右偏航
        if  self._key_is_down(glfw.KEY_LEFT):
            drot = rotation_matrix(angle=0.1 * 0.3, direction=[0.0, 1.0, 0.0])[:3, :3]  # 向左滚动
        if  self._key_is_down(glfw.KEY_RIGHT):
            drot = rotation_matrix(angle=-0.1 * 0.3, direction=[0.0, 1.0, 0.0])[:3, :3]  # 向右滚动
        if self._key_is_down(glfw.KEY_UP):
            drot = rotation_matrix(angle=0.1 * 0.3, direction=[1.0, 0.0, 0.0])[:3, :3]  # 向上倾斜
        if self._key_is_down(glfw.KEY_DOWN):
            drot = rotation_matrix(angle=-0.1 * 0.3, direction=[1.0, 0.0, 0.0])[:3, :3]  # 俯仰
        if self._key_is_pressed_once(glfw.KEY_Z):
            return np.zeros(7, dtype=np.float32), True
        if self._key_is_pressed_once(glfw.KEY_SPACE):
            self.gripper_state = 0.0 if self.gripper_state > 0.5 else 1.0  # 切换打开/关闭
        if self._key_is_down(glfw.KEY_RIGHT_BRACKET):   # ] → 逐渐关闭
            self.gripper_state = min(1.0, self.gripper_state + 0.05)
        if self._key_is_down(glfw.KEY_LEFT_BRACKET):    # [ → 逐渐打开
            self.gripper_state = max(0.0, self.gripper_state - 0.05)
        drot = r2rpy(drot)
        action = np.concatenate([dpos, drot, np.array([self.gripper_state],dtype=np.float32)],dtype=np.float32)
        self._teleop_debug_counter += 1
        if np.linalg.norm(action[:6]) > 1e-8 or self._teleop_debug_counter % 200 == 0:
            focused, active_keys = self._teleop_debug_status()
            print(f"[teleop-debug] focused={focused} keys={active_keys} action={np.round(action, 4)}")
        return action, False
    
    def _teleop_mocap(self):
        '''
        Mocap / IK Cartesian teleoperation.

        In the MuJoCo viewer: double-click the red sphere to select it, then
        drag to reposition the arm.  MuJoCo moves the mocap body natively;
        this method reads its position each step and feeds it to the IK solver
        via the eef_pose action (delta_p = mocap_pos - current_IK_target).

        Keyboard rotation still works:
            Q/E        : yaw left / right
            LEFT/RIGHT : roll
            UP/DOWN    : pitch
        SPACE: gripper open/close
        Z    : reset episode
        '''
        delta_p = np.zeros(3, dtype=np.float32)

        try:
            target_body_id = mujoco.mj_name2id(
                self.env.model, mujoco.mjtObj.mjOBJ_BODY, 'target')
            if target_body_id >= 0:
                mocap_id = self.env.model.body_mocapid[target_body_id]
                if mocap_id >= 0:
                    # 预先指定 pert.select 每一步，以便始终按住 Ctrl 键并拖动
                    # 控制该主体 - 无需双击选择。
                    viewer = getattr(self.env, 'viewer', None)
                    if viewer is not None and hasattr(viewer, 'pert'):
                        if viewer.pert.select != target_body_id:
                            viewer.pert.select = target_body_id
                            mujoco.mjv_initPerturb(
                                self.env.model, self.env.data,
                                viewer.scn, viewer.pert)

                    mocap_pos = self.env.data.mocap_pos[mocap_id].copy()
                    delta_p = (mocap_pos - self.p0).astype(np.float32)
                    # 夹紧以避免第一帧出现巨大的跳跃
                    delta_p = np.clip(delta_p, -0.02, 0.02)
        except Exception as e:
            print(f'[mocap] {e}')

        # 键盘旋转
        drot = np.eye(3)
        if self._key_is_down(glfw.KEY_Q):
            drot = rotation_matrix(angle=0.03, direction=[0, 0, 1])[:3, :3]
        if self._key_is_down(glfw.KEY_E):
            drot = rotation_matrix(angle=-0.03, direction=[0, 0, 1])[:3, :3]
        if self._key_is_down(glfw.KEY_LEFT):
            drot = rotation_matrix(angle=0.03, direction=[0, 1, 0])[:3, :3]
        if self._key_is_down(glfw.KEY_RIGHT):
            drot = rotation_matrix(angle=-0.03, direction=[0, 1, 0])[:3, :3]
        if self._key_is_down(glfw.KEY_UP):
            drot = rotation_matrix(angle=0.03, direction=[1, 0, 0])[:3, :3]
        if self._key_is_down(glfw.KEY_DOWN):
            drot = rotation_matrix(angle=-0.03, direction=[1, 0, 0])[:3, :3]

        if self._key_is_pressed_once(glfw.KEY_SPACE):
            self.gripper_state = 0.0 if self.gripper_state > 0.5 else 1.0  # 切换打开/关闭
        if self._key_is_down(glfw.KEY_RIGHT_BRACKET):   # ] → 逐渐关闭
            self.gripper_state = min(1.0, self.gripper_state + 0.05)
        if self._key_is_down(glfw.KEY_LEFT_BRACKET):    # [ → 逐渐打开
            self.gripper_state = max(0.0, self.gripper_state - 0.05)
        if self._key_is_pressed_once(glfw.KEY_Z):
            return np.zeros(7, dtype=np.float32), True

        drpy = r2rpy(drot).astype(np.float32)
        action = np.concatenate([delta_p, drpy,
                                  np.array([self.gripper_state], dtype=np.float32)])
        return action, False

    def _teleop_joint_space(self):
        '''
        Joint-space teleoperation for SO101.

        Mouse (middle button drag):
            Drag left/right     : shoulder_pan
            Drag up/down        : shoulder_lift  (drag up = lift up)
            Ctrl + drag L/R     : elbow_flex
            Ctrl + drag up/down : wrist_flex

        Keyboard (unchanged):
            A/D        : shoulder_pan
            W/S        : shoulder_lift
            R/F        : elbow_flex
            UP/DOWN    : wrist_flex
            LEFT/RIGHT : wrist_roll
            SPACE      : gripper open/close
            Z          : reset
        '''
        dq = np.zeros(self.n_arm_joints, dtype=np.float32)
        joint_vel = 0.03  # 每次按下键盘的关节速度大小

        # Mocap IK 跟踪：按住 Ctrl 键并右键拖动红色球体来移动手臂。
        # pert.select 是在 init_viewer 中预先分配的，因此拖动可以立即生效。
        if self._mocap_id >= 0:
            mocap_pos = self.env.data.mocap_pos[self._mocap_id].copy()
            mocap_quat = self.env.data.mocap_quat[self._mocap_id].copy()
            if self._last_mocap_pos is None:
                self._last_mocap_pos = mocap_pos.copy()
                self._last_mocap_quat = mocap_quat.copy()
            else:
                pos_changed = np.linalg.norm(mocap_pos - self._last_mocap_pos) > 5e-4
                quat_changed = np.linalg.norm(mocap_quat - self._last_mocap_quat) > 5e-4
                if pos_changed or quat_changed:
                    q_current = self.env.get_qpos_joints(joint_names=self.joint_names).copy()
                    R_trgt = quat2r(mocap_quat) if quat_changed else None
                    q_ik, _, _ = solve_ik(
                        env                = self.env,
                        joint_names_for_ik = self.joint_names,
                        body_name_trgt     = self.tcp_body_name,
                        q_init             = q_current,
                        p_trgt             = mocap_pos,
                        R_trgt             = R_trgt,
                        max_ik_tick        = 30,
                        ik_stepsize        = 0.5,
                        ik_eps             = 0.01,
                        ik_err_th          = 0.01,
                        verbose_warning    = False,
                        restore_state      = False,
                    )
                    dq += np.clip(q_ik - q_current, -joint_vel * 4, joint_vel * 4)
            self._last_mocap_pos = mocap_pos.copy()
            self._last_mocap_quat = mocap_quat.copy()

        # 鼠标中键拖动：精细的每个关节点动（不变）
        _mouse_jnt_scale = 0.0005
        _mdx, _mdy, _mctrl = self._get_mouse_delta()
        if abs(_mdx) > 0.2 or abs(_mdy) > 0.2:
            if _mctrl:
                dq[2] += _mdx * _mouse_jnt_scale
                dq[3] -= _mdy * _mouse_jnt_scale
            else:
                dq[0] += _mdx * _mouse_jnt_scale
                dq[1] -= _mdy * _mouse_jnt_scale

        # W/S：肩部提升
        if self._key_is_down(glfw.KEY_W):
            dq[1] += joint_vel  # 肩膀抬起
        if self._key_is_down(glfw.KEY_S):
            dq[1] -= joint_vel  # 肩膀向下抬起

        # A/D：左-右（肩盘）
        if self._key_is_down(glfw.KEY_A):
            dq[0] -= joint_vel  # 肩部向左平移
        if self._key_is_down(glfw.KEY_D):
            dq[0] += joint_vel  # 右肩平移

        # UP/DOWN：上下（手腕弯曲）
        if self._key_is_down(glfw.KEY_UP):
            dq[3] += joint_vel  # 手腕向上弯曲
        if self._key_is_down(glfw.KEY_DOWN):
            dq[3] -= joint_vel  # 手腕向下弯曲
        
        # RF：控制弯头
        if self._key_is_down(glfw.KEY_R):
            dq[2] += joint_vel  # 肘部向上弯曲
        if self._key_is_down(glfw.KEY_F):
            dq[2] -= joint_vel  # 肘部向下弯曲
        
        # 左/右：控制手腕滚动
        if self._key_is_down(glfw.KEY_LEFT):
            dq[4] += joint_vel  # 手腕滚动
        if self._key_is_down(glfw.KEY_RIGHT):
            dq[4] -= joint_vel  # 手腕滚动
        
        # UP/DOWN reserved for future use or finer control
        
        # 复位和夹具
        if self._key_is_pressed_once(glfw.KEY_Z):
            return np.zeros(self.n_arm_joints + 1, dtype=np.float32), True
        if self._key_is_pressed_once(glfw.KEY_SPACE):
            self.gripper_state = 0.0 if self.gripper_state > 0.5 else 1.0  # 切换打开/关闭
        if self._key_is_down(glfw.KEY_RIGHT_BRACKET):   # ] → 逐渐关闭
            self.gripper_state = min(1.0, self.gripper_state + 0.05)
        if self._key_is_down(glfw.KEY_LEFT_BRACKET):    # [ → 逐渐打开
            self.gripper_state = max(0.0, self.gripper_state - 0.05)

        # 将关节​​增量与夹具状态相结合
        action = np.concatenate([dq, np.array([self.gripper_state], dtype=np.float32)])
        self._teleop_debug_counter += 1
        if np.linalg.norm(dq) > 1e-8 or self._teleop_debug_counter % 200 == 0:
            focused, active_keys = self._teleop_debug_status()
            print(f"[teleop-debug] focused={focused} keys={active_keys} dq={np.round(dq, 4)} gripper={self.gripper_state:.2f}")
        return action, False
    
    def get_delta_q(self):
        '''
        Get the delta joint angles of the robot
        returns:
            delta: np.array, delta joint angles of the robot + normalised gripper position
            [dj1,dj2,dj3,dj4,dj5,dj6,gripper]  where gripper ∈ [0,1] (0=open, 1=closed)
        '''
        delta = self.compute_q - self.prev_q
        gripper_val = self._get_gripper_fraction()
        return np.concatenate([delta, [gripper_val]], dtype=np.float32)

    def _get_gripper_fraction(self):
        """
        Return the normalized task-space gripper command where 0=open, 1=closed.
        """
        gripper = self.env.get_qpos_joint(self.gripper_joint_name)
        raw_qpos = float(gripper[0])
        if self.gripper_continuous:
            return float(
                np.clip(
                    (self.gripper_open_rad - raw_qpos) / self.gripper_motion_span,
                    0.0,
                    1.0,
                )
            )
        return 1.0 if raw_qpos > 0.5 else 0.0

    def _is_gripper_open(self, threshold=0.2):
        """
        Return True when the normalized close fraction is below the open threshold.
        """
        return self._get_gripper_fraction() < threshold

    def check_success(self):
        '''
        Check if the object has been placed in/on the target.

        SO100/SO101 (bin task):
            The episode ends only when the block is truly inside the bin AND the
            gripper has been retracted above the bin opening.
            Conditions:
              1. Block center is within bin inner XY bounds (not just near bin).
              2. Block center Z is inside bin cavity bounds (resting on bin floor).
              3. Block was previously lifted off the table (prevents pushing/rolling in).
              4. Block is in contact with the bin (confirms physical placement).
              5. Gripper is open (not holding the block).
              6. TCP is above the bin wall tops (gripper retracted after release).
              7. Gripper is not in contact with the block.

        OMY (plate task):
            1. Object XY within 10 cm of plate centre.
            2. Object Z within 60 cm of plate Z.
            3. Gripper open (raw joint < 0.1 rad).
            4. TCP Z > success_height_threshold (arm retracted upward).
        '''
        p_obj = self.env.get_p_body(self.pick_body_name)
        p_tgt = self.env.get_p_body(self.place_body_name)

        if self.success_height_threshold is None:
            # obj_bin.xml中的 SO100/SO101 箱几何形状：
            # - inner half-width ~= 0.054 - 0.006 = 0.048 m
            # - floor top offset   = 0.012 m
            # - wall top offset    = 0.076 m
            # 块是obj_blocks.xml中的 25 mm 立方体（半尺寸 0.0125 m）。
            inner_half_extent = 0.048
            block_half_size = 0.0125
            floor_top = p_tgt[2] + 0.012
            wall_top = p_tgt[2] + 0.076

            dx = abs(p_obj[0] - p_tgt[0])
            dy = abs(p_obj[1] - p_tgt[1])
            xy_ok = (dx < (inner_half_extent - 0.003)) and (dy < (inner_half_extent - 0.003))

            # 块必须放置在垃圾箱地板上。  放置一个 25 毫米的立方体
            # 地板的中心高度为floor_top + block_half_size。
            expected_floor_contact_center_z = floor_top + block_half_size
            z_min = expected_floor_contact_center_z - 0.006
            z_max = expected_floor_contact_center_z + 0.010
            z_ok = (p_obj[2] > z_min) and (p_obj[2] < z_max)

            # Task intent is pick-and-place; pushing/rolling directly into the
            # bin 不应终止该剧集。  上方需要有无障碍升降机
            # 首先设定重置高度。
            rest_h = self._target_rest_height if self._target_rest_height is not None else p_obj[2]
            if p_obj[2] > (rest_h + self._lift_height_delta):
                self._target_was_lifted = True
            if not self._target_was_lifted:
                return False

            # 接触门控：成功需要实际的块箱接触，并且不需要
            # 主动块夹具接触。
            contact_pairs = self.env.get_contact_body_names()

            def _has_contact(body_a, body_b):
                for c1, c2 in contact_pairs:
                    if (c1 == body_a and c2 == body_b) or (c1 == body_b and c2 == body_a):
                        return True
                return False

            block_bin_contact = _has_contact(self.pick_body_name, self.place_body_name)
            block_gripper_contact = _has_contact(self.pick_body_name, 'gripper') or _has_contact(
                self.pick_body_name, 'moving_jaw_so101_v1'
            )

            gripper_open = self._is_gripper_open(threshold=0.20)

            # 夹爪尖端必须缩回到料仓壁顶部上方。
            # 重要提示：tcp_body_name（'gripper'）是手掌，距手掌上方约 98 毫米
            # 实际的下巴尖端。  使用手掌位置意味着检查甚至可以通过
            # 当钳口位于料仓腔内时，这是根本原因
            # 仅仅接触块时就会出现误报成功。
            # Use the 'gripperframe' site (jaw tip) for an accurate measurement.
            _site_id = mujoco.mj_name2id(
                self.env.model, mujoco.mjtObj.mjOBJ_SITE, 'gripperframe')
            if _site_id >= 0:
                tip_z = float(self.env.data.site_xpos[_site_id][2])
            else:
                # 后备：从手掌主体中减去标称下颌尖端偏移。
                tip_z = float(self.env.get_p_body(self.tcp_body_name)[2]) - 0.098
            tcp_above = tip_z > wall_top + 0.015  # 钳口尖端距仓壁顶部 ≥ 15 毫米

            success = bool(xy_ok and z_ok and gripper_open and block_bin_contact
                        and tcp_above and not block_gripper_contact)
            
            # 调试日志记录
            if success or (xy_ok and z_ok):  # 接近成功时记录
                print(f"[DEBUG check_success] xy_ok={xy_ok} z_ok={z_ok} lifted={self._target_was_lifted} "
                      f"gripper_open={gripper_open} block_bin={block_bin_contact} "
                      f"tcp_above={tcp_above} no_grip_contact={not block_gripper_contact} => {success}")
            
            return success
        else:
            # OMY：原始的印版放置逻辑。
            xy_ok        = np.linalg.norm(p_obj[:2] - p_tgt[:2]) < 0.1
            z_ok         = np.linalg.norm(p_obj[2]  - p_tgt[2])  < 0.6
            gripper_open = self._is_gripper_open(threshold=0.10)
            if not (xy_ok and z_ok and gripper_open):
                return False
            tcp_z = self.env.get_p_body(self.tcp_body_name)[2]
            return tcp_z > self.success_height_threshold
    
    def get_obj_pose(self):
        '''
        returns: 
            p_mug: np.array, position of the mug
            p_plate: np.array, position of the plate
        '''
        p_pick = self.env.get_p_body(self.pick_body_name)
        p_place = self.env.get_p_body(self.place_body_name)
        return p_pick, p_place
    
    def set_obj_pose(self, p_pick, p_place):
        '''
        Set the object poses
        args:
            p_mug: np.array, position of the mug
            p_plate: np.array, position of the plate
        '''
        self.env.set_p_base_body(body_name=self.pick_body_name,p=p_pick)
        self.env.set_R_base_body(body_name=self.pick_body_name,R=np.eye(3,3))
        # Only attempt to move the plate/target body if it actually has a free joint.
        # 在SO100/SO101 bin任务中，bin是一个没有自由接头的固定体，
        # 因此调用 set_p_base_body 会导致 qpos 形状不匹配。
        try:
            body = self.env.model.body(self.place_body_name)
            n_joint = body.jntnum
            if n_joint > 0:
                first_joint = self.env.model.joint(body.jntadr[0])
                if first_joint.type[0] == mujoco.mjtJoint.mjJNT_FREE:
                    self.env.set_p_base_body(body_name=self.place_body_name, p=p_place)
                    self.env.set_R_base_body(body_name=self.place_body_name, R=np.eye(3, 3))
        except Exception:
            # 如果出现任何问题（例如未找到尸体），请退回到保持板固定的状态。
            pass
        self.step_env()


    def get_ee_pose(self):
        '''
        get the end effector pose of the robot + gripper state
        '''
        p, R = self.env.get_pR_body(body_name=self.tcp_body_name)
        rpy = r2rpy(R)
        return np.concatenate([p, rpy],dtype=np.float32)