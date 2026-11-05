import sys
import subprocess
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from mujoco_env.y_env import SimpleEnv
import controllers.scripted_fsm_controller as expert

# 减少 FSM 调试输出
expert.FSM_DEBUG = False

OUTPUT = ROOT / "media/simulation_demo.mp4"
TEMP = ROOT / "media/.simulation_demo_temp.mp4"

FPS = 20
WIDTH = 640
HEIGHT = 480
SEEDS = [42, 7, 123]
CONTROL_HZ = 40
MAX_SIM_SECONDS = 50

if OUTPUT.exists():
    raise SystemExit(f"文件已经存在，为避免覆盖请先检查：{OUTPUT}")

# 复用原项目的 SO-101 环境
env = SimpleEnv(
    xml_path=str((ROOT / "asset/scene_so101_y.xml").resolve()),
    action_type="delta_joint_angle",
    state_type="joint_angle",
    robot_profile="so101",
    seed=SEEDS[0],
    pick_body_name="body_obj_block_2",
    place_body_name="body_obj_bin",
    spawn_x_range=(0.21, 0.22),
    spawn_y_range=(0.04, 0.16),
    spawn_z_range=(0.815, 0.815),
    spawn_min_dist=0.01,
)

# 设置可视化视角
env.env.viewer.cam.lookat[:] = [0.35, 0.0, 0.85]
env.env.viewer.cam.distance = 1.5
env.env.viewer.cam.azimuth = 180
env.env.viewer.cam.elevation = -30


def start_encoder():
    return subprocess.Popen(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "error",
            "-y",
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-s:v", f"{WIDTH}x{HEIGHT}",
            "-r", str(FPS),
            "-i", "-",
            "-an",
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-threads", "2",
            "-crf", "24",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(TEMP),
        ],
        stdin=subprocess.PIPE,
    )


def capture_frame():
    # 使用项目自带的固定相机
    rgb = env.env.get_fixed_cam_rgb("sideview")

    frame = cv2.resize(
        rgb,
        (WIDTH, HEIGHT),
        interpolation=cv2.INTER_AREA,
    )

    return np.ascontiguousarray(frame, dtype=np.uint8)


def get_ee_xyz(environment):
    site_id = environment.model.site("gripperframe").id
    return environment.data.site_xpos[site_id].copy().astype(np.float32)


def get_cube_xyz(environment):
    cube, _ = environment.get_obj_pose()
    return np.asarray(cube, dtype=np.float32)


try:
    for attempt, seed in enumerate(SEEDS, start=1):
        print(f"\n===== Attempt {attempt}, seed={seed} =====", flush=True)

        env.reset(seed=seed)

        fsm = expert.setup_so101_controller(
            env=env,
            ee_site_name="gripperframe",
        )

        encoder = start_encoder()
        success = False
        success_streak = 0
        frame_count = 0
        previous_phase = None
        last_frame = None

        try:
            timestep = float(env.model.opt.timestep)
            max_steps = int(MAX_SIM_SECONDS / timestep)

            for _ in range(max_steps):
                if not env.env.is_viewer_alive():
                    print("仿真窗口已关闭，停止录制")
                    break

                # 物理仿真前进一步
                env.step_env()

                # 专家控制频率：40 Hz
                if env.env.loop_every(HZ=CONTROL_HZ):
                    success_now = bool(env.check_success())

                    success_streak = (
                        success_streak + 1 if success_now else 0
                    )

                    if success_streak >= 6:
                        success = True
                        print("TASK_SUCCESS", flush=True)
                        break

                    action = expert.fsm_step(
                        env=env,
                        fsm=fsm,
                        get_ee_xyz_fn=get_ee_xyz,
                        cube_pos_fn=get_cube_xyz,
                    )

                    env.step(action)

                    phase = expert.PHASES[fsm["phase"]]["name"]

                    if phase != previous_phase:
                        print(f"FSM Phase: {phase}", flush=True)
                        previous_phase = phase

                # 视频采样频率：20 FPS
                if env.env.loop_every(HZ=FPS):
                    env.env.render()
                    last_frame = capture_frame()
                    encoder.stdin.write(last_frame.tobytes())
                    frame_count += 1

            # 成功后让最终画面保留一秒
            if success and last_frame is not None:
                for _ in range(FPS):
                    encoder.stdin.write(last_frame.tobytes())
                    frame_count += 1

        finally:
            if encoder.stdin:
                encoder.stdin.close()
            encoder.wait()

        if success and encoder.returncode == 0 and frame_count > 0:
            TEMP.replace(OUTPUT)

            print("\n========== DEMO SUCCESS ==========")
            print("视频路径:", OUTPUT)
            print("帧数:", frame_count)
            print("视频时长:", round(frame_count / FPS, 2), "秒")
            print("文件大小:", round(OUTPUT.stat().st_size / 1024 / 1024, 2), "MB")
            break

        print(f"Attempt {attempt} failed", flush=True)
        TEMP.unlink(missing_ok=True)

    else:
        print("\nDEMO_FAILED: 三次尝试均未通过成功判定")
        sys.exit(1)

finally:
    env.env.close_viewer()
