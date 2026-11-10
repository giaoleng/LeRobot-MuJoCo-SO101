#!/usr/bin/env python3
"""SO-101 MuJoCo 三策略统一评测入口（LeRobot 0.5.1）。

运行方式见 docs/POLICY_COMPARISON.md。无模型的 --dry-run 不会加载 LeRobot/MuJoCo。
仅使用本地 checkpoint、数据元信息与预处理器；不访问 Hub，不会连接真机。
环境动作契约：5个关节的绝对弧度 + 夹爪闭合比例 [0,1]。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval.act_rollout import JsonlRecorder, run_evaluation

POLICIES = ("act", "diffusion", "smolvla")
ACTION_SEMANTICS = "joint_radians_gripper_closed_0_1"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--policy", choices=POLICIES, required=True)
    p.add_argument("--checkpoint", type=Path, required=True, help="本地 LeRobot pretrained_model 目录")
    p.add_argument("--dataset-root", type=Path, required=True, help="本地 LeRobotDataset 根目录（含 meta/info.json）")
    p.add_argument("--processors", choices=("saved", "dataset"), default="saved",
                   help="saved: 使用本地 processor JSON；dataset: 从 meta/stats.json 重建")
    p.add_argument("--device", choices=("cuda", "cpu", "auto"), default="auto")
    p.add_argument("--top-feature", default="observation.image",
                   help="俯视图在训练数据中的特征键")
    p.add_argument("--wrist-feature", default="observation.wrist_image",
                   help="侧视图在训练数据中的特征键")
    p.add_argument("--action-semantics", choices=(ACTION_SEMANTICS,), default=None,
                   help="显式确认模型输出为绝对关节弧度+夹爪闭合比例；不支持隐式单位转换")
    p.add_argument("--task", default="Put blue block in the bin")
    p.add_argument("--xml-path", type=Path, default=ROOT / "asset/scene_so101_y.xml")
    p.add_argument("--pick-body-name", default="body_obj_block_2")
    p.add_argument("--place-body-name", default="body_obj_bin")
    p.add_argument("--spawn-x-min", type=float, default=0.21)
    p.add_argument("--spawn-x-max", type=float, default=0.22)
    p.add_argument("--spawn-y-min", type=float, default=0.04)
    p.add_argument("--spawn-y-max", type=float, default=0.16)
    p.add_argument("--spawn-z-min", type=float, default=0.815)
    p.add_argument("--spawn-z-max", type=float, default=0.815)
    p.add_argument("--spawn-min-dist", type=float, default=0.01)
    p.add_argument("--spawn-fallback-min-dist", type=float, default=0.1)
    p.add_argument("--spawn-xy-margin", type=float, default=0.0)
    p.add_argument("--num-trials", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--hz", type=int, default=20)
    p.add_argument("--timeout-s", type=float, default=60.0)
    p.add_argument("--timeout-clock", choices=("sim", "wall"), default="sim")
    p.add_argument("--split", choices=("id", "ood"), default="id")
    p.add_argument("--output", type=Path, help="新建 JSONL 路径；拒绝覆盖")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)
    if a.num_trials <= 0 or a.hz <= 0 or not math.isfinite(a.timeout_s) or a.timeout_s <= 0:
        p.error("num-trials、hz、timeout-s 必须为正")
    if not all(math.isfinite(v) for v in
               (a.spawn_x_min, a.spawn_x_max, a.spawn_y_min, a.spawn_y_max,
                a.spawn_z_min, a.spawn_z_max, a.spawn_min_dist,
                a.spawn_fallback_min_dist, a.spawn_xy_margin)):
        p.error("采样范围必须为有限数字")
    if a.spawn_x_min > a.spawn_x_max or a.spawn_y_min > a.spawn_y_max or a.spawn_z_min > a.spawn_z_max:
        p.error("spawn min 不得大于 max")
    if a.top_feature == a.wrist_feature:
        p.error("top-feature 与 wrist-feature 必须不同")
    if not a.dry_run and a.action_semantics != ACTION_SEMANTICS:
        p.error("实际评测必须显式传入 --action-semantics " + ACTION_SEMANTICS)
    for key in ("checkpoint", "dataset_root", "xml_path", "output"):
        value = getattr(a, key)
        if value is not None:
            setattr(a, key, (ROOT / value).expanduser().resolve())
    if a.output is None:
        a.output = (ROOT / "outputs/eval" / f"{a.policy}_{a.split}_seed{a.seed}.jsonl").resolve()
    return a


def check_schema(policy_cfg, dataset_info, *, policy_name, top_feature, wrist_feature):
    """明确校验模型/数据/动作契约，不猜测维度、相机或控制单位。"""
    if policy_cfg.get("type") != policy_name:
        raise ValueError("checkpoint policy.type 与 --policy 不一致")
    features = dataset_info.get("features")
    if not isinstance(features, dict):
        raise ValueError("LeRobot 元数据缺少 features")
    required = {"observation.state", top_feature, wrist_feature, "action"}
    if required - features.keys():
        raise ValueError(f"数据集缺失字段: {sorted(required - features.keys())}")
    state = features["observation.state"]
    action = features["action"]
    if state.get("shape") != [6] or len(state.get("names", [])) != 6:
        raise ValueError("仿真 observation.state 必须是六维末端 xyz+rpy")
    if len(set(state["names"])) != 6:
        raise ValueError("state.names 必须是六个不同名称，且顺序对应末端 xyz+rpy")
    if action.get("shape") != [6]:
        raise ValueError("仿真 action 必须是六维（关节弧度×5 + 夹爪闭合比例）")
    if policy_cfg.get("output_features", {}).get("action", {}).get("shape") != [6]:
        raise ValueError("checkpoint output_features.action 与六维仿真动作不一致")
    model_inputs = policy_cfg.get("input_features", {})
    expected_inputs = {"observation.state", top_feature, wrist_feature}
    if set(model_inputs) != expected_inputs:
        raise ValueError(f"不支持的 checkpoint 输入特征: {sorted(model_inputs)}；期望: {sorted(expected_inputs)}。"
                         "任务类别条件等额外输入必须使用配套适配器，禁止自动忽略")
    if model_inputs["observation.state"].get("shape") != [6]:
        raise ValueError("checkpoint 状态输入不是六维")
    for key in (top_feature, wrist_feature):
        f = features[key]
        if f.get("dtype") not in ("image", "video") or len(f.get("shape", [])) != 3 or f["shape"][-1] != 3:
            raise ValueError(f"图像特征 {key} 必须是 H×W×3")
        if model_inputs[key].get("type") != "VISUAL":
            raise ValueError(f"checkpoint 中 {key} 不是视觉特征")
    # 如果数据记录了动作单位，它必须与本评测动作契约完全一致。
    semantics = dataset_info.get("action_semantics")
    if semantics is not None and semantics != ACTION_SEMANTICS:
        raise ValueError(f"训练数据动作语义 {semantics!r} 不适用于当前仿真")
    return features


def choose_device(flag, torch):
    if flag == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("显式指定 CUDA 但没有可用 GPU，不自动回退 CPU")
    return torch.device("cuda" if (flag == "cuda" or (flag == "auto" and torch.cuda.is_available())) else "cpu")


class PolicyAdapter:
    """模型/处理器与环境隔离：统一 reset()、select_action(step) 契约。"""

    def __init__(self, args, env):
        import numpy as np
        import torch
        from PIL import Image
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import get_policy_class, make_pre_post_processors
        from lerobot.policies.utils import build_inference_frame

        self.np, self.torch, self.Image = np, torch, Image
        self.device = choose_device(args.device, torch)
        self.env, self.args = env, args
        self.build_inference_frame = build_inference_frame
        ckpt = args.checkpoint
        info = json.loads((args.dataset_root / "meta/info.json").read_text(encoding="utf-8"))
        cfg = json.loads((ckpt / "config.json").read_text(encoding="utf-8"))
        self.features = check_schema(cfg, info, policy_name=args.policy,
                                     top_feature=args.top_feature, wrist_feature=args.wrist_feature)
        self.state_names = self.features["observation.state"]["names"]
        self.in_features = {key: self.features[key] for key in cfg["input_features"]}
        self.image_keys = (args.top_feature, args.wrist_feature)
        for key in self.image_keys:
            if self.features[key]["shape"][0] <= 0 or self.features[key]["shape"][1] <= 0:
                raise ValueError(f"图像尺寸必须为正数: {key}")
        if not (ckpt / "model.safetensors").is_file():
            raise FileNotFoundError(f"本地 checkpoint 缺少 model.safetensors: {ckpt}")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        policy_cfg = PreTrainedConfig.from_pretrained(str(ckpt), local_files_only=True)
        policy_cfg.device = str(self.device)
        if hasattr(policy_cfg, "pretrained_backbone_weights"):
            policy_cfg.pretrained_backbone_weights = None
        if hasattr(policy_cfg, "load_vlm_weights"):
            policy_cfg.load_vlm_weights = False
        self.policy = get_policy_class(args.policy).from_pretrained(
            str(ckpt), config=policy_cfg, local_files_only=True
        ).to(self.device).eval()
        if args.processors == "saved":
            for name in ("policy_preprocessor.json", "policy_postprocessor.json"):
                if not (ckpt / name).is_file():
                    raise FileNotFoundError(f"缺少已保存的预处理器: {ckpt / name}")
            self.pre, self.post = make_pre_post_processors(
                self.policy.config, str(ckpt),
                preprocessor_overrides={"device_processor": {"device": str(self.device)}},
            )
        else:
            stats_file = args.dataset_root / "meta/stats.json"
            stats = json.loads(stats_file.read_text(encoding="utf-8"))
            tensor_stats = {k: {n: torch.as_tensor(v) for n, v in item.items()} for k, item in stats.items()}
            self.pre, self.post = make_pre_post_processors(self.policy.config, dataset_stats=tensor_stats)

    def reset(self):
        self.policy.reset()
        for module in (self.pre, self.post):
            if hasattr(module, "reset"):
                module.reset()

    def _rgb(self, image, key):
        array = self.np.asarray(image)
        if array.ndim != 3 or array.shape[-1] != 3 or array.dtype != self.np.uint8:
            raise ValueError(f"{key} 必须为 H×W×3 uint8 RGB 图像")
        height, width = self.features[key]["shape"][:2]
        if array.shape[:2] != (height, width):
            array = self.np.asarray(self.Image.fromarray(array).resize((width, height)))
        return array

    def select_action(self, step):
        state = self.np.asarray(self.env.get_ee_pose(), dtype=self.np.float32)
        if state.shape != (6,) or not self.np.all(self.np.isfinite(state)):
            raise ValueError("MuJoCo 返回的末端 xyz+rpy 状态不是六维有限数值")
        top, wrist = self.env.grab_image()
        raw = dict(zip(self.state_names, map(float, state), strict=True))
        for key, image in zip(self.image_keys, (top, wrist), strict=True):
            # LeRobot 0.5.1 的 build_dataset_frame 对 observation.images.* 使用去前缀后的值键。
            raw[key.removeprefix("observation.images.")] = self._rgb(image, key)
        frame = self.build_inference_frame(
            observation=raw, device=self.device, ds_features=self.in_features,
            task=self.args.task, robot_type="so101_follower",
        )
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        with self.torch.inference_mode():
            action = self.post(self.policy.select_action(self.pre(frame)))
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)
        ms = (time.perf_counter() - started) * 1000.0
        values = action.detach().cpu().numpy().reshape(-1)
        if values.shape != (6,) or not self.np.all(self.np.isfinite(values)):
            raise ValueError(f"策略输出必须是六维有限动作，实际为 {values.shape}")
        if not (0 <= values[-1] <= 1):
            raise ValueError("策略夹爪输出必须是闭合比例 [0,1]；可能缺少反归一化")
        return values.astype(self.np.float32), ms


def main(argv=None):
    args = parse_args(argv)
    plan = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    if args.dry_run:
        print("DRY RUN: 不读取 checkpoint / 数据集、不加载 MuJoCo/模型、不使用 GPU")
        return
    import numpy as np
    from mujoco_env.y_env import SimpleEnv
    for p in (args.checkpoint / "config.json", args.dataset_root / "meta/info.json", args.xml_path):
        if not p.is_file():
            raise FileNotFoundError(p)
    env = SimpleEnv(
        str(args.xml_path), action_type="joint_angle",
        pick_body_name=args.pick_body_name, place_body_name=args.place_body_name,
        spawn_x_range=(args.spawn_x_min, args.spawn_x_max),
        spawn_y_range=(args.spawn_y_min, args.spawn_y_max),
        spawn_z_range=(args.spawn_z_min, args.spawn_z_max),
        spawn_min_dist=args.spawn_min_dist,
        spawn_xy_margin=args.spawn_xy_margin,
        spawn_fallback_min_dist=args.spawn_fallback_min_dist,
    )
    recorder = None
    try:
        adapter = PolicyAdapter(args, env)
        try:
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
            dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
        except (OSError, subprocess.CalledProcessError):
            commit, dirty = None, None
        files = ("scripts/eval/eval_mujoco.py", "scripts/eval/act_rollout.py", "mujoco_env/y_env.py")
        checksums = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in files}
        recorder = JsonlRecorder(args.output, {
            "split": args.split, "checkpoint": str(args.checkpoint), "dataset_root": str(args.dataset_root),
            "task": args.task, "control_hz": args.hz, "timeout_s": args.timeout_s,
            "action_semantics": args.action_semantics, "processors": args.processors,
            "top_feature": args.top_feature, "wrist_feature": args.wrist_feature,
            "git_commit": commit, "working_tree_dirty": dirty, "source_sha256": checksums,
            "model_type": args.policy, "device": str(adapter.device),
            "spawn_ranges": {"x": [args.spawn_x_min, args.spawn_x_max],
                             "y": [args.spawn_y_min, args.spawn_y_max],
                             "z": [args.spawn_z_min, args.spawn_z_max]},
        })
        result = run_evaluation(
            env, adapter.select_action, adapter.reset, policy_name=args.policy,
            num_trials=args.num_trials, hz=args.hz, timeout_s=args.timeout_s,
            seed=args.seed, timeout_clock=args.timeout_clock, record=recorder,
        )
        print(json.dumps({"policy": args.policy, "split": args.split, **result}, ensure_ascii=False))
    finally:
        if recorder is not None:
            recorder.close()
        env.env.close_viewer()


if __name__ == "__main__":
    main()
