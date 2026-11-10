"""MuJoCo策略评测调度与JSONL记录；支持ACT/DP/SmolVLA，保留旧ACT调用兼容。"""
from __future__ import annotations

import json
import math
from pathlib import Path
import time


class JsonlRecorder:
    def __init__(self, path, metadata):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.metadata = dict(metadata)
        # 每次评测使用新文件，避免混入旧checkpoint或覆盖旧记录。
        self._file = self.path.open('x', encoding='utf-8')

    def __call__(self, event):
        row = {**self.metadata, **event}
        payload = json.dumps(row, ensure_ascii=False, allow_nan=False)
        self._file.write(payload + '\n')
        self._file.flush()

    def close(self):
        self._file.close()


def run_evaluation(env, select_action, reset_policy, *, num_trials, hz, timeout_s,
                   seed=0, timeout_clock='wall', record=None, clock=time.monotonic, policy_name='act'):
    """select_action返回(action, inference_ms)，不承担模型归一化或动作转换。"""
    if num_trials < 0 or hz <= 0 or not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError('num_trials必须非负，hz和timeout_s必须为正')
    if timeout_clock not in ('wall', 'sim'):
        raise ValueError('timeout_clock must be wall or sim')
    if policy_name not in ('act', 'diffusion', 'smolvla'):
        raise ValueError('policy_name必须是act/diffusion/smolvla')
    completed = successes = timeouts = 0
    record = record or (lambda event: None)
    while num_trials == 0 or completed < num_trials:
        episode_seed = seed + completed
        env.reset(seed=episode_seed)
        reset_policy()
        start_wall = clock()
        start_sim = env.env.get_sim_time()
        latencies = []
        steps = 0
        wall_elapsed = sim_elapsed = 0.0

        def event(success, reason, done):
            return {
                'policy': policy_name, 'episode': completed, 'seed': episode_seed,
                'success': success, 'completed': done, 'reason': reason,
                'action_emitted': steps > 0, 'control_steps': steps,
                'completion_time_s': sim_elapsed if timeout_clock == 'sim' else wall_elapsed,
                'wall_time_s': wall_elapsed, 'simulation_time_s': sim_elapsed,
                'timeout_clock': timeout_clock, 'inference_ms': list(latencies),
            }

        while env.env.is_viewer_alive():
            env.step_env()
            if not env.env.loop_every(HZ=hz):
                continue
            wall_elapsed = clock() - start_wall
            sim_elapsed = env.env.get_sim_time() - start_sim
            elapsed = sim_elapsed if timeout_clock == 'sim' else wall_elapsed
            try:
                success = bool(env.check_success())
                if success or elapsed >= timeout_s:
                    record(event(success, 'success' if success else 'timeout', True))
                    successes += int(success)
                    timeouts += int(not success)
                    completed += 1
                    break
                action, latency = select_action(steps)
                if not math.isfinite(latency) or latency < 0:
                    raise ValueError('inference latency must be finite and nonnegative')
                env.step(action)
                latencies.append(float(latency))
                steps += 1
                env.render()
            except Exception as exc:
                row = event(False, 'execution_error', False)
                row['error_type'] = type(exc).__name__
                record(row)
                raise
        else:
            record(event(False, 'viewer_closed', False))
            break
    return {'completed': completed, 'successes': successes, 'timeouts': timeouts}
