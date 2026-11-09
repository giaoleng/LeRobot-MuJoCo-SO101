#!/usr/bin/env python3
"""人工复位与判定的SO-101多Episode评测，输出兼容汇总器的JSONL。"""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.real.deploy_policy import metadata, parser, prepare
from scripts.real.runtime import RealPolicyRunner, local_path
from scripts.real.task_conditioning import TaskConditioner
from scripts.eval.act_rollout import JsonlRecorder


def read_outcome(input_fn=input):
    """不将运行结束或机械臂动作当作任务成功，收集操作者判定。"""
    while True:
        raw = input_fn('输入结果JSON，例如 {"success":true,"correct_target":true,"grasped":true,"placed":true}： ')
        try:
            result = json.loads(raw)
            allowed = {'success', 'correct_target', 'grasped', 'placed'}
            if not isinstance(result, dict) or set(result) - allowed or type(result.get('success')) is not bool:
                raise ValueError('需要success布尔字段，其他字段仅允许correct_target/grasped/placed')
            if any(type(value) is not bool for value in result.values()):
                raise ValueError('结果字段必须为布尔值')
            if result['success'] and any(value is False for value in result.values()):
                raise ValueError('success=true不能同时包含失败阶段')
            if all(key in result for key in ('correct_target', 'grasped', 'placed')):
                actual = all(result[key] for key in ('correct_target', 'grasped', 'placed'))
                if result['success'] != actual:
                    raise ValueError('success应等于正确目标、抓取、放置三项的逻辑与')
            return result
        except (ValueError, TypeError) as exc:
            print(f'结果格式错误：{exc}')


def main(argv=None):
    p = parser(__doc__)
    p.add_argument('--episodes', type=int)
    p.add_argument('--targets', nargs='+', help='按此顺序循环切换目标')
    p.add_argument('--split')
    args = p.parse_args(argv)
    cfg, default_target = prepare(args)
    episodes = args.episodes if args.episodes is not None else cfg['evaluation']['episodes']
    if episodes <= 0:
        p.error('--episodes必须为正整数')
    targets = args.targets or [default_target]
    conditioner = TaskConditioner(cfg['conditioning'])
    for target in targets:
        conditioner.task(target)
    print(json.dumps({'episodes': episodes, 'target_sequence': targets,
                      'split': args.split or cfg['evaluation']['split']}, ensure_ascii=False))
    if not args.execute:
        return
    output = local_path(args.output) if args.output else Path(cfg['evaluation']['output'])
    recorder = JsonlRecorder(output, {**metadata(cfg), 'split': args.split or cfg['evaluation']['split'],
                                      'outcome_source': 'operator'})
    episode, target, result = 0, targets[0], {}
    try:
        runner = RealPolicyRunner(cfg)
        for episode in range(episodes):
            target = targets[episode % len(targets)]
            result = {}
            input(f'Episode {episode}: {conditioner.task(target)}\n请复位场景、将机械臂置于起始姿态，按Enter连接并执行： ')
            result = runner.run_episode(target)
            if result['reason'] == 'stop_file':
                recorder({**result, 'episode': episode, 'target_id': target,
                          'success': False, 'completed': False})
                break
            outcome = read_outcome()
            recorder({**result, **outcome, 'episode': episode, 'target_id': target, 'completed': True})
    except BaseException as exc:
        recorder({**result, 'episode': episode, 'target_id': target, 'success': False,
                  'completed': False, 'reason': 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'error',
                  'error_type': type(exc).__name__})
        raise
    finally:
        recorder.close()


if __name__ == '__main__':
    main()
