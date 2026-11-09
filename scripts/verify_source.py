#!/usr/bin/env python3
"""无模型/仿真依赖的 AST、XML 和主场景本地资源检查。"""
import ast
import subprocess
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]


def check_scene(path, visited=None):
    """验证 include 和显式资源路径存在；不替代 MuJoCo 编译器语义检查。"""
    visited = set() if visited is None else visited
    path = path.resolve()
    if path in visited:
        return
    visited.add(path)
    tree = ET.parse(path)
    for element in tree.iter():
        if 'file' not in element.attrib:
            continue
        target = path.parent / element.attrib['file']
        if not target.is_file():
            raise FileNotFoundError(f'{path.relative_to(ROOT)}: missing {target}')
        if element.tag == 'include':
            check_scene(target, visited)
    return visited


def main():
    for relative in (
        'scripts/real/deploy_policy.py', 'scripts/real/evaluate_policy.py',
        'scripts/real/task_conditioning.py', 'scripts/real/runtime.py',
        'scripts/real/train_target_act.py', 'configs/real/so101.yaml',
        'configs/real/requirements.txt',
    ):
        if not (ROOT / relative).is_file():
            raise FileNotFoundError(f'缺少真机入口/配置：{relative}')
    tracked = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0')
    sources = set(ROOT / f for f in tracked if f.endswith('.py'))
    for folder in ('scripts', 'tests', 'controllers', 'mujoco_env', 'so101'):
        sources.update((ROOT / folder).rglob('*.py'))
    sources.add(Path(__file__).resolve())
    for p in sources:
        ast.parse(p.read_text(encoding='utf-8'), filename=str(p))
    xmls = [ROOT / f for f in tracked if f.endswith('.xml')]
    for p in xmls:
        ET.parse(p)
    visited = check_scene(ROOT / 'asset/scene_so101_y.xml')
    print(f'AST: {len(sources)} Python; XML parse: {len(xmls)}; SO101 scene includes: {len(visited)}')
    print('未运行 MuJoCo 编译、reset/step、模型推理或训练；其他可选场景可能缺少外部资产。')


if __name__ == '__main__':
    main()
