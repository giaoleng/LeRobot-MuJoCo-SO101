SO-101 three-policy non-destructive overlay (2026-11-04)

Contents:
- scripts/train/train_act.sh : optional standardized ACT entry (preserves existing ACT code)
- scripts/train/train_diffusion.sh : LeRobot Diffusion training entry
- scripts/train/finetune_smolvla.sh : pretrained SmolVLA fine-tune entry
- scripts/train/_common.sh : shared privacy checks and flags
- scripts/eval/summarize_rollouts.py : privacy-preserving aggregate statistics
- docs/README_UPGRADE.md : README introduction and headings to merge
- docs/POLICY_COMPARISON.md : unified evaluation plan and commands
- docs/REAL_DEPLOYMENT.md : real-hardware safety gate
- tests/test_summarize_rollouts.py : lightweight unit tests

Important: no existing repo code is overridden; README must be merged manually.
No eval_mujoco.py or real-hardware driver is included because those need the actual repository API / robot calibration.

Do not copy test files over existing versions without reviewing git diff.
