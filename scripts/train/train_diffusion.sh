#!/usr/bin/env bash
# Train LeRobot's built-in Diffusion Policy on a private local dataset.
# Example:
# DATASET_ROOT=/data/so101_sim DATASET_REPO_ID=local/so101_sim \
#   bash scripts/train/train_diffusion.sh --dry-run
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh" "$@"
so101_required_environment
so101_private_defaults

so101_execute lerobot-train \
  "--dataset.repo_id=${DATASET_REPO_ID}" \
  "--dataset.root=${DATASET_ROOT}" \
  "--policy.type=diffusion" \
  "--policy.device=cuda" \
  "--policy.push_to_hub=false" \
  "--output_dir=${OUTPUT_DIR:-${SO101_REPO_ROOT}/outputs/train/diffusion}" \
  "--job_name=${JOB_NAME:-so101_diffusion}" \
  "--steps=${TRAIN_STEPS:-30000}" \
  "--batch_size=${BATCH_SIZE:-8}" \
  "${SO101_EXTRA_ARGS[@]}"
