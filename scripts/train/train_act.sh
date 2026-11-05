#!/usr/bin/env bash
# Optional standardized ACT trainer for comparison; does not replace existing ACT workflow.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh" "$@"
so101_required_environment
so101_private_defaults

so101_execute lerobot-train \
  "--dataset.repo_id=${DATASET_REPO_ID}" \
  "--dataset.root=${DATASET_ROOT}" \
  "--policy.type=act" \
  "--policy.device=cuda" \
  "--policy.push_to_hub=false" \
  "--output_dir=${OUTPUT_DIR:-${SO101_REPO_ROOT}/outputs/train/act}" \
  "--job_name=${JOB_NAME:-so101_act}" \
  "--steps=${TRAIN_STEPS:-30000}" \
  "--batch_size=${BATCH_SIZE:-8}" \
  "${SO101_EXTRA_ARGS[@]}"
