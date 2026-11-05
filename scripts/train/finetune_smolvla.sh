#!/usr/bin/env bash
# Finetune a *pre-downloaded* SmolVLA checkpoint. Intended for remote BitaHub GPU.
# SmolVLA's base weights may not be downloadable from BitaHub; supply a local path.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh" "$@"
so101_required_environment
so101_private_defaults

if [[ -z "${SMOLVLA_CHECKPOINT:-}" ]]; then
  echo 'Set SMOLVLA_CHECKPOINT to a local pretrained SmolVLA directory (e.g., /weights/smolvla_base).' >&2
  exit 2
fi
if [[ "$SO101_DRY_RUN" != 1 && ! -d "$SMOLVLA_CHECKPOINT" ]]; then
  echo "Missing pretrained directory: $SMOLVLA_CHECKPOINT" >&2
  exit 2
fi

cmd=(lerobot-train
  "--dataset.repo_id=${DATASET_REPO_ID}"
  "--dataset.root=${DATASET_ROOT}"
  "--policy.path=${SMOLVLA_CHECKPOINT}"
  "--policy.device=cuda"
  "--policy.push_to_hub=false"
  "--output_dir=${OUTPUT_DIR:-${SO101_REPO_ROOT}/outputs/train/smolvla}"
  "--job_name=${JOB_NAME:-so101_smolvla}"
  "--steps=${TRAIN_STEPS:-20000}"
  "--batch_size=${BATCH_SIZE:-2}"
)
if [[ -n "${SMOLVLA_RENAME_MAP:-}" ]]; then
  cmd+=("--rename_map=${SMOLVLA_RENAME_MAP}")
fi
so101_execute "${cmd[@]}" "${SO101_EXTRA_ARGS[@]}"
