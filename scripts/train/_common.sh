#!/usr/bin/env bash
# Shared checks for the repository's LeRobot training entry points.
set -euo pipefail

SO101_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SO101_DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  SO101_DRY_RUN=1
  shift
fi
SO101_EXTRA_ARGS=("$@")

so101_required_environment() {
  if [[ -z "${DATASET_ROOT:-}" || -z "${DATASET_REPO_ID:-}" ]]; then
    cat >&2 <<'HELP'
Missing environment variables:
  DATASET_ROOT       absolute location of a private LeRobotDataset v3 directory
  DATASET_REPO_ID    dataset identifier, e.g. local/so101_sim (no upload involved)
Example:
  DATASET_ROOT=/private/data/so101 DATASET_REPO_ID=local/so101_sim bash scripts/train/train_diffusion.sh --dry-run
HELP
    exit 2
  fi
  if [[ "$SO101_DRY_RUN" != 1 ]]; then
    [[ -f "${DATASET_ROOT}/meta/info.json" ]] || { echo "Missing ${DATASET_ROOT}/meta/info.json" >&2; exit 2; }
    command -v lerobot-train >/dev/null || { echo "lerobot-train is not installed in this environment" >&2; exit 2; }
  fi
}

so101_private_defaults() {
  export WANDB_MODE=disabled
  export HF_HUB_DISABLE_TELEMETRY=1
  export DO_NOT_TRACK=1
}

so101_execute() {
  local -a cmd=("$@")
  printf 'Command: '
  printf '%q ' "${cmd[@]}"
  printf '\n'
  if [[ "$SO101_DRY_RUN" == 1 ]]; then
    echo 'DRY RUN: no dataset, model, or GPU accessed.'
  else
    cd "$SO101_REPO_ROOT"
    exec "${cmd[@]}"
  fi
}
