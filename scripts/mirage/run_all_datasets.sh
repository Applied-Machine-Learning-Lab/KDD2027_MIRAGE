#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

DATASETS=(beauty sports toys cds)
DEVICE="${DEVICE:-cuda:0}"
SEED="${SEED:-2026}"
DRY_RUN="${DRY_RUN:-0}"
EXTRA_ARGS=("$@")

for dataset in "${DATASETS[@]}"; do
  run_name="official-${dataset}-seed${SEED}-$(date +%Y%m%d-%H%M%S)"
  command=(bash scripts/mirage/run_mirage.sh "$dataset" "${EXTRA_ARGS[@]}")
  echo ">>> MIRAGE dataset=$dataset run=$run_name"
  if [[ "$DRY_RUN" == "1" ]]; then
    printf 'COMMAND:'
    printf ' %q' "${command[@]}"
    printf '\n'
  else
    DEVICE="$DEVICE" SEED="$SEED" RUN_NAME="$run_name" "${command[@]}"
  fi
done
