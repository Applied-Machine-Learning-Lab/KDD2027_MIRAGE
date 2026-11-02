#!/usr/bin/env bash
# ===========================================================================
# MIRAGE Ablation Study — all datasets (RQ2)
#
# Variants per dataset:
#   1. full          — all MIRAGE components enabled
#   2. no_graph      — graph_weight=0 (naive Euclidean flow)
#   3. no_time_mod   — constant graph penalty, no w(t)=4t(1-t)
#   4. softmin       — softmin distance aggregation instead of hard-min
#   5. user_anchor   — user-level anchor instead of target-only anchor
#   6. endpoint_reg  — endpoint-only regularization (w/ Endpoint Reg.)
#                       Applies graph penalty only at t=1, testing whether
#                       path-interior rectification is necessary or if
#                       generic graph smoothing at endpoints suffices.
#
# Dataset hyperparameters are the guarded-best recommendations from:
#   scripts/mirage/analyze/analyze_hparam_sweep.py
# using actual completed runs rather than per-parameter stitched settings.
#
# Conservative scheduling defaults:
#   GPU_LIST=0       use only one GPU unless explicitly overridden
#   MAX_PARALLEL=1   run one experiment at a time by default
#
# Examples:
#   DRY_RUN=1 bash scripts/mirage/run_all_ablation.sh
#   GPU_LIST=0 MAX_PARALLEL=1 bash scripts/mirage/run_all_ablation.sh
#   GPU_LIST=0,1 MAX_PARALLEL=2 bash scripts/mirage/run_all_ablation.sh
#   DATASETS=toys,cds GPU_LIST=0 MAX_PARALLEL=1 bash scripts/mirage/run_all_ablation.sh
#
# To run only the new endpoint_reg variant (if other variants already completed):
#   VARIANTS=endpoint_reg GPU_LIST=0 bash scripts/mirage/run_all_ablation.sh
# ===========================================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

OUTPUT_BASE="${OUTPUT_BASE:-scripts/output/mirage/ablation_all}"
SEED="${SEED:-2026}"
RUN_PREFIX="${RUN_PREFIX:-abl}"
DATASETS="${DATASETS:-beauty,sports,toys,cds}"
VARIANTS="${VARIANTS:-full,no_graph,no_time_mod,softmin,user_anchor,endpoint_reg}"
GRAPH_SOURCE="${GRAPH_SOURCE:-cooccur}"
SAMPLE_STEPS="${SAMPLE_STEPS:-1}"
GPU_LIST="${GPU_LIST:-0}"
MAX_PARALLEL="${MAX_PARALLEL:-1}"
MEMORY_LIMIT_MB="${MEMORY_LIMIT_MB:-19000}"
POLL_SECONDS="${POLL_SECONDS:-60}"
DRY_RUN="${DRY_RUN:-0}"
EXTRA_ARGS=("$@")

mkdir -p "$OUTPUT_BASE"

# dataset|graph_k|graph_weight|recommendation_weight|hidden_size
declare -a DATASET_CONFIGS=(
  "beauty|20|0.1|0.2|128"
  "sports|20|0.1|0.2|128"
  "toys|40|0.1|0.2|128"
  "cds|20|0.1|0.5|128"
)

# Parse VARIANTS from environment variable
IFS=',' read -r -a VARIANTS <<< "$VARIANTS"
if [[ "${#VARIANTS[@]}" -eq 0 || -z "${VARIANTS[0]}" ]]; then
  echo "VARIANTS must contain at least one variant" >&2
  exit 2
fi

IFS=',' read -r -a GPUS <<< "$GPU_LIST"
if [[ "${#GPUS[@]}" -eq 0 || -z "${GPUS[0]}" ]]; then
  echo "GPU_LIST must contain at least one GPU id, e.g. GPU_LIST=0 or GPU_LIST=0,1" >&2
  exit 2
fi
if [[ "$MAX_PARALLEL" -lt 1 ]]; then
  echo "MAX_PARALLEL must be >= 1" >&2
  exit 2
fi

variant_env() {
  local variant="$1"
  local graph_weight="$2"
  case "$variant" in
    full)
      printf 'GRAPH_WEIGHT=%s' "$graph_weight"
      ;;
    no_graph)
      printf 'GRAPH_WEIGHT=0.0'
      ;;
    no_time_mod)
      printf 'GRAPH_WEIGHT=%s;NO_GRAPH_TIME_MOD=1' "$graph_weight"
      ;;
    softmin)
      printf 'GRAPH_WEIGHT=%s;GRAPH_DISTANCE_MODE=softmin' "$graph_weight"
      ;;
    user_anchor)
      printf 'GRAPH_WEIGHT=%s;GRAPH_ANCHOR_MODE=user' "$graph_weight"
      ;;
    endpoint_reg)
      printf 'GRAPH_WEIGHT=%s;ENDPOINT_ONLY_REG=1' "$graph_weight"
      ;;
    *)
      echo "unknown variant: $variant" >&2
      return 2
      ;;
  esac
}

dataset_enabled() {
  local dataset="$1"
  local selected
  IFS=',' read -r -a selected <<< "$DATASETS"
  local value
  for value in "${selected[@]}"; do
    [[ "$value" == "$dataset" ]] && return 0
  done
  return 1
}

gpu_memory_used() {
  local gpu="$1"
  nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
    | tr -dc '0-9' || true
}

gpu_is_available() {
  local gpu="$1"
  local used
  used="$(gpu_memory_used "$gpu")"
  if [[ -z "$used" ]]; then
    return 0
  fi
  [[ "$used" -lt "$MEMORY_LIMIT_MB" ]]
}

running_count() {
  jobs -pr | wc -l | tr -d ' '
}

wait_for_slot() {
  while [[ "$(running_count)" -ge "$MAX_PARALLEL" ]]; do
    sleep "$POLL_SECONDS"
  done
}

# 全局变量记录状态，避免使用子进程 $() 导致状态丢失
CURRENT_GPU_IDX=0
SELECTED_GPU=""

select_gpu() {
  local num_gpus=${#GPUS[@]}
  local check_count=0
  
  while true; do
    while [[ $check_count -lt $num_gpus ]]; do
      local gpu="${GPUS[$CURRENT_GPU_IDX]}"
      # 指针移到下一张卡
      CURRENT_GPU_IDX=$(( (CURRENT_GPU_IDX + 1) % num_gpus ))
      check_count=$(( check_count + 1 ))
      
      if gpu_is_available "$gpu"; then
        # 直接修改全局变量，而不是 echo
        SELECTED_GPU="$gpu"
        return 0
      fi
    done
    
    # 如果一圈找下来都没有空闲的，重置计数器并等待
    check_count=0
    sleep "$POLL_SECONDS"
  done
}

launch_task() {
  local dataset="$1"
  local graph_k="$2"
  local graph_weight="$3"
  local recommendation_weight="$4"
  local hidden_size="$5"
  local variant="$6"
  local gpu="$7"

  local run_name="${RUN_PREFIX}_${variant}_${dataset}_seed${SEED}_gk${graph_k}_gw${graph_weight}_rw${recommendation_weight}_hs${hidden_size}"
  local dataset_log_dir="${OUTPUT_BASE}/${dataset}"
  local launcher_log="${dataset_log_dir}/${run_name}.launcher.log"
  local env_str
  env_str="$(variant_env "$variant" "$graph_weight")"
  mkdir -p "$dataset_log_dir"

  local command=(
    env
    DEVICE="cuda:${gpu}"
    SEED="$SEED"
    RUN_NAME="$run_name"
    GRAPH_K="$graph_k"
    RECOMMENDATION_WEIGHT="$recommendation_weight"
    HIDDEN_SIZE="$hidden_size"
    GRAPH_SOURCE="$GRAPH_SOURCE"
    SAMPLE_STEPS="$SAMPLE_STEPS"
  )

  IFS=';' read -r -a env_parts <<< "$env_str"
  local part
  for part in "${env_parts[@]}"; do
    [[ -n "$part" ]] && command+=("$part")
  done
  command+=(bash scripts/mirage/run_mirage.sh "$dataset")
  command+=("${EXTRA_ARGS[@]}")

  echo ">>> dataset=${dataset} variant=${variant} gpu=${gpu} run=${run_name}"
  echo "    hparams: graph_k=${graph_k} graph_weight=${graph_weight} recommendation_weight=${recommendation_weight} hidden_size=${hidden_size}"
  echo "    launcher_log=${launcher_log}"
  if [[ "$DRY_RUN" == "1" ]]; then
    printf 'COMMAND:'
    printf ' %q' "${command[@]}"
    printf '\n'
    return 0
  fi

  "${command[@]}" > "$launcher_log" 2>&1 &
}

summarize() {
  local summary_file="${OUTPUT_BASE}/summary.tsv"
  {
    printf "dataset\tvariant\trun_name\tHR@5\tNDCG@5\tHR@10\tNDCG@10\tHR@20\tNDCG@20\tbest_epoch\n"
    local row dataset graph_k graph_weight recommendation_weight hidden_size variant run_name metrics_file
    for row in "${DATASET_CONFIGS[@]}"; do
      IFS='|' read -r dataset graph_k graph_weight recommendation_weight hidden_size <<< "$row"
      dataset_enabled "$dataset" || continue
      for variant in "${VARIANTS[@]}"; do
        run_name="${RUN_PREFIX}_${variant}_${dataset}_seed${SEED}_gk${graph_k}_gw${graph_weight}_rw${recommendation_weight}_hs${hidden_size}"
        metrics_file="scripts/output/mirage/${dataset}/${run_name}/metrics.json"
        if [[ -f "$metrics_file" ]]; then
          ./.venv/bin/python - "$dataset" "$variant" "$run_name" "$metrics_file" <<'PY'
import json
import sys

dataset, variant, run_name, metrics_file = sys.argv[1:]
with open(metrics_file) as handle:
    data = json.load(handle)
test = data.get("test", {})
fields = [
    dataset,
    variant,
    run_name,
    *(f"{test.get(metric, float('nan')):.6f}" for metric in ("HR@5", "NDCG@5", "HR@10", "NDCG@10", "HR@20", "NDCG@20")),
    str(data.get("best_epoch", "")),
]
print("\t".join(fields))
PY
        else
          printf "%s\t%s\t%s\tNA\tNA\tNA\tNA\tNA\tNA\tNA\n" "$dataset" "$variant" "$run_name"
        fi
      done
    done
  } > "$summary_file"
  echo "Summary written to ${summary_file}"
}

echo "MIRAGE ablation runner"
echo "  datasets: ${DATASET_CONFIGS[*]}"
echo "  selected_datasets: ${DATASETS}"
echo "  variants: ${VARIANTS[*]}"
echo "  gpu_list: ${GPU_LIST}"
echo "  max_parallel: ${MAX_PARALLEL}"
echo "  memory_limit_mb: ${MEMORY_LIMIT_MB}"
echo "  output_base: ${OUTPUT_BASE}"

TASK_INDEX=0
for row in "${DATASET_CONFIGS[@]}"; do
  IFS='|' read -r dataset graph_k graph_weight recommendation_weight hidden_size <<< "$row"
  dataset_enabled "$dataset" || continue
  for variant in "${VARIANTS[@]}"; do
    if [[ "$DRY_RUN" == "1" ]]; then
      gpu="${GPUS[$((TASK_INDEX % ${#GPUS[@]}))]}"
    else
      wait_for_slot
      # 直接调用函数，通过全局变量 SELECTED_GPU 拿结果，避开子进程陷阱
      select_gpu
      gpu="$SELECTED_GPU"
    fi
    launch_task "$dataset" "$graph_k" "$graph_weight" "$recommendation_weight" "$hidden_size" "$variant" "$gpu"
    TASK_INDEX=$((TASK_INDEX + 1))
    [[ "$DRY_RUN" == "1" ]] || sleep 15
  done
done

if [[ "$DRY_RUN" == "1" ]]; then
  exit 0
fi

status=0
while [[ "$(running_count)" -gt 0 ]]; do
  if ! wait -n; then
    status=1
  fi
done

summarize
exit "$status"
