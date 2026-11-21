#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

DATASET="${1:-beauty}"
DEVICE="${DEVICE:-cuda:0}"
SEED="${SEED:-2026}"
RUN_NAME="${RUN_NAME:-official-${DATASET}-seed${SEED}-$(date +%Y%m%d-%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-scripts/output/mirage}"
NUM_WORKERS="${NUM_WORKERS:-4}"

case "$DATASET" in
  beauty)
    GRAPH_K="${GRAPH_K:-20}"
    GRAPH_WEIGHT="${GRAPH_WEIGHT:-0.1}"
    RECOMMENDATION_WEIGHT="${RECOMMENDATION_WEIGHT:-0.2}"
    HIDDEN_SIZE="${HIDDEN_SIZE:-128}"
    GRAPH_DISTANCE_MODE="${GRAPH_DISTANCE_MODE:-hardmin}"
    ;;
  sports)
    GRAPH_K="${GRAPH_K:-20}"
    GRAPH_WEIGHT="${GRAPH_WEIGHT:-0.1}"
    RECOMMENDATION_WEIGHT="${RECOMMENDATION_WEIGHT:-0.2}"
    HIDDEN_SIZE="${HIDDEN_SIZE:-128}"
    GRAPH_DISTANCE_MODE="${GRAPH_DISTANCE_MODE:-hardmin}"
    ;;
  toys)
    GRAPH_K="${GRAPH_K:-40}"
    GRAPH_WEIGHT="${GRAPH_WEIGHT:-0.1}"
    RECOMMENDATION_WEIGHT="${RECOMMENDATION_WEIGHT:-0.2}"
    HIDDEN_SIZE="${HIDDEN_SIZE:-128}"
    GRAPH_DISTANCE_MODE="${GRAPH_DISTANCE_MODE:-hardmin}"
    ;;
  cds)
    GRAPH_K="${GRAPH_K:-20}"
    GRAPH_WEIGHT="${GRAPH_WEIGHT:-0.1}"
    RECOMMENDATION_WEIGHT="${RECOMMENDATION_WEIGHT:-0.5}"
    HIDDEN_SIZE="${HIDDEN_SIZE:-128}"
    GRAPH_DISTANCE_MODE="${GRAPH_DISTANCE_MODE:-softmin}"
    ;;
  *)
    echo "Unsupported dataset: $DATASET" >&2
    exit 2
    ;;
esac

if [[ -z "${PYTHON:-}" && -x ./.venv/bin/python ]]; then
  PYTHON_BIN="./.venv/bin/python"
else
  PYTHON_BIN="${PYTHON:-python}"
fi

exec "$PYTHON_BIN" -m mirage.train \
  --dataset "$DATASET" \
  --device "$DEVICE" \
  --seed "$SEED" \
  --eval-seed "$SEED" \
  --run-name "$RUN_NAME" \
  --output-root "$OUTPUT_ROOT" \
  --num-workers "$NUM_WORKERS" \
  --max-len 20 \
  --hidden-size "$HIDDEN_SIZE" \
  --num-blocks 4 \
  --attention-heads 4 \
  --last-blocks 2 \
  --dropout 0.1 \
  --embedding-dropout 0.3 \
  --decoder-dims 512,2048 \
  --eps 0.001 \
  --lambda-uncertainty 0.001 \
  --sample-steps "${SAMPLE_STEPS:-1}" \
  --sampling-method mode \
  --mode-scale 1.0 \
  --mask-ratio 1.0 \
  --recommendation-weight "$RECOMMENDATION_WEIGHT" \
  --reconstruction-weight 0.4 \
  --loss-type sampled_softmax \
  --negative-count 10 \
  --negative-sampling shared_pool \
  --graph-weight "$GRAPH_WEIGHT" \
  --graph-k "$GRAPH_K" \
  --graph-source "${GRAPH_SOURCE:-cooccur}" \
  --graph-warmup-epochs "${GRAPH_WARMUP_EPOCHS:-0}" \
  --graph-anchor-mode "${GRAPH_ANCHOR_MODE:-target}" \
  ${USE_HISTORY_ANCHOR:+--use-history-anchor} \
  ${NO_GRAPH_TIME_MOD:+--no-graph-time-mod} \
  ${ENDPOINT_ONLY_REG:+--endpoint-only-reg} \
  --graph-distance-mode "$GRAPH_DISTANCE_MODE" \
  --batch-size "${BATCH_SIZE:-512}" \
  --eval-batch-size "${EVAL_BATCH_SIZE:-512}" \
  --epochs "${EPOCHS:-500}" \
  --learning-rate 0.001 \
  --weight-decay 0 \
  --decay-step 100 \
  --decay-gamma 0.1 \
  --eval-interval "${EVAL_INTERVAL:-20}" \
  --patience "${PATIENCE:-5}" \
  --metric-ks 5,10,20 \
  --early-stopping-metric NDCG@20 \
  "${@:2}"
