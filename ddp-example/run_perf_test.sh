#!/bin/bash
# ──────────────────────────────────────────────────────────────
# Gradient-as-Bucket-View Performance Test
# ──────────────────────────────────────────────────────────────
# This runs test_gb_perf.py in standalone mode (single machine).
#
# Usage:
#   bash run_perf_test.sh [--num_params N] [--iters N] [--warmup N]
#
# Examples:
#   bash run_perf_test.sh                          # default 16M params
#   bash run_perf_test.sh --num_params 33554432     # 32M params
#   bash run_perf_test.sh --iters 100 --warmup 20   # more iterations
#   NGPUS=2 bash run_perf_test.sh --num_params 33554432
#
# The script pipes output to tee so you get both terminal display
# and a log file. The C++ "bms#:" lines are captured in the log.
# ──────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
RUN_ID=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="${SCRIPT_DIR}/data/perf_test_${RUN_ID}"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/output.log"

NGPUS=${NGPUS:-1}
WARMUP=10
ITERS=50
ARGS=("$@")

while (($#)); do
    case "$1" in
        --warmup)
            WARMUP="$2"
            shift 2
            ;;
        --warmup=*)
            WARMUP="${1#*=}"
            shift
            ;;
        --iters)
            ITERS="$2"
            shift 2
            ;;
        --iters=*)
            ITERS="${1#*=}"
            shift
            ;;
        *)
            shift
            ;;
    esac
done

echo "=================================================="
echo "  Gradient Bucket View - Performance Test"
echo "  GPUs: ${NGPUS}"
echo "  Log file: ${LOG_FILE}"
echo "=================================================="

torchrun \
    --standalone \
    --nproc-per-node=$NGPUS \
    "${SCRIPT_DIR}/test_gb_perf.py" "${ARGS[@]}" 2>&1 | tee "$LOG_FILE"

echo ""
echo "=================================================="
echo "  Test complete. Full log saved to:"
echo "  ${LOG_FILE}"
echo "=================================================="

# ── Parse and summarize C++ copy times from the log ──────────
echo ""
echo "── C++ Reducer Copy Time Summary (from bms# lines) ──"
echo ""

eval "$(
    python3 "${SCRIPT_DIR}/parse_bms_log.py" \
        --log-file "$LOG_FILE" \
        --warmup "$WARMUP" \
        --iters "$ITERS" \
        --world-size "$NGPUS"
)"

if [ "$TOTAL_BMS" -gt "0" ]; then
    echo "Total bms# lines: $TOTAL_BMS (${PHASE_ITERS} iterations per setting x ${WORLD_SIZE} ranks, ${STEADY_ITERS} steady-state iterations)"
    echo "  gb=0 avg forward copy time: ${FWD_COPY_GB0_MEAN} μs"
    echo "  gb=1 avg forward copy time: ${FWD_COPY_GB1_MEAN} μs"
    echo "  Forward-copy improvement: ${FWD_COPY_IMPROVE_PCT}%"
    echo ""
    echo "  gb=0 avg reverse copy time: ${REV_COPY_GB0_MEAN} μs"
    echo "  gb=1 avg reverse copy time: ${REV_COPY_GB1_MEAN} μs"
    echo "  Reverse-copy improvement: ${REV_COPY_IMPROVE_PCT}%"
    echo ""
    echo "  gb=0 total init-views time: ${INIT_VIEWS_GB0_TOTAL} μs"
    echo "  gb=1 total init-views time: ${INIT_VIEWS_GB1_TOTAL} μs"
else
    echo "  (No bms# lines found. Is USE_CUDA defined in the build?)"
fi

echo ""
