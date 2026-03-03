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

echo "=================================================="
echo "  Gradient Bucket View - Performance Test"
echo "  Log file: ${LOG_FILE}"
echo "=================================================="

torchrun \
    --standalone \
    --nproc-per-node=$NGPUS \
    "${SCRIPT_DIR}/test_gb_perf.py" "$@" 2>&1 | tee "$LOG_FILE"

echo ""
echo "=================================================="
echo "  Test complete. Full log saved to:"
echo "  ${LOG_FILE}"
echo "=================================================="

# ── Parse and summarize C++ copy times from the log ──────────
echo ""
echo "── C++ Reducer Copy Time Summary (from bms# lines) ──"
echo ""

# Count bms# lines - they alternate: first half from gb=0, second from gb=1
TOTAL_BMS=$(grep -c "bms#: DDP_BACKWARD: copy=" "$LOG_FILE" 2>/dev/null || echo 0)

if [ "$TOTAL_BMS" -gt "0" ]; then
    HALF=$((TOTAL_BMS / 2))
    echo "Total bms# lines: $TOTAL_BMS (${HALF} per setting)"

    # Extract copy times
    GB0_TIMES=$(grep "bms#: DDP_BACKWARD: copy=" "$LOG_FILE" | head -n $HALF | sed 's/.*copy=\([0-9]*\)us/\1/')
    GB1_TIMES=$(grep "bms#: DDP_BACKWARD: copy=" "$LOG_FILE" | tail -n $HALF | sed 's/.*copy=\([0-9]*\)us/\1/')

    # Calculate averages using awk
    GB0_AVG=$(echo "$GB0_TIMES" | awk '{sum+=$1; n++} END {if(n>0) printf "%.0f", sum/n; else print "N/A"}')
    GB1_AVG=$(echo "$GB1_TIMES" | awk '{sum+=$1; n++} END {if(n>0) printf "%.0f", sum/n; else print "N/A"}')

    echo "  gb=0 avg copy time: ${GB0_AVG} μs"
    echo "  gb=1 avg copy time: ${GB1_AVG} μs"

    if [ "$GB0_AVG" != "N/A" ] && [ "$GB1_AVG" != "N/A" ] && [ "$GB0_AVG" -gt 0 ]; then
        IMPROVE=$(awk "BEGIN {printf \"%.1f\", (($GB0_AVG - $GB1_AVG) / $GB0_AVG) * 100}")
        echo "  Improvement: ${IMPROVE}%"
    fi
else
    echo "  (No bms# lines found. Is USE_CUDA defined in the build?)"
fi

echo ""
