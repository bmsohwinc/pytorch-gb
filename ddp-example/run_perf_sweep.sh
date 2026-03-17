#!/bin/bash
# ──────────────────────────────────────────────────────────────
# Parameter Sweep: Gradient-as-Bucket-View Performance
# ──────────────────────────────────────────────────────────────
# Runs the performance test across multiple parameter sizes to
# see how gradient_as_bucket_view improvement scales.
#
# Usage:
#   bash run_perf_sweep.sh
#   bash run_perf_sweep.sh 20 26   # exponents from 2^20 to 2^26
#   NGPUS=2 bash run_perf_sweep.sh 20 26   # 2 GPUs on one machine
# ──────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
RUN_ID=$(date +"%Y%m%d_%H%M%S")
SWEEP_DIR="${SCRIPT_DIR}/data/sweep_${RUN_ID}"
mkdir -p "$SWEEP_DIR"

EXPO_START=${1:-15}   # 2^18 = 262K params
EXPO_END=${2:-28}     # 2^26 = 67M params

WARMUP=10
ITERS=50
NGPUS=${NGPUS:-1}

echo "=================================================="
echo "  Parameter Sweep: Gradient Bucket View"
echo "  Range: 2^${EXPO_START} to 2^${EXPO_END}"
echo "  GPUs: ${NGPUS}"
echo "  Output: ${SWEEP_DIR}"
echo "=================================================="

# CSV header for aggregated results
# CSV header for aggregated results
RESULTS_CSV="${SWEEP_DIR}/sweep_results.csv"
echo "num_params,expo,bwd_gb0_mean,bwd_gb0_std,bwd_gb1_mean,bwd_gb1_std,bwd_improve_pct,iter_gb0_mean,iter_gb0_std,iter_gb1_mean,iter_gb1_std,iter_improve_pct,fwd_copy_gb0_mean,fwd_copy_gb1_mean,fwd_copy_improve_pct,rev_copy_gb0_mean,rev_copy_gb1_mean,rev_copy_improve_pct,init_views_gb0_total,init_views_gb1_total" > "$RESULTS_CSV"

for (( e=$EXPO_START; e<=$EXPO_END; e++ )); do
    PARAMS=$((2**e))
    LOG_FILE="${SWEEP_DIR}/params_2e${e}_${PARAMS}.log"

    echo ""
    echo "──────────────────────────────────────────────"
    echo "  Running 2^${e} = ${PARAMS} parameters"
    echo "──────────────────────────────────────────────"

    python3 -m torch.distributed.run \
        --standalone \
        --nproc-per-node=$NGPUS \
        "${SCRIPT_DIR}/test_gb_perf.py" \
        --num_params $PARAMS \
        --warmup $WARMUP \
        --iters $ITERS \
        2>&1 | tee "$LOG_FILE"

    # Parse results from the log
    BWD_GB0=$(grep "gb=0 :" "$LOG_FILE" | head -1 | awk '{print $3}')
    BWD_GB0_STD=$(grep "gb=0 :" "$LOG_FILE" | head -1 | awk '{print $5}')
    BWD_GB1=$(grep "gb=1 :" "$LOG_FILE" | head -1 | awk '{print $3}')
    BWD_GB1_STD=$(grep "gb=1 :" "$LOG_FILE" | head -1 | awk '{print $5}')
    BWD_IMPROVE=$(grep "Improvement" "$LOG_FILE" | head -1 | awk '{print $3}' | tr -d '%+')

    ITER_GB0=$(grep "gb=0 :" "$LOG_FILE" | tail -1 | awk '{print $3}')
    ITER_GB0_STD=$(grep "gb=0 :" "$LOG_FILE" | tail -1 | awk '{print $5}')
    ITER_GB1=$(grep "gb=1 :" "$LOG_FILE" | tail -1 | awk '{print $3}')
    ITER_GB1_STD=$(grep "gb=1 :" "$LOG_FILE" | tail -1 | awk '{print $5}')
    ITER_IMPROVE=$(grep "Improvement" "$LOG_FILE" | tail -1 | awk '{print $3}' | tr -d '%+')

    eval "$(
        python3 "${SCRIPT_DIR}/parse_bms_log.py" \
            --log-file "$LOG_FILE" \
            --warmup "$WARMUP" \
            --iters "$ITERS" \
            --world-size "$NGPUS"
    )"

    FWD_COPY_GB0=${FWD_COPY_GB0_MEAN}
    FWD_COPY_GB1=${FWD_COPY_GB1_MEAN}
    FWD_COPY_IMPROVE=${FWD_COPY_IMPROVE_PCT}
    REV_COPY_GB0=${REV_COPY_GB0_MEAN}
    REV_COPY_GB1=${REV_COPY_GB1_MEAN}
    REV_COPY_IMPROVE=${REV_COPY_IMPROVE_PCT}
    INIT_VIEWS_GB0=${INIT_VIEWS_GB0_TOTAL}
    INIT_VIEWS_GB1=${INIT_VIEWS_GB1_TOTAL}

    echo "${PARAMS},${e},${BWD_GB0},${BWD_GB0_STD},${BWD_GB1},${BWD_GB1_STD},${BWD_IMPROVE},${ITER_GB0},${ITER_GB0_STD},${ITER_GB1},${ITER_GB1_STD},${ITER_IMPROVE},${FWD_COPY_GB0},${FWD_COPY_GB1},${FWD_COPY_IMPROVE},${REV_COPY_GB0},${REV_COPY_GB1},${REV_COPY_IMPROVE},${INIT_VIEWS_GB0},${INIT_VIEWS_GB1}" >> "$RESULTS_CSV"

    echo "  → Saved to $RESULTS_CSV"

    # Short pause between runs
    sleep 3
done

echo ""
echo "=================================================="
echo "  Sweep complete!"
echo "  Results CSV: ${RESULTS_CSV}"
echo "  Individual logs: ${SWEEP_DIR}/"
echo "=================================================="
echo ""
echo "To plot results, run:"
echo "  python3 ${SCRIPT_DIR}/plot_sweep.py ${RESULTS_CSV}"
