#!/bin/bash
set -euo pipefail

# Usage:
#   Multi-node:
#     bash run.sh <node_rank> [mode] [ngpus]
#
#   Examples:
#     bash run.sh 0 multi 1
#     bash run.sh 1 multi 1
#
#   Standalone:
#     bash run.sh 0 standalone 1

NODE_RANK=${1:?need node rank}
MODE=${2:-multi}
NGPUS=${3:-1}

MASTER_IP="IP1"
MASTER_PORT=29500
NNODES=2

# Training config
MODEL="resnet18"
EPOCHS=10
BATCH_SIZE=128
NUM_WORKERS=4
LR=0.1
DATA_DIR="./datasets"
UTIL_INTERVAL_MS=5
RUN_ID=$(date +"%Y%m%d_%H%M%S")


# Determine endpoint based on rank
if [ "$NODE_RANK" -eq 0 ]; then
    ENDPOINT="localhost:$MASTER_PORT"
else
    ENDPOINT="$MASTER_IP:$MASTER_PORT"
fi


mkdir -p "./data/${RUN_ID}"

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET,ENV
export NCCL_SOCKET_IFNAME=eno33np0
export NCCL_SOCKET_FAMILY=AF_INET
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5_0
export NCCL_NET_GDR_LEVEL=5

# Optional stability helpers
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTHONUNBUFFERED=1

run_case () {
    local GB_FLAG="$1"
    local GB_VAL="$2"

    local LOG_FILE="./data/${RUN_ID}/stdout-node-${NODE_RANK}-gb-${GB_VAL}.log"

    echo "============================================================"
    echo "NODE_RANK=${NODE_RANK} RUN_ID=${RUN_ID}"
    echo "MODEL=${MODEL} EPOCHS=${EPOCHS} BATCH_SIZE=${BATCH_SIZE}"
    echo "grad_as_bucket_view=${GB_VAL}"
    echo "LOG_FILE=${LOG_FILE}"
    echo "============================================================"

    rm -f snapshot.pt

    if [ "$MODE" == "standalone" ]; then
        torchrun \
            --standalone \
            --nproc-per-node="${NGPUS}" \
            ddp.py \
            --data_dir "${DATA_DIR}" \
            --run_id "${RUN_ID}" \
            --model "${MODEL}" \
            --epochs "${EPOCHS}" \
            --batch_size "${BATCH_SIZE}" \
            --num_workers "${NUM_WORKERS}" \
            --lr "${LR}" \
            --util_interval_ms "${UTIL_INTERVAL_MS}" \
            --trace_all_epochs \
            ${GB_FLAG} \
            2>&1 | tee "${LOG_FILE}"
    else
        torchrun \
            --nproc-per-node="${NGPUS}" \
            --nnodes="${NNODES}" \
            --node-rank="${NODE_RANK}" \
            --rdzv-id=123 \
            --rdzv-backend=c10d \
            --rdzv-endpoint=$ENDPOINT \
            ddp.py \
            --data_dir "${DATA_DIR}" \
            --run_id "${RUN_ID}" \
            --model "${MODEL}" \
            --epochs "${EPOCHS}" \
            --batch_size "${BATCH_SIZE}" \
            --num_workers "${NUM_WORKERS}" \
            --lr "${LR}" \
            --util_interval_ms "${UTIL_INTERVAL_MS}" \
            --trace_all_epochs \
            ${GB_FLAG} \
            2>&1 | tee "${LOG_FILE}"
    fi

    sleep 5
}

# Case 1: gradient_as_bucket_view = False
run_case "" 0

# Case 2: gradient_as_bucket_view = True
run_case "--grad_as_bucket_view" 1