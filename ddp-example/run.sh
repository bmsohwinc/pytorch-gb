#!/bin/bash

# Run as:
# bash run.sh <node_id> <start_exponent> <end_exponent> <start_data_exponent> <end_data_exponent> [mode] [ngpus]
# node_id = 0 for master, 1 for worker, and so on
# start_exponent = 1, 2, ... is starting parameter size treated as 2^start
# end_exponent = 1, 2, ... is ending parameter size treated as 2^end
# start_data_exponent = 1, 2, ... is starting data size treated as 10^start
# end_data_exponent = 1, 2, ... is ending data size treated as 10^end
#
# Define the experiment range (2^1 to 2^26)
# 2^26 is ~67 million. 2^27 would exceed 10^8.

# Multi:
# bash run.sh 0 24 24 5 5 multi 1
# bash run.sh 1 24 24 5 5 multi 1

# Standalone:
# bash run.sh 0 24 24 5 5 standalone 4

NODE_RANK=$1  # Pass 0 for master, 1 for worker
EXPO_START=$2
EXPO_END=$3
DATA_START=$4
DATA_END=$5

MODE=${6:-"multi"}          # "multi" or "standalone"
NGPUS=${7:-1}               # number of GPUs per node (default 1)

UTIL_STEP=${8:-4}           # which global step to trace (default 6)
UTIL_INTERVAL_MS=${9:-1}    # sampling period in ms (default 1)

# Configuration
MASTER_IP="IP1" # Replace with your Master's IP
MASTER_PORT="29500"
NNODES=2
EPOCHS=1
RUN_ID=$(date +"%Y%m%d_%H%M%S") # Generate timestamp ONCE here

# Determine endpoint based on rank
if [ "$NODE_RANK" -eq 0 ]; then
    ENDPOINT="localhost:$MASTER_PORT"
else
    ENDPOINT="$MASTER_IP:$MASTER_PORT"
fi


# Enable verbose initialization and network logs
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET
export NCCL_SOCKET_IFNAME=enp94s0f0np0  # you get this name through ifconfig or ip addr commands

# Force GDR even across the "SYS" (inter-socket) boundary
# Level 5 = Enable GDR regardless of topology distance
export NCCL_NET_GDR_LEVEL=5

# Optional: If using RoCE (as seen in your logs), ensure GID index is correct
# (though NCCL usually finds this automatically)
# export NCCL_IB_GID_INDEX=3


BUCKET_SETTINGS=("" "--grad_as_bucket_view")

for (( i=$EXPO_START; i<=$EXPO_END; i++ )); do
    # Calculate 2^i
    params=$((2**i))

    for bucket in "${BUCKET_SETTINGS[@]}"; do

        for (( j=$DATA_START; j<=$DATA_END; j++ )); do
            # Calculate 10^j
            data_size=$((10**j))
            echo "Running with $params parameters and data size $data_size"

            # Cleanup to ensure a clean start
            rm -f snapshot.pt

            gb_val=0
            if [[ "$bucket" == "--grad_as_bucket_view" ]]; then gb_val=1; fi
            mkdir -p ./data/${RUN_ID}
            LOG_FILE="./data/${RUN_ID}/stdout-node-${NODE_RANK}-gb-${gb_val}-param-${params}-data-${data_size}.log"

            echo "------------------------------------------------"
            echo "RANK $NODE_RANK: Running 2^$i ($params params), GradBucket=$gb_val, Log=$LOG_FILE, DataSize=$data_size"
            echo "------------------------------------------------"


            if [ "$MODE" == "standalone" ]; then
                # Start background nvidia-smi with 1ms looping interval
                NVSMI_LOG="./data/${RUN_ID}/nvsmi-node-${NODE_RANK}-gb-${gb_val}-param-${params}-data-${data_size}.csv"
                nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv -lms 1 > "$NVSMI_LOG" &
                NVSMI_PID=$!

                torchrun \
                    --standalone \
                    --nproc-per-node=$NGPUS \
                    ddp.py $EPOCHS \
                    --num_params $params \
                    $bucket \
                    --data_size $data_size \
                    --util_trace_step $UTIL_STEP \
                    --util_interval_ms $UTIL_INTERVAL_MS \
                    --run_id "$RUN_ID" 2>&1 | tee "$LOG_FILE"
                
                # Stop the background nvidia-smi poll once torchrun is done
                kill $NVSMI_PID 2>/dev/null
            else
                # Start background nvidia-smi with 1ms looping interval
                NVSMI_LOG="./data/${RUN_ID}/nvsmi-node-${NODE_RANK}-gb-${gb_val}-param-${params}-data-${data_size}.csv"
                nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv -lms 1 > "$NVSMI_LOG" &
                NVSMI_PID=$!

                torchrun \
                    --nproc-per-node=$NGPUS \
                    --nnodes=$NNODES \
                    --node-rank=$NODE_RANK \
                    --rdzv-id=123 \
                    --rdzv-backend=c10d \
                    --rdzv-endpoint=$ENDPOINT \
                    ddp.py $EPOCHS \
                    --num_params $params \
                    $bucket \
                    --data_size $data_size \
                    --util_trace_step $UTIL_STEP \
                    --util_interval_ms $UTIL_INTERVAL_MS \
                    --run_id "$RUN_ID" 2>&1 | tee "$LOG_FILE"

                # Stop the background nvidia-smi poll once torchrun is done
                kill $NVSMI_PID 2>/dev/null
            fi

            # Short sleep to allow sockets to clear
            sleep 5
        done
    done
done
