#!/bin/bash

# Run as:
# bash run.sh <node_id> <start_exponent> <end_exponent>
# node_id = 0 for master, 1 for worker, and so on
# start_exponent = 1, 2, ... is starting parameter size treated as 2^start
# end_exponent = 1, 2, ... is ending parameter size treated as 2^end
#
# Define the experiment range (2^1 to 2^26)
# 2^26 is ~67 million. 2^27 would exceed 10^8.

NODE_RANK=$1  # Pass 0 for master, 1 for worker
EXPO_START=$2
EXPO_END=$3

# Configuration
MASTER_IP="IP1" # Replace with your Master's IP
MASTER_PORT="29500"
NNODES=2
EPOCHS=2
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

        # Cleanup to ensure a clean start
        rm -f snapshot.pt

        gb_val=0
        if [[ "$bucket" == "--grad_as_bucket_view" ]]; then gb_val=1; fi
        mkdir -p ./data/${RUN_ID}
        LOG_FILE="./data/${RUN_ID}/stdout-node-${NODE_RANK}-gb-${gb_val}-param-${params}.log"

        echo "------------------------------------------------"
        echo "RANK $NODE_RANK: Running 2^$i ($params params), GradBucket=$gb_val, Log=$LOG_FILE"
        echo "------------------------------------------------"


        torchrun --nproc-per-node=1 \
                 --nnodes=$NNODES \
                 --node-rank=$NODE_RANK \
                 --rdzv-id=123 \
                 --rdzv-backend=c10d \
                 --rdzv-endpoint=$ENDPOINT \
                 ddp.py $EPOCHS \
                 --num_params $params \
                 $bucket \
                 --run_id "$RUN_ID" 2>&1 | tee "$LOG_FILE"

        # Short sleep to allow sockets to clear
        sleep 2
    done
done
