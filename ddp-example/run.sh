#!/bin/bash

# Run as:
# bash run.sh <node_id> <start_exponent> <end_exponent> <nproc_per_node> <standalone_flag> <data_size>
# node_id = 0 for master, 1 for worker, and so on
# start_exponent = 1, 2, ... is starting parameter size treated as 2^start
# end_exponent = 1, 2, ... is ending parameter size treated as 2^end
#
# Define the experiment range (2^1 to 2^26)
# 2^26 is ~67 million. 2^27 would exceed 10^8.

NODE_RANK=$1  # Pass 0 for master, 1 for worker
EXPO_START=$2
EXPO_END=$3

NPROC_PER_NODE=${4:-1}  # GPUs per node (single-node multi-GPU: set >1)
STANDALONE_FLAG=${5:-}  # pass --standalone as 5th arg to run single-node
DATA_SIZE=${6:-1024}  # dataset size (number of samples), default to 1024 if not provided

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
        LOG_FILE="./data/${RUN_ID}/stdout-node-${NODE_RANK}-gb-${gb_val}-param-${params}-data-${DATA_SIZE}.log"

        echo "------------------------------------------------"
        echo "RANK $NODE_RANK: Running 2^$i ($params params), GradBucket=$gb_val, Log=$LOG_FILE, DataSize=$DATA_SIZE"
        echo "------------------------------------------------"


        # Build torchrun args: multi-node (default) vs single-node (--standalone)
        if [[ "$STANDALONE_FLAG" == "--standalone" ]]; then
          TORCHRUN_ARGS=(--standalone --nproc-per-node="$NPROC_PER_NODE")
        else
          RDZV_ID="${RUN_ID}-gb-${gb_val}-param-${params}"
          TORCHRUN_ARGS=(--nproc-per-node="$NPROC_PER_NODE" --nnodes="$NNODES" --node-rank="$NODE_RANK" \
                         --rdzv-id="$RDZV_ID" --rdzv-backend=c10d --rdzv-endpoint="$ENDPOINT")
        fi

        torchrun "${TORCHRUN_ARGS[@]}" \
                 ddp.py $EPOCHS \
                 --num_params $params \
                 --data $DATA_SIZE \
                 $bucket \
                 --run_id "$RUN_ID" 2>&1 | tee "$LOG_FILE"

        # Short sleep to allow sockets to clear
        sleep 4

        # Move tracefile in /tmp to /data dir
        # summary: just find the latest non-empty profile_* file created after RUN_ID timestamp
        mapfile -t traces < <(
          find /tmp -maxdepth 1 -type f -name 'profile_*' -size +0c -printf '%f\n' 2>/dev/null \
          | awk -v rid="$RUN_ID" '
              /^profile_[0-9]{8}_[0-9]{6}/ {
                ts = substr($0, 9, 15)  # YYYYMMDD_HHMMSS
                if (ts > rid) print $0
              }' \
          | sort
        )

        if [ "${#traces[@]}" -gt 0 ]; then
          latest="${traces[-1]}"   # pick latest after sort
          src="/tmp/${latest}"
          dst="./data/${RUN_ID}/profile-node-${NODE_RANK}-gb-${gb_val}-param-${params}-data-${DATA_SIZE}.log"
          mv -f "$src" "$dst"
        else
          echo "WARNING: No matching profile_* trace found for RUN_ID=$RUN_ID" >&2
        fi
    done
done
