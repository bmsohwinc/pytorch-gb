# Use Ubuntu 22.04 Machines
## install uv
```
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

## create project and activate
```
uv venv p3
source p3/bin/activate
```

## Install drivers
- this is done as part of the `install_all.sh` script from Songyu

## Get pytorch src
```
git clone https://github.com/pytorch/pytorch
cd pytorch
# if you are updating an existing checkout
git submodule sync
git submodule update --init --recursive
```

## Upgrade pip with uv (Optional but good for compatibility)
```
sudo apt install -y python3-pip
uv pip install --upgrade pip
```

## Install deps using uv
```
# Run this command from the PyTorch directory after cloning the source code using the “Get the PyTorch Source“ section above
uv pip install -r requirements.txt
```

## Install more deps and build tools (cmake)
```
uv pip install cmake ninja mkl-static mkl-include
# CUDA only: Add LAPACK support for the GPU if needed
# magma installation: The original script uses conda. If you need it, you might have to build from source or use conda.
# .ci/docker/common/install_magma_conda.sh 12.4

# (optional) If using torch.compile with inductor/triton, install the matching version of triton
# Run from the pytorch directory after cloning
# For Intel GPU support, please explicitly `export USE_XPU=1` before running command.
make triton
```

## Install pytorch
```
export CMAKE_PREFIX_PATH="${VIRTUAL_ENV}:${CMAKE_PREFIX_PATH}"
uv pip install --no-build-isolation -v -e .

```

## Run torchrun DDP test
```
# In worker,
# Tag node0's IP address as IP1 in /etc/hosts. Else torchrun fails to recognize

export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=enp94s0f0np0  # you get this name through ifconfig or ip addr commands

# master
torchrun --nproc-per-node=1          --nnodes=2          --node-rank=0          --rdzv-id=123          --rdzv-backend=c10d          --rdzv-endpoint='localhost:29500'          ddp.py 50 10

# worker 
torchrun --nproc-per-node=1          --nnodes=2          --node-rank=1          --rdzv-id=123          --rdzv-backend=c10d          --rdzv-endpoint='IP1:29500'          ddp.py 50 10
```

## additional options if above does not run at first
- fixed the `Unable to connect to c10d` error using https://github.com/pytorch/pytorch/issues/67547
```
unset http_proxy
unset https_proxy
unset HTTP_PROXY
unset HTTPS_PROXY

export MASTER_ADDR=10.10.1.1
export MASTER_PORT=29500

export GLOO_SOCKET_IFNAME=enp94s0f0np0

sudo ufw allow 29500/tcp
sudo iptables -A INPUT -p tcp --dport 29500 -j ACCEPT

```


## Experiment plan
- Measure DDP time
- For different gradient (or parameter) counts
- With and without gradient_as_bucket_view flag set in the DDP() constructor
- With 20 epochs for each setting
- Final csv data file (on both master and worker):
    - Node id (rank)
    - Num params
    - Grad as bucket (1/0)
    - Epoch number
    - Timestamp before forward pass
    - Timestamp before backward pass
    - Timestamp before optimizer pass
    - Timestamp after optimizer pass
    - Forward time in seconds
    - Backward time in seconds
    - Optimizer time in seconds
    - Total time in seconds



## scp commands
```
# send code from local to remote
scp -r ddp-example bmsb235@c240g5-110217.wisc.cloudlab.us:/users/bmsb235/

# get files from remote to local
scp -r bmsb235@c240g5-110217.wisc.cloudlab.us:/users/bmsb235/ddp-example/data/20260216_143615 ./ddp-a/data/
```

## timer code
```cpp
auto start = std::chrono::steady_clock::now();
initialize_bucket_views(bucket);
auto end = std::chrono::steady_clock::now();
copy_times_us_.push_back(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
```


## Running ddp
```
bash run.sh 0 10 20   # on node 0
bash run.sh 1 10 20   # on node 1

```