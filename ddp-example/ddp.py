import argparse
import datetime
import os
import time

import torch
import torch.nn.functional as F
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch.profiler import profile, ProfilerActivity, schedule, tensorboard_trace_handler
import torch.distributed as dist



class MyTrainDataset(Dataset):
    def __init__(self, size, input_dim):
        self.size = size
        self.input_dim = input_dim

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        # Generate on-the-fly to save System RAM
        return torch.rand(self.input_dim), torch.rand(1)


def build_model(target_params):
    """Dynamically builds an MLP to match a target parameter count."""
    input_dim = 128
    width = 512
    layers = []

    # Param count approx: (in * out) + out
    l1_params = (input_dim * width) + width

    if target_params <= l1_params:
        actual_width = max(1, target_params // (input_dim + 1))
        model = torch.nn.Linear(input_dim, actual_width)
        return (
            torch.nn.Sequential(model, torch.nn.Linear(actual_width, 1)),
            2,
            input_dim,
        )

    # Hidden layers and Output layer
    hidden_params = (width * width) + width
    out_params = (width * 1) + 1

    # Calculate how many hidden layers we can fit
    remaining = target_params - l1_params - out_params
    num_hidden = max(0, remaining // hidden_params)

    layers.append(torch.nn.Linear(input_dim, width))
    layers.append(torch.nn.ReLU())

    for _ in range(num_hidden):
        layers.append(torch.nn.Linear(width, width))
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Dropout(0.1))

    layers.append(torch.nn.Linear(width, 1))

    # Total "layer depth" count for logging
    depth = 2 + num_hidden
    return torch.nn.Sequential(*layers), depth, input_dim


def build_profiler(run_id, num_layers, grad_bucket, data_size):
    log_dir = os.path.join("data", run_id)
    os.makedirs(log_dir, exist_ok=True)
    gb_val = 1 if grad_bucket else 0
    tb_log_dir = os.path.join(
        log_dir, f"profile-node-{dist.get_rank()}-gb-{gb_val}-layers-{num_layers}-data-{data_size}"
    )

    prof = profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule(wait=2, warmup=2, active=6, repeat=1),
        on_trace_ready=tensorboard_trace_handler(tb_log_dir),
        record_shapes=True,
        profile_memory=True,
        with_stack=False,   # set True if you want call stacks (slower)
    )

    return prof


class Trainer:
    def __init__(self, model, train_data, optimizer, grad_bucket, num_layers, profiler):
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.global_rank = int(os.environ["RANK"])
        self.model = model.to(self.local_rank)
        self.train_data = train_data
        self.optimizer = optimizer
        self.grad_bucket = grad_bucket
        self.num_layers = num_layers
        self.logs = []
        self.profiler = profiler

        self.model = DDP(
            self.model,
            device_ids=[self.local_rank],
            gradient_as_bucket_view=self.grad_bucket,
        )

    def _run_epoch(self, epoch):
        rank = dist.get_rank() if dist.is_initialized() else 0

        epoch_logs = []

        self.train_data.sampler.set_epoch(epoch)
        for source, targets in self.train_data:
            source, targets = source.to(self.local_rank), targets.to(self.local_rank)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            ts_fwd = time.time()

            output = self.model(source)
            loss = F.mse_loss(output, targets)

            torch.cuda.synchronize()
            t1 = time.perf_counter()
            ts_bwd = time.time()

            self.optimizer.zero_grad()
            loss.backward()

            torch.cuda.synchronize()
            t2 = time.perf_counter()
            ts_opt = time.time()

            self.optimizer.step()
            torch.cuda.synchronize()
            t3 = time.perf_counter()
            ts_after = time.time()

            epoch_logs.append(
                {
                    "epoch": epoch,
                    "ts_fwd": ts_fwd,
                    "ts_bwd": ts_bwd,
                    "ts_opt": ts_opt,
                    "ts_after": ts_after,
                    "fwd_time": t1 - t0,
                    "bwd_time": t2 - t1,
                    "opt_time": t3 - t2,
                    "total": t3 - t0,
                }
            )

            self.profiler.step()  # Advance profiler to capture this iteration

        self.logs.append({
            "epoch": epoch,
            "ts_fwd": epoch_logs[0]["ts_fwd"],
            "ts_bwd": epoch_logs[0]["ts_bwd"],
            "ts_opt": epoch_logs[0]["ts_opt"],
            "ts_after": epoch_logs[0]["ts_after"],
            "fwd_time": sum(e["fwd_time"] for e in epoch_logs),
            "bwd_time": sum(e["bwd_time"] for e in epoch_logs),
            "opt_time": sum(e["opt_time"] for e in epoch_logs),
            "total": sum(e["total"] for e in epoch_logs),
        })

    def train(self, max_epochs):
        self.profiler.start()
        for epoch in range(max_epochs):
            self._run_epoch(epoch)
        self.profiler.stop()

    def save_logs(self, num_params, run_id, data_size):
        log_dir = os.path.join("data", run_id)
        os.makedirs(log_dir, exist_ok=True)
        gb_val = 1 if self.grad_bucket else 0
        filepath = os.path.join(
            log_dir, f"node-{self.global_rank}-gb-{gb_val}-param-{num_params}-data-{data_size}.csv"
        )

        headers = [
            "node_id",
            "num_params",
            "num_layers",
            "grad_bucket",
            "epoch",
            "ts_fwd",
            "ts_bwd",
            "ts_opt",
            "ts_after",
            "fwd_time",
            "bwd_time",
            "opt_time",
            "total_time",
        ]

        with open(filepath, "w") as f:
            f.write(",".join(headers) + "\n")
            for e in self.logs:
                f.write(
                    f"{self.global_rank},{num_params},{self.num_layers},{gb_val},{e['epoch']},"
                    f"{e['ts_fwd']},{e['ts_bwd']},{e['ts_opt']},{e['ts_after']},"
                    f"{e['fwd_time']},{e['bwd_time']},{e['opt_time']},{e['total']}\n"
                )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("epochs", type=int)
    parser.add_argument("--num_params", type=int, default=1000)
    parser.add_argument("--grad_as_bucket_view", action="store_true")
    parser.add_argument("--data_size", type=int, default=1000)
    parser.add_argument("--run_id", type=str, required=True)
    args = parser.parse_args()

    init_process_group(backend="nccl")

    model, num_layers, input_dim = build_model(args.num_params)
    prof = build_profiler(args.run_id, num_layers, args.grad_as_bucket_view, args.data_size)
    dataset = MyTrainDataset(args.data_size, input_dim)
    train_data = DataLoader(dataset, batch_size=100, sampler=DistributedSampler(dataset))
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    trainer = Trainer(
        model, train_data, optimizer, args.grad_as_bucket_view, num_layers, prof
    )

    print(f"Starting DDP on device: {int(os.environ['LOCAL_RANK'])}")
    trainer.train(args.epochs)
    trainer.save_logs(args.num_params, args.run_id, args.data_size)
    destroy_process_group()


if __name__ == "__main__":
    main()
