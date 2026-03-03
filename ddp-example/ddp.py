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


import time, threading
import psutil
from pynvml import (
    nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetUtilizationRates,
    nvmlDeviceGetMemoryInfo
)

import csv

def write_util_csv(samples, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if not samples:
        return
    keys = list(samples[0].keys())
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(samples)

class UtilSampler:
    def __init__(self, gpu_index=0, interval_ms=1):
        self.gpu_index = gpu_index
        self.interval_ns = int((interval_ms / 1000.0) * 1e9)
        self.stop_evt = threading.Event()
        self.samples = []
        self.proc = psutil.Process()

        nvmlInit()
        self.h = nvmlDeviceGetHandleByIndex(gpu_index)
        self.num_cores = psutil.cpu_count(logical=True)

    def _snap(self, t_ns):
        # CPU
        cpu_per_core = psutil.cpu_percent(interval=None, percpu=True)
        if cpu_per_core:
            cpu_agg = sum(cpu_per_core) / len(cpu_per_core)
        else:
            cpu_agg = 0.0

        vm = psutil.virtual_memory()
        rss = self.proc.memory_info().rss

        # GPU
        util = nvmlDeviceGetUtilizationRates(self.h)
        mem = nvmlDeviceGetMemoryInfo(self.h)

        # PyTorch allocator stats (process-local, for this GPU)
        try:
            alloc = torch.cuda.memory_allocated(self.gpu_index)
            reserv = torch.cuda.memory_reserved(self.gpu_index)
            max_alloc = torch.cuda.max_memory_allocated(self.gpu_index)
        except Exception:
            alloc = reserv = max_alloc = -1

        sample = {
            "t_ns": t_ns,
            "cpu_pct_agg": cpu_agg,
        }
        for i, pct in enumerate(cpu_per_core):
            sample[f"cpu_core_{i}_pct"] = pct

        sample.update({
            "ram_used_bytes": vm.used,
            "ram_avail_bytes": vm.available,
            "proc_rss_bytes": rss,
            "gpu_util_pct": util.gpu,
            "gpu_mem_util_pct": util.memory,
            "vram_used_bytes": mem.used,
            "vram_total_bytes": mem.total,
            "torch_alloc_bytes": alloc,
            "torch_reserved_bytes": reserv,
            "torch_max_alloc_bytes": max_alloc,
        })
        self.samples.append(sample)

    def start(self):
        # Warm-up for psutil.cpu_percent
        psutil.cpu_percent(interval=None, percpu=True)

        def run():
            next_t = time.perf_counter_ns()
            while not self.stop_evt.is_set():
                now = time.perf_counter_ns()
                if now >= next_t:
                    self._snap(now)
                    next_t += self.interval_ns
                else:
                    # Busy wait for guaranteed <1ms precision
                    pass

        self.th = threading.Thread(target=run, daemon=True)
        self.th.start()

    def stop(self):
        self.stop_evt.set()
        self.th.join()


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
        schedule=schedule(wait=2, warmup=2, active=1), #, repeat=1),
        on_trace_ready=tensorboard_trace_handler(tb_log_dir),
        record_shapes=True,
        profile_memory=True,
        with_stack=False,   # set True if you want call stacks (slower)
    )

    return prof


class Trainer:
    def __init__(self, model, train_data, optimizer, grad_bucket, num_layers, profiler, util_trace_step=-1, util_interval_ms=10, run_id="default"):
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.global_rank = int(os.environ["RANK"])
        self.model = model.to(self.local_rank)
        self.train_data = train_data
        self.optimizer = optimizer
        self.grad_bucket = grad_bucket
        self.num_layers = num_layers
        self.logs = []
        self.profiler = profiler
        self.util_trace_step = util_trace_step
        self.util_interval_ms = util_interval_ms
        self.run_id = run_id
        self.util_dumped = False
        self.global_step = 0

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

            sampler = None
            if (not self.util_dumped) and (self.util_trace_step >= 0) and (self.global_step == self.util_trace_step):
                sampler = UtilSampler(gpu_index=self.local_rank, interval_ms=self.util_interval_ms)
                sampler.start()
                util_t0_ns = time.perf_counter_ns()

            #torch.cuda.synchronize()
            t0 = time.perf_counter()
            ts_fwd = time.time()

            output = self.model(source)
            loss = F.mse_loss(output, targets)

            #torch.cuda.synchronize()
            t1 = time.perf_counter()
            ts_bwd = time.time()

            self.optimizer.zero_grad()
            loss.backward()

            #torch.cuda.synchronize()
            t2 = time.perf_counter()
            ts_opt = time.time()

            self.optimizer.step()
            #torch.cuda.synchronize()
            t3 = time.perf_counter()
            ts_after = time.time()

            if sampler is not None:
                util_t1_ns = time.perf_counter_ns()
                sampler.stop()

                gb_val = 1 if self.grad_bucket else 0
                out_dir = os.path.join("data", self.run_id)
                out_path = os.path.join(
                    out_dir,
                    f"util-node-{dist.get_rank()}-gb-{gb_val}-layers-{self.num_layers}-data-{len(self.train_data.dataset)}-step-{self.global_step}.csv"
                )

                # Add step window metadata into each sample (helps align later)
                for s in sampler.samples:
                    s["step"] = self.global_step
                    s["step_t0_ns"] = util_t0_ns
                    s["step_t1_ns"] = util_t1_ns

                write_util_csv(sampler.samples, out_path)
                if dist.get_rank() == 0:
                    print(f"[util] wrote {len(sampler.samples)} samples to {out_path}")

                self.util_dumped = True

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
            self.global_step += 1

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
        # Explicitly flush the caching allocator and reset peak stats 
        # to ensure a clean slate between sequential torchrun executions.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
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
    parser.add_argument("--util_trace_step", type=int, default=-1)
    parser.add_argument("--util_interval_ms", type=int, default=1)
    parser.add_argument("--run_id", type=str, required=True)
    args = parser.parse_args()

    init_process_group(backend="nccl")

    model, num_layers, input_dim = build_model(args.num_params)
    prof = build_profiler(args.run_id, num_layers, args.grad_as_bucket_view, args.data_size)
    dataset = MyTrainDataset(args.data_size, input_dim)
    train_data = DataLoader(dataset, batch_size=100, sampler=DistributedSampler(dataset))
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    trainer = Trainer(
        model, train_data, optimizer, args.grad_as_bucket_view, num_layers, prof,
        util_trace_step=args.util_trace_step,
        util_interval_ms=args.util_interval_ms,
        run_id=args.run_id,
    )

    print(f"Starting DDP on device: {int(os.environ['LOCAL_RANK'])}")
    trainer.train(args.epochs)
    trainer.save_logs(args.num_params, args.run_id, args.data_size)
    destroy_process_group()


if __name__ == "__main__":
    main()
