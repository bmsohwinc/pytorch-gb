import argparse
import csv
import os
import random
import threading
import time

import psutil
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from pynvml import (
    nvmlInit,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetMemoryInfo,
    nvmlDeviceGetUtilizationRates,
)
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, models, transforms


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def write_csv(rows, out_path):
    if not rows:
        return
    ensure_dir(os.path.dirname(out_path))
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


class UtilSampler:
    """
    Lightweight sampler for process RAM + GPU VRAM + allocator stats.
    Enough to compare grad_as_bucket_view=False vs True.
    """
    def __init__(self, gpu_index=0, interval_ms=5):
        self.gpu_index = gpu_index
        self.interval_ns = int(interval_ms * 1e6)
        self.stop_evt = threading.Event()
        self.samples = []
        self.proc = psutil.Process()

        nvmlInit()
        self.h = nvmlDeviceGetHandleByIndex(gpu_index)

    def _snap(self, t_ns):
        cpu_per_core = psutil.cpu_percent(interval=None, percpu=True)
        cpu_agg = sum(cpu_per_core) / len(cpu_per_core) if cpu_per_core else 0.0

        vm = psutil.virtual_memory()
        rss = self.proc.memory_info().rss

        util = nvmlDeviceGetUtilizationRates(self.h)
        mem = nvmlDeviceGetMemoryInfo(self.h)

        try:
            alloc = torch.cuda.memory_allocated(self.gpu_index)
            reserved = torch.cuda.memory_reserved(self.gpu_index)
            max_alloc = torch.cuda.max_memory_allocated(self.gpu_index)
            max_reserved = torch.cuda.max_memory_reserved(self.gpu_index)
        except Exception:
            alloc = reserved = max_alloc = max_reserved = -1

        self.samples.append({
            "t_ns": t_ns,
            "cpu_pct_agg": cpu_agg,
            "ram_used_bytes": vm.used,
            "ram_avail_bytes": vm.available,
            "proc_rss_bytes": rss,
            "gpu_util_pct": util.gpu,
            "gpu_mem_util_pct": util.memory,
            "vram_used_bytes": mem.used,
            "vram_total_bytes": mem.total,
            "torch_alloc_bytes": alloc,
            "torch_reserved_bytes": reserved,
            "torch_max_alloc_bytes": max_alloc,
            "torch_max_reserved_bytes": max_reserved,
        })

    def start(self):
        psutil.cpu_percent(interval=None, percpu=True)

        def run():
            next_t = time.perf_counter_ns()
            while not self.stop_evt.is_set():
                now = time.perf_counter_ns()
                if now >= next_t:
                    self._snap(now)
                    next_t += self.interval_ns

        self.th = threading.Thread(target=run, daemon=True)
        self.th.start()

    def stop(self):
        self.stop_evt.set()
        self.th.join()


def setup_ddp():
    init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    return local_rank, global_rank, world_size


def cleanup_ddp():
    destroy_process_group()


def build_dataloaders(data_dir, batch_size, num_workers):
    train_tfms = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2023, 0.1994, 0.2010),
        ),
    ])

    test_tfms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2023, 0.1994, 0.2010),
        ),
    ])

    train_ds = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=train_tfms,
    )
    test_ds = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=test_tfms,
    )

    train_sampler = DistributedSampler(
        train_ds,
        shuffle=True,
        drop_last=False,
    )
    test_sampler = DistributedSampler(
        test_ds,
        shuffle=False,
        drop_last=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        sampler=test_sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    return train_loader, test_loader


def build_model(model_name: str):
    if model_name == "resnet18":
        return models.resnet18(num_classes=10)
    if model_name == "resnet34":
        return models.resnet34(num_classes=10)
    raise ValueError(f"Unsupported model: {model_name}")


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def reduce_scalar(value: float, device: torch.device):
    t = torch.tensor([value], dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t.item()


def train_one_epoch(model, loader, optimizer, criterion, device, epoch, rank):
    loader.sampler.set_epoch(epoch)
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    fwd_time = 0.0
    bwd_time = 0.0
    opt_time = 0.0

    epoch_start = time.perf_counter()

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        t0 = time.perf_counter()
        outputs = model(images)
        loss = criterion(outputs, targets)
        t1 = time.perf_counter()

        loss.backward()
        t2 = time.perf_counter()

        optimizer.step()
        t3 = time.perf_counter()

        fwd_time += (t1 - t0)
        bwd_time += (t2 - t1)
        opt_time += (t3 - t2)

        bs = images.size(0)
        running_loss += loss.item() * bs
        total += bs
        correct += (outputs.argmax(dim=1) == targets).sum().item()

    epoch_time = time.perf_counter() - epoch_start

    loss_sum = reduce_scalar(running_loss, device)
    correct_sum = reduce_scalar(correct, device)
    total_sum = reduce_scalar(total, device)
    fwd_sum = reduce_scalar(fwd_time, device)
    bwd_sum = reduce_scalar(bwd_time, device)
    opt_sum = reduce_scalar(opt_time, device)
    epoch_sum = reduce_scalar(epoch_time, device)

    world_size = dist.get_world_size()

    return {
        "train_loss": loss_sum / total_sum,
        "train_acc": correct_sum / total_sum,
        "train_samples": int(total_sum),
        "epoch_time_sec_avg_across_ranks": epoch_sum / world_size,
        "fwd_time_sec_sum_across_ranks": fwd_sum,
        "bwd_time_sec_sum_across_ranks": bwd_sum,
        "opt_time_sec_sum_across_ranks": opt_sum,
    }


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, targets)

        bs = images.size(0)
        running_loss += loss.item() * bs
        total += bs
        correct += (outputs.argmax(dim=1) == targets).sum().item()

    loss_sum = reduce_scalar(running_loss, device)
    correct_sum = reduce_scalar(correct, device)
    total_sum = reduce_scalar(total, device)

    return {
        "val_loss": loss_sum / total_sum,
        "val_acc": correct_sum / total_sum,
        "val_samples": int(total_sum),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--run_id", type=str, required=True)
    parser.add_argument("--model", type=str, default="resnet18", choices=["resnet18", "resnet34"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--grad_as_bucket_view", action="store_true")
    parser.add_argument("--util_interval_ms", type=int, default=5)
    parser.add_argument("--trace_all_epochs", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    local_rank, rank, world_size = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    train_loader, test_loader = build_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = build_model(args.model).to(device)
    num_params = count_params(model)

    ddp_model = DDP(
        model,
        device_ids=[local_rank],
        gradient_as_bucket_view=args.grad_as_bucket_view,
    )

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(
        ddp_model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[5, 8],
        gamma=0.1,
    )

    gb_val = 1 if args.grad_as_bucket_view else 0
    out_dir = os.path.join("data", args.run_id)
    ensure_dir(out_dir)

    epoch_rows = []
    util_sampler = None

    if args.trace_all_epochs:
        util_sampler = UtilSampler(gpu_index=local_rank, interval_ms=args.util_interval_ms)
        util_sampler.start()

    for epoch in range(args.epochs):
        train_stats = train_one_epoch(
            ddp_model, train_loader, optimizer, criterion, device, epoch, rank
        )
        val_stats = evaluate(
            ddp_model, test_loader, criterion, device
        )
        scheduler.step()

        if rank == 0:
            print(
                f"[gb={gb_val}] "
                f"[epoch {epoch+1}/{args.epochs}] "
                f"train_loss={train_stats['train_loss']:.4f} "
                f"train_acc={train_stats['train_acc']:.4f} "
                f"val_loss={val_stats['val_loss']:.4f} "
                f"val_acc={val_stats['val_acc']:.4f} "
                f"epoch_avg_rank_time={train_stats['epoch_time_sec_avg_across_ranks']:.2f}s"
            )

        row = {
            "rank": rank,
            "world_size": world_size,
            "epoch": epoch + 1,
            "model": args.model,
            "num_params": num_params,
            "grad_as_bucket_view": gb_val,
            "batch_size_per_gpu": args.batch_size,
            "train_loss": train_stats["train_loss"],
            "train_acc": train_stats["train_acc"],
            "val_loss": val_stats["val_loss"],
            "val_acc": val_stats["val_acc"],
            "train_samples": train_stats["train_samples"],
            "val_samples": val_stats["val_samples"],
            "epoch_time_sec_avg_across_ranks": train_stats["epoch_time_sec_avg_across_ranks"],
            "fwd_time_sec_sum_across_ranks": train_stats["fwd_time_sec_sum_across_ranks"],
            "bwd_time_sec_sum_across_ranks": train_stats["bwd_time_sec_sum_across_ranks"],
            "opt_time_sec_sum_across_ranks": train_stats["opt_time_sec_sum_across_ranks"],
            "torch_mem_alloc_bytes_end": torch.cuda.memory_allocated(device),
            "torch_mem_reserved_bytes_end": torch.cuda.memory_reserved(device),
            "torch_max_mem_alloc_bytes": torch.cuda.max_memory_allocated(device),
            "torch_max_mem_reserved_bytes": torch.cuda.max_memory_reserved(device),
        }
        epoch_rows.append(row)

    if util_sampler is not None:
        util_sampler.stop()
        util_path = os.path.join(
            out_dir,
            f"util-rank-{rank}-gb-{gb_val}.csv",
        )
        write_csv(util_sampler.samples, util_path)

    epoch_csv = os.path.join(
        out_dir,
        f"train-rank-{rank}-gb-{gb_val}.csv",
    )
    write_csv(epoch_rows, epoch_csv)

    if rank == 0:
        summary = {
            "rank": 0,
            "world_size": world_size,
            "model": args.model,
            "num_params": num_params,
            "epochs": args.epochs,
            "batch_size_per_gpu": args.batch_size,
            "global_batch_size": args.batch_size * world_size,
            "grad_as_bucket_view": gb_val,
            "final_train_loss": epoch_rows[-1]["train_loss"],
            "final_train_acc": epoch_rows[-1]["train_acc"],
            "final_val_loss": epoch_rows[-1]["val_loss"],
            "final_val_acc": epoch_rows[-1]["val_acc"],
            "peak_alloc_bytes": epoch_rows[-1]["torch_max_mem_alloc_bytes"],
            "peak_reserved_bytes": epoch_rows[-1]["torch_max_mem_reserved_bytes"],
        }
        write_csv(
            [summary],
            os.path.join(out_dir, f"summary-gb-{gb_val}.csv")
        )

    cleanup_ddp()


if __name__ == "__main__":
    main()
    