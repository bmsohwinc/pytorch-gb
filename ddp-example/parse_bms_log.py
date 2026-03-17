#!/usr/bin/env python3
"""
Parse DDP reducer timing lines emitted as:
  bms#: DDP_BACKWARD: fwd_copy=<us> rev_copy=<us> init_views=<us>

The sweep benchmark runs gb=0 first and gb=1 second in the same process, so
the per-iteration bms# lines split evenly into two phases. For steady-state
copy metrics, we skip warmup iterations inside each phase. For init_views, we
keep the full-phase total because bucket/view setup is itself a startup cost.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path


LINE_RE = re.compile(
    r"bms#: DDP_BACKWARD: "
    r"fwd_copy=(?P<fwd_copy>\d+)us "
    r"rev_copy=(?P<rev_copy>\d+)us "
    r"init_views=(?P<init_views>\d+)us"
)


@dataclass(frozen=True)
class BmsRecord:
    fwd_copy: int
    rev_copy: int
    init_views: int


def parse_bms_records(log_file: Path) -> list[BmsRecord]:
    records = []
    for line in log_file.read_text().splitlines():
        match = LINE_RE.search(line)
        if not match:
            continue
        records.append(
            BmsRecord(
                fwd_copy=int(match.group("fwd_copy")),
                rev_copy=int(match.group("rev_copy")),
                init_views=int(match.group("init_views")),
            )
        )
    return records


def split_phases(records: list[BmsRecord]) -> tuple[list[BmsRecord], list[BmsRecord]]:
    if len(records) % 2 != 0:
        raise ValueError(
            f"expected an even number of bms# lines, but found {len(records)}"
        )
    half = len(records) // 2
    return records[:half], records[half:]


def steady_state_records(
    records: list[BmsRecord], warmup_lines: int, iter_lines: int | None
) -> list[BmsRecord]:
    if warmup_lines < 0:
        raise ValueError("warmup_lines must be non-negative")
    start = min(warmup_lines, len(records))
    if iter_lines is None:
        return records[start:]
    if iter_lines < 0:
        raise ValueError("iter_lines must be non-negative")
    return records[start : start + iter_lines]


def mean_attr(records: list[BmsRecord], attr: str) -> int:
    if not records:
        return 0
    total = sum(getattr(record, attr) for record in records)
    return round(total / len(records))


def total_attr(records: list[BmsRecord], attr: str) -> int:
    return sum(getattr(record, attr) for record in records)


def improve_pct(baseline: int, candidate: int) -> str:
    if baseline <= 0:
        return "0"
    return f"{((baseline - candidate) / baseline) * 100:.1f}"


def shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def build_shell_vars(
    records: list[BmsRecord], warmup: int, iters: int | None, world_size: int
) -> dict[str, str]:
    if world_size <= 0:
        raise ValueError("world_size must be positive")

    if not records:
        return {
            "TOTAL_BMS": "0",
            "WORLD_SIZE": str(world_size),
            "PHASE_LEN": "0",
            "PHASE_ITERS": "0",
            "STEADY_LEN": "0",
            "STEADY_ITERS": "0",
            "FWD_COPY_GB0_MEAN": "0",
            "FWD_COPY_GB1_MEAN": "0",
            "FWD_COPY_IMPROVE_PCT": "0",
            "REV_COPY_GB0_MEAN": "0",
            "REV_COPY_GB1_MEAN": "0",
            "REV_COPY_IMPROVE_PCT": "0",
            "INIT_VIEWS_GB0_TOTAL": "0",
            "INIT_VIEWS_GB1_TOTAL": "0",
        }

    gb0_records, gb1_records = split_phases(records)
    warmup_lines = warmup * world_size
    iter_lines = None if iters is None else iters * world_size
    gb0_steady = steady_state_records(
        gb0_records, warmup_lines=warmup_lines, iter_lines=iter_lines
    )
    gb1_steady = steady_state_records(
        gb1_records, warmup_lines=warmup_lines, iter_lines=iter_lines
    )

    fwd0 = mean_attr(gb0_steady, "fwd_copy")
    fwd1 = mean_attr(gb1_steady, "fwd_copy")
    rev0 = mean_attr(gb0_steady, "rev_copy")
    rev1 = mean_attr(gb1_steady, "rev_copy")
    init0 = total_attr(gb0_records, "init_views")
    init1 = total_attr(gb1_records, "init_views")

    return {
        "TOTAL_BMS": str(len(records)),
        "WORLD_SIZE": str(world_size),
        "PHASE_LEN": str(len(gb0_records)),
        "PHASE_ITERS": str(len(gb0_records) // world_size),
        "STEADY_LEN": str(min(len(gb0_steady), len(gb1_steady))),
        "STEADY_ITERS": str(min(len(gb0_steady), len(gb1_steady)) // world_size),
        "FWD_COPY_GB0_MEAN": str(fwd0),
        "FWD_COPY_GB1_MEAN": str(fwd1),
        "FWD_COPY_IMPROVE_PCT": improve_pct(fwd0, fwd1),
        "REV_COPY_GB0_MEAN": str(rev0),
        "REV_COPY_GB1_MEAN": str(rev1),
        "REV_COPY_IMPROVE_PCT": improve_pct(rev0, rev1),
        "INIT_VIEWS_GB0_TOTAL": str(init0),
        "INIT_VIEWS_GB1_TOTAL": str(init1),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Parse DDP reducer bms# logs")
    parser.add_argument("--log-file", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--iters", type=int, default=None)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument(
        "--format", choices=("shell",), default="shell", help="output format"
    )
    args = parser.parse_args()

    records = parse_bms_records(args.log_file)
    shell_vars = build_shell_vars(
        records,
        warmup=args.warmup,
        iters=args.iters,
        world_size=args.world_size,
    )
    for key, value in shell_vars.items():
        print(f"{key}={shell_quote(value)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
