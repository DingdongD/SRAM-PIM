#!/usr/bin/env python3
"""Command-line entry point for the strict stacked-SRAM NPU C-model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.cmodel import GemmOp, StrictCModel, load_cmodel_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Strict cycle-level comparison of centralized SRAM-on-logic and "
            "bank-local near-SRAM systolic NPUs"
        )
    )
    parser.add_argument("--config", required=True, help="Strict C-model YAML")
    parser.add_argument("--M", type=int, required=True)
    parser.add_argument("--N", type=int, required=True)
    parser.add_argument("--K", type=int, required=True)
    parser.add_argument("--op-id", default="gemm0")
    parser.add_argument("--input-bits", type=int, default=8)
    parser.add_argument("--weight-bits", type=int, default=8)
    parser.add_argument("--accumulator-bits", type=int, default=32)
    parser.add_argument("--output", help="Optional JSON report path")
    parser.add_argument(
        "--include-events",
        action="store_true",
        help="Keep the full per-event schedule in stdout/output",
    )
    args = parser.parse_args()

    config = load_cmodel_config(args.config)
    op = GemmOp(
        op_id=args.op_id,
        m=args.M,
        n=args.N,
        k=args.K,
        input_bits=args.input_bits,
        weight_bits=args.weight_bits,
        accumulator_bits=args.accumulator_bits,
    )
    run = StrictCModel(config).simulate_gemm(op)
    report = run.to_dict(config.architecture.frequency_hz)
    if not args.include_events:
        report.pop("events", None)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
