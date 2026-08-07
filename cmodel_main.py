#!/usr/bin/env python3
"""Command-line entry point for the strict C-model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.cmodel import (
    AttentionMode,
    AttentionOp,
    Conv2dOp,
    GemmOp,
    LayerNormOp,
    SoftmaxOp,
    StrictCModel,
    load_cmodel_config,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    add_operator_subparsers(parser)
    return parser


def add_operator_subparsers(parser: argparse.ArgumentParser) -> None:
    subparsers = parser.add_subparsers(dest="operator", required=True)

    gemm = subparsers.add_parser("gemm")
    _id(gemm)
    for name in ("m", "n", "k", "input_bits", "weight_bits", "output_bits", "accumulator_bits"):
        gemm.add_argument(f"--{name.replace('_', '-')}", dest=name, type=int, required=True)

    conv = subparsers.add_parser("conv")
    _id(conv)
    for name in (
        "batch", "in_channels", "out_channels", "input_h", "input_w", "kernel_h", "kernel_w",
        "stride_h", "stride_w", "pad_h", "pad_w", "dilation_h", "dilation_w", "groups",
        "input_bits", "weight_bits", "output_bits", "accumulator_bits",
    ):
        conv.add_argument(f"--{name.replace('_', '-')}", dest=name, type=int, required=True)

    softmax = subparsers.add_parser("softmax")
    _id(softmax)
    for name in ("rows", "cols", "element_bits"):
        softmax.add_argument(f"--{name.replace('_', '-')}", dest=name, type=int, required=True)

    layernorm = subparsers.add_parser("layernorm")
    _id(layernorm)
    for name in ("rows", "cols", "element_bits", "parameter_bits"):
        layernorm.add_argument(f"--{name.replace('_', '-')}", dest=name, type=int, required=True)
    layernorm.add_argument("--epsilon", type=float, required=True)

    attention = subparsers.add_parser("attention")
    _id(attention)
    for name in (
        "batch", "query_tokens", "kv_tokens", "model_dim", "heads", "head_dim",
        "input_bits", "weight_bits", "activation_bits", "accumulator_bits",
    ):
        attention.add_argument(f"--{name.replace('_', '-')}", dest=name, type=int, required=True)
    attention.add_argument("--mode", choices=("prefill", "decode"), required=True)
    attention.add_argument("--causal", choices=("true", "false"), required=True)


def _id(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--op-id", required=True)


def build_operation(args: argparse.Namespace):
    if args.operator == "gemm":
        return GemmOp(args.op_id, args.m, args.n, args.k, args.input_bits, args.weight_bits, args.output_bits, args.accumulator_bits)
    if args.operator == "conv":
        return Conv2dOp(
            args.op_id, args.batch, args.in_channels, args.out_channels, args.input_h, args.input_w,
            args.kernel_h, args.kernel_w, args.stride_h, args.stride_w, args.pad_h, args.pad_w,
            args.dilation_h, args.dilation_w, args.groups, args.input_bits, args.weight_bits,
            args.output_bits, args.accumulator_bits,
        )
    if args.operator == "softmax":
        return SoftmaxOp(args.op_id, args.rows, args.cols, args.element_bits)
    if args.operator == "layernorm":
        return LayerNormOp(args.op_id, args.rows, args.cols, args.element_bits, args.parameter_bits, args.epsilon)
    if args.operator == "attention":
        return AttentionOp(
            args.op_id, args.batch, args.query_tokens, args.kv_tokens, args.model_dim, args.heads, args.head_dim,
            AttentionMode(args.mode), args.causal == "true", args.input_bits, args.weight_bits,
            args.activation_bits, args.accumulator_bits,
        )
    raise ValueError(f"unsupported operator {args.operator}")


def main() -> None:
    args = build_parser().parse_args()
    config = load_cmodel_config(args.config)
    operation = build_operation(args)
    run = StrictCModel.from_config(config).simulate(operation)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(run.to_dict(config.architecture.frequency_hz), indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
