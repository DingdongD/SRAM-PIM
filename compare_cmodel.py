#!/usr/bin/env python3
"""Apples-to-apples comparison between centralized and bank-local placement."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from cmodel_main import add_operator_subparsers, build_operation
from src.cmodel import StrictCModel, load_cmodel_config
from src.cmodel.errors import ConfigurationError


def comparison_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--centralized-config", required=True)
    parser.add_argument("--bank-local-config", required=True)
    parser.add_argument("--output", required=True)
    add_operator_subparsers(parser)
    return parser


def _assert_same_model(left, right) -> None:
    if left.mapping != right.mapping:
        raise ConfigurationError("comparison requires identical mapping")
    if left.addresses != right.addresses:
        raise ConfigurationError("comparison requires identical address map")
    if left.backends != right.backends:
        raise ConfigurationError("comparison requires identical external backends")
    if left.simulation != right.simulation:
        raise ConfigurationError("comparison requires identical simulation limits")
    left_arch = asdict(left.architecture)
    right_arch = asdict(right.architecture)
    del left_arch["name"]
    del right_arch["name"]
    del left_arch["compute_placement"]
    del right_arch["compute_placement"]
    if left_arch != right_arch:
        raise ConfigurationError("comparison architectures may differ only in name and compute_placement")
    if left.architecture.compute_placement != "centralized_logic":
        raise ConfigurationError("centralized config must use centralized_logic")
    if right.architecture.compute_placement != "bank_local_logic":
        raise ConfigurationError("bank-local config must use bank_local_logic")


def main() -> None:
    args = comparison_parser().parse_args()
    centralized = load_cmodel_config(args.centralized_config)
    bank_local = load_cmodel_config(args.bank_local_config)
    _assert_same_model(centralized, bank_local)
    operation = build_operation(args)
    centralized_run = StrictCModel.from_config(centralized).simulate(operation)
    bank_local_run = StrictCModel.from_config(bank_local).simulate(operation)
    output = {
        "centralized": centralized_run.to_dict(centralized.architecture.frequency_hz),
        "bank_local": bank_local_run.to_dict(bank_local.architecture.frequency_hz),
    }
    output["comparison"] = {
        "cycle_ratio_bank_local_over_centralized": (
            bank_local_run.result.total_cycles / centralized_run.result.total_cycles
        )
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
