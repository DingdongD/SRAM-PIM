#!/usr/bin/env python3
"""Run a strict apples-to-apples architecture comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.cmodel import GemmOp, StrictCModel, load_cmodel_config
from src.cmodel.errors import ConfigurationError


def _compute_signature(config) -> tuple[int, int, str, str]:
    arrays = config.architecture.arrays
    return arrays.rows, arrays.cols, arrays.dataflow, config.backend.kind


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare centralized and bank-local stacked-SRAM NPU C-models"
    )
    parser.add_argument("--centralized-config", required=True)
    parser.add_argument("--bank-local-config", required=True)
    parser.add_argument("--M", type=int, required=True)
    parser.add_argument("--N", type=int, required=True)
    parser.add_argument("--K", type=int, required=True)
    parser.add_argument("--output")
    args = parser.parse_args()

    central_config = load_cmodel_config(args.centralized_config)
    local_config = load_cmodel_config(args.bank_local_config)
    if _compute_signature(central_config) != _compute_signature(local_config):
        raise ConfigurationError(
            "strict comparison requires identical array rows/cols/dataflow and "
            "the same backend kind"
        )

    op = GemmOp("compare_gemm", args.M, args.N, args.K)
    central_run = StrictCModel(central_config).simulate_gemm(op)
    local_run = StrictCModel(local_config).simulate_gemm(op)
    central_report = central_run.to_dict(central_config.architecture.frequency_hz)
    local_report = local_run.to_dict(local_config.architecture.frequency_hz)

    central_manifests = central_report["model"]["backend_manifests"]
    local_manifests = local_report["model"]["backend_manifests"]
    central_versions = {(item["name"], item["version"]) for item in central_manifests}
    local_versions = {(item["name"], item["version"]) for item in local_manifests}
    if central_versions != local_versions:
        raise ConfigurationError(
            "external backend identity/version differs between compared runs: "
            f"centralized={sorted(central_versions)}, bank_local={sorted(local_versions)}"
        )

    central_cycles = central_report["latency"]["total_cycles"]
    local_cycles = local_report["latency"]["total_cycles"]
    result = {
        "operation": central_report["operation"],
        "fairness_checks": {
            "compute_signature": _compute_signature(central_config),
            "backend_manifests": central_manifests,
        },
        "centralized": {
            "latency": central_report["latency"],
            "diagnosis": central_report["diagnosis"],
            "model": central_report["model"],
            "resource_busy_cycles": central_report["resource_busy_cycles"],
        },
        "bank_local": {
            "latency": local_report["latency"],
            "diagnosis": local_report["diagnosis"],
            "model": local_report["model"],
            "resource_busy_cycles": local_report["resource_busy_cycles"],
        },
        "comparison": {
            "centralized_cycles_div_bank_local_cycles": (
                central_cycles / local_cycles if local_cycles else None
            ),
            "cycle_delta_bank_local_minus_centralized": local_cycles - central_cycles,
        },
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
