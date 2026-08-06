from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.cmodel.backends.recorded import RecordedSystolicBackend
from src.cmodel.config import load_cmodel_config
from src.cmodel.errors import BackendOutputError, ConfigurationError
from src.cmodel.ir import SystolicInvocation


def test_recorded_backend_requires_exact_invocation(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    path.write_text(
        json.dumps(
            {
                "backend_name": "SCALE-Sim",
                "backend_version": "pinned",
                "invocation": {
                    "op_id": "g",
                    "m": 4,
                    "n": 4,
                    "k": 4,
                    "array_rows": 8,
                    "array_cols": 8,
                    "dataflow": "ws",
                    "input_bits": 8,
                    "weight_bits": 8,
                    "accumulator_bits": 32,
                },
                "total_cycles": 4,
                "utilization": 0.5,
                "demands": [
                    {
                        "cycle": 0,
                        "operand": "input",
                        "address": 0,
                        "nbytes": 1,
                    },
                    {
                        "cycle": 0,
                        "operand": "weight",
                        "address": 0,
                        "nbytes": 1,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    backend = RecordedSystolicBackend(path)
    mismatch = SystolicInvocation(
        "different", 4, 4, 4, 8, 8, "ws", 8, 8, 32
    )
    with pytest.raises(BackendOutputError):
        backend.run(mismatch)


def test_config_rejects_unknown_keys(tmp_path: Path) -> None:
    config = tmp_path / "bad.yaml"
    config.write_text(
        """
architecture:
  name: bad
  frequency_hz: 1
  compute_placement: centralized_logic
  bank_groups: 1
  unexpected: true
  arrays: {count: 1, rows: 1, cols: 1, dataflow: ws}
  stacked_sram:
    tiers: 1
    banks_per_tier: 1
    bank_capacity_bytes: 64
    line_bytes: 64
    read_ports_per_bank: 1
    write_ports_per_bank: 1
    read_latency_cycles: 1
    write_latency_cycles: 1
    read_bytes_per_cycle: 1
    write_bytes_per_cycle: 1
  vertical_link: {groups: 1, latency_cycles: 1, bytes_per_cycle: 1}
  global_buffer:
    capacity_bytes: 64
    read_ports: 1
    write_ports: 1
    read_latency_cycles: 1
    write_latency_cycles: 1
    read_bytes_per_cycle: 1
    write_bytes_per_cycle: 1
  noc: {groups: 1, latency_cycles: 1, bytes_per_cycle: 1}
  reduction: {units: 1, lanes_per_unit: 1, latency_cycles: 1}
mapping:
  tile_m: 1
  tile_n: 1
  tile_k: 1
  partition_dim: N
  prefetch_distance_cycles: 0
backend:
  kind: recorded
  result_path: x.json
""",
        encoding="utf-8",
    )
    with pytest.raises(ConfigurationError, match="unknown keys"):
        load_cmodel_config(config)
