"""Replay explicitly selected, immutable backend results.

This backend is intended for regression tests and reproducible offline runs. It
is not a fallback: every requested GEMM/array invocation must have exactly one
matching record or the simulation fails.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from ..errors import BackendOutputError, BackendUnavailableError
from ..ir import Operand, OperandDemand, SystolicInvocation, SystolicResult
from .base import SystolicBackend


class RecordedSystolicBackend(SystolicBackend):
    def __init__(self, result_path: str | Path):
        self.result_path = Path(result_path)
        if not self.result_path.is_file():
            raise BackendUnavailableError(
                f"recorded systolic result does not exist: {self.result_path}"
            )
        self._raw_bytes = self.result_path.read_bytes()
        self._digest = hashlib.sha256(self._raw_bytes).hexdigest()
        try:
            data = json.loads(self._raw_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise BackendOutputError(
                f"invalid recorded result {self.result_path}: {exc}"
            ) from exc
        if not isinstance(data, Mapping):
            raise BackendOutputError("recorded backend root must be a JSON object")
        self._root = data
        records = data.get("results")
        if records is None:
            records = [data]
        if not isinstance(records, list) or not records:
            raise BackendOutputError("recorded backend needs one or more results")
        if not all(isinstance(record, Mapping) for record in records):
            raise BackendOutputError("every recorded result must be a JSON object")
        self._records: list[Mapping[str, Any]] = list(records)

    def run(self, invocation: SystolicInvocation) -> SystolicResult:
        expected = self._invocation_dict(invocation)
        matches = [record for record in self._records if record.get("invocation") == expected]
        if len(matches) != 1:
            available = [record.get("invocation") for record in self._records]
            raise BackendOutputError(
                "strict recorded backend requires exactly one matching invocation; "
                f"matches={len(matches)}\nrequested={expected}\navailable={available}"
            )
        record = matches[0]
        demands_raw = record.get("demands")
        if not isinstance(demands_raw, list) or not demands_raw:
            raise BackendOutputError("recorded result has no demand trace")
        try:
            demands = tuple(
                OperandDemand(
                    cycle=int(item["cycle"]),
                    operand=Operand(str(item["operand"])),
                    address=int(item["address"]),
                    nbytes=int(item["nbytes"]),
                )
                for item in demands_raw
            )
            total_cycles = int(record["total_cycles"])
            utilization = float(record["utilization"])
        except (KeyError, TypeError, ValueError) as exc:
            raise BackendOutputError(f"malformed recorded result: {exc}") from exc
        return SystolicResult(
            backend_name=str(
                record.get(
                    "backend_name", self._root.get("backend_name", "recorded")
                )
            ),
            backend_version=str(
                record.get(
                    "backend_version", self._root.get("backend_version", "unknown")
                )
            ),
            config_digest=self._digest,
            total_cycles=total_cycles,
            utilization=utilization,
            demands=tuple(
                sorted(demands, key=lambda d: (d.cycle, d.operand.value, d.address))
            ),
            raw_metrics=record.get("raw_metrics", {}),
        )

    @staticmethod
    def _invocation_dict(invocation: SystolicInvocation) -> dict[str, Any]:
        return {
            "op_id": invocation.op_id,
            "m": invocation.m,
            "n": invocation.n,
            "k": invocation.k,
            "array_rows": invocation.array_rows,
            "array_cols": invocation.array_cols,
            "dataflow": invocation.dataflow,
            "input_bits": invocation.input_bits,
            "weight_bits": invocation.weight_bits,
            "accumulator_bits": invocation.accumulator_bits,
        }
