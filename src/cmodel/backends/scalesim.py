"""Strict SCALE-Sim adapter used as the common systolic-array backend."""

from __future__ import annotations

import configparser
import csv
import os
import subprocess
import tempfile
from pathlib import Path

from ..config import ArraySpec, ScaleSimSpec
from ..errors import BackendProtocolError
from .base import ScaleSimResult, TensorDemand
from .process import verify_repository


class ScaleSimBackend:
    def __init__(self, spec: ScaleSimSpec, array: ArraySpec):
        verify_repository(spec.repository)
        self.spec = spec
        self.repository = Path(spec.repository.path).resolve()
        self.architecture_config = Path(spec.architecture_config).resolve()
        if not self.architecture_config.is_file():
            raise BackendProtocolError(
                f"SCALE-Sim architecture config does not exist: {self.architecture_config}"
            )
        parser = configparser.ConfigParser()
        parser.optionxform = str
        with self.architecture_config.open("r", encoding="utf-8") as handle:
            parser.read_file(handle)
        architecture = parser["architecture_presets"]
        if int(architecture["ArrayHeight"]) != array.rows:
            raise BackendProtocolError("SCALE-Sim ArrayHeight does not match C-model array rows")
        if int(architecture["ArrayWidth"]) != array.cols:
            raise BackendProtocolError("SCALE-Sim ArrayWidth does not match C-model array cols")
        if architecture["Dataflow"].strip() != array.dataflow:
            raise BackendProtocolError("SCALE-Sim Dataflow does not match C-model dataflow")
        general = parser["general"]
        if general["run_name"].strip() != spec.run_name:
            raise BackendProtocolError("SCALE-Sim run_name does not match backend configuration")

    def run_gemm(
        self,
        *,
        op_id: str,
        m: int,
        n: int,
        k: int,
        input_base: int,
        weight_base: int,
        output_base: int,
    ) -> ScaleSimResult:
        with tempfile.TemporaryDirectory(prefix="strict_scalesim_") as temp_dir:
            root = Path(temp_dir)
            topology = root / "topology.csv"
            output_root = root / "output"
            output_root.mkdir()
            self._write_topology(topology, op_id, m, n, k)
            env = os.environ.copy()
            env["PYTHONPATH"] = str(self.repository)
            completed = subprocess.run(
                [
                    self.spec.python_executable,
                    "-m",
                    self.spec.module,
                    "-c",
                    str(self.architecture_config),
                    "-t",
                    str(topology),
                    "-i",
                    "gemm",
                    "-p",
                    str(output_root),
                ],
                cwd=self.repository,
                env=env,
                check=False,
                capture_output=True,
                text=True,
                timeout=self.spec.timeout_seconds,
            )
            if completed.returncode != 0:
                raise BackendProtocolError(
                    "SCALE-Sim execution failed:\n"
                    f"stdout:\n{completed.stdout}\n"
                    f"stderr:\n{completed.stderr}"
                )
            run_root = output_root / self.spec.run_name
            report_path = run_root / "COMPUTE_REPORT.csv"
            trace_root = run_root / "layer0"
            if not report_path.is_file():
                raise BackendProtocolError(f"SCALE-Sim compute report is missing: {report_path}")
            if not trace_root.is_dir():
                raise BackendProtocolError(f"SCALE-Sim layer trace directory is missing: {trace_root}")
            cycles, utilization = self._parse_report(report_path)
            demands = (
                *self._parse_trace(trace_root / "IFMAP_SRAM_TRACE.csv", "input", input_base),
                *self._parse_trace(trace_root / "FILTER_SRAM_TRACE.csv", "weight", weight_base),
                *self._parse_trace(trace_root / "OFMAP_SRAM_TRACE.csv", "output", output_base),
            )
            if not any(demand.operand == "input" for demand in demands):
                raise BackendProtocolError("SCALE-Sim IFMAP trace contains no demands")
            if not any(demand.operand == "weight" for demand in demands):
                raise BackendProtocolError("SCALE-Sim FILTER trace contains no demands")
            return ScaleSimResult(
                cycles=cycles,
                utilization=utilization,
                demands=tuple(sorted(demands, key=lambda item: (item.cycle, item.operand, item.address))),
                report_path=str(report_path),
                trace_directory=str(trace_root),
            )

    @staticmethod
    def _write_topology(path: Path, op_id: str, m: int, n: int, k: int) -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["Layer Name", "M", "N", "K", ""])
            writer.writerow([op_id, m, n, k, ""])

    @staticmethod
    def _parse_report(path: Path) -> tuple[int, float]:
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != 1:
            raise BackendProtocolError(f"SCALE-Sim report must contain exactly one layer, got {len(rows)}")
        row = rows[0]
        total_cycles = int(float(row[" Total Cycles"].strip()))
        stall_cycles = int(float(row[" Stall Cycles"].strip()))
        if stall_cycles != 0:
            raise BackendProtocolError(
                f"SCALE-Sim produced {stall_cycles} internal SRAM stall cycles; strict external-memory mode requires zero"
            )
        compute_cycles = total_cycles
        if compute_cycles <= 0:
            raise BackendProtocolError("SCALE-Sim compute cycles are non-positive")
        utilization = float(row[" Overall Util %"].strip()) / 100.0
        if utilization < 0.0 or utilization > 1.0:
            raise BackendProtocolError("SCALE-Sim utilization is outside [0, 1]")
        return compute_cycles, utilization

    def _parse_trace(self, path: Path, operand: str, base_address: int) -> tuple[TensorDemand, ...]:
        if not path.is_file():
            raise BackendProtocolError(f"SCALE-Sim SRAM trace is missing: {path}")
        rows: list[tuple[int, list[int]]] = []
        all_addresses: list[int] = []
        with path.open("r", newline="", encoding="utf-8") as handle:
            for raw_row in csv.reader(handle):
                if not raw_row:
                    continue
                cycle = int(raw_row[0])
                addresses: list[int] = []
                for token in raw_row[1:]:
                    value = int(token)
                    if value >= 0:
                        addresses.append(value)
                        all_addresses.append(value)
                rows.append((cycle, addresses))
        if not all_addresses and operand != "output":
            raise BackendProtocolError(f"SCALE-Sim {operand} trace contains no addresses")
        if not all_addresses:
            return ()
        trace_base = min(all_addresses)
        demands: list[TensorDemand] = []
        for cycle, addresses in rows:
            for address in addresses:
                demands.append(
                    TensorDemand(
                        cycle=cycle,
                        operand=operand,
                        address=base_address + (address - trace_base) * self.spec.trace_word_bytes,
                        nbytes=self.spec.trace_word_bytes,
                    )
                )
        return tuple(demands)
