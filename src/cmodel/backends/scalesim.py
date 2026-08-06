"""Strict SCALE-Sim subprocess adapter.

The adapter uses SCALE-Sim's documented module CLI and requires both a compute
report and cycle-tagged SRAM traces. Missing traces are fatal because summary
cycles alone are insufficient for architecture-level bank/link diagnosis.
"""

from __future__ import annotations

import csv
import hashlib
from dataclasses import asdict
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable

from ..errors import (
    BackendExecutionError,
    BackendOutputError,
    BackendUnavailableError,
)
from ..ir import Operand, OperandDemand, SystolicInvocation, SystolicResult
from .base import SystolicBackend


class ScaleSimBackend(SystolicBackend):
    def __init__(
        self,
        *,
        architecture_config: str | Path,
        python_executable: str = sys.executable,
        module: str = "scalesim.scale",
        timeout_seconds: int = 600,
        trace_word_bytes: int = 1,
        expected_version: str | None = None,
        keep_outputs: str | Path | None = None,
    ) -> None:
        self.architecture_config = Path(architecture_config)
        self.python_executable = python_executable
        self.module = module
        self.timeout_seconds = timeout_seconds
        self.trace_word_bytes = trace_word_bytes
        self.expected_version = expected_version
        self.keep_outputs = Path(keep_outputs) if keep_outputs else None
        if not self.architecture_config.is_file():
            raise BackendUnavailableError(
                f"SCALE-Sim architecture config does not exist: {self.architecture_config}"
            )
        if shutil.which(self.python_executable) is None and not Path(
            self.python_executable
        ).is_file():
            raise BackendUnavailableError(
                f"Python executable for SCALE-Sim not found: {self.python_executable}"
            )
        if timeout_seconds <= 0 or trace_word_bytes <= 0:
            raise BackendUnavailableError("timeout and trace_word_bytes must be positive")

    def run(self, invocation: SystolicInvocation) -> SystolicResult:
        with tempfile.TemporaryDirectory(prefix="strict_scalesim_") as tmp:
            root = Path(tmp)
            topology = root / "topology.csv"
            output = root / "output"
            output.mkdir()
            self._write_gemm_topology(topology, invocation)
            command = [
                self.python_executable,
                "-m",
                self.module,
                "-c",
                str(self.architecture_config),
                "-t",
                str(topology),
                "-p",
                str(output),
            ]
            env = dict(os.environ)
            env["PYTHONHASHSEED"] = "0"
            try:
                completed = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_seconds,
                    env=env,
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                raise BackendExecutionError(
                    f"failed to execute SCALE-Sim command {command}: {exc}"
                ) from exc
            if completed.returncode != 0:
                raise BackendExecutionError(
                    "SCALE-Sim failed with exit code "
                    f"{completed.returncode}\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
                )

            version = self._detect_version(completed.stdout, completed.stderr)
            if self.expected_version is not None and version != self.expected_version:
                raise BackendOutputError(
                    f"SCALE-Sim version mismatch: expected {self.expected_version!r}, "
                    f"got {version!r}"
                )
            total_cycles, utilization, report_metrics = self._parse_compute_report(
                output, invocation.op_id
            )
            demands = self._parse_sram_traces(output)
            operands = {d.operand for d in demands}
            missing = {Operand.INPUT, Operand.WEIGHT} - operands
            if missing:
                raise BackendOutputError(
                    "SCALE-Sim output is missing strict operand traces: "
                    f"{sorted(item.value for item in missing)}"
                )

            digest = self._config_digest(topology)
            if self.keep_outputs is not None:
                target = self.keep_outputs / f"{invocation.op_id}_{digest[:12]}"
                if target.exists():
                    shutil.rmtree(target)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(output, target)
                shutil.copy2(topology, target / "topology.csv")
                (target / "invocation.json").write_text(
                    json.dumps(asdict(invocation), indent=2, sort_keys=True),
                    encoding="utf-8",
                )

            return SystolicResult(
                backend_name="SCALE-Sim",
                backend_version=version,
                config_digest=digest,
                total_cycles=total_cycles,
                utilization=utilization,
                demands=tuple(demands),
                raw_metrics=report_metrics,
            )

    def _config_digest(self, topology: Path) -> str:
        hasher = hashlib.sha256()
        hasher.update(self.architecture_config.read_bytes())
        hasher.update(topology.read_bytes())
        hasher.update(self.module.encode("utf-8"))
        return hasher.hexdigest()

    @staticmethod
    def _write_gemm_topology(path: Path, invocation: SystolicInvocation) -> None:
        # Represent A[M,K] x B[K,N] as a 1x1 convolution over M spatial
        # positions, K input channels, and N output filters. This is the
        # standard GEMM-equivalent form consumed by SCALE-Sim topology files.
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "Layer name",
                    "IFMAP Height",
                    "IFMAP Width",
                    "Filter Height",
                    "Filter Width",
                    "Channels",
                    "Num Filter",
                    "Strides",
                ]
            )
            writer.writerow(
                [invocation.op_id, 1, invocation.m, 1, 1, invocation.k, invocation.n, 1]
            )

    @staticmethod
    def _detect_version(stdout: str, stderr: str) -> str:
        for line in (stdout + "\n" + stderr).splitlines():
            lower = line.lower()
            if "scale-sim" in lower and "version" in lower:
                return line.strip()
        return "unreported"

    @staticmethod
    def _find_files(root: Path, tokens: Iterable[str]) -> list[Path]:
        lowered = tuple(token.lower() for token in tokens)
        return sorted(
            path
            for path in root.rglob("*")
            if path.is_file() and all(token in path.name.lower() for token in lowered)
        )

    def _parse_compute_report(
        self, root: Path, op_id: str
    ) -> tuple[int, float, dict[str, str]]:
        candidates = self._find_files(root, ("compute", "report"))
        if len(candidates) != 1:
            raise BackendOutputError(
                f"expected exactly one SCALE-Sim compute report, found {candidates}"
            )
        with candidates[0].open("r", newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise BackendOutputError("SCALE-Sim compute report is empty")
        row = next(
            (
                item
                for item in rows
                if any(str(value).strip() == op_id for value in item.values())
            ),
            rows[0] if len(rows) == 1 else None,
        )
        if row is None:
            raise BackendOutputError(
                f"cannot identify layer {op_id!r} in compute report {candidates[0]}"
            )

        def normalized(text: str) -> str:
            return "".join(ch for ch in text.lower() if ch.isalnum())

        normalized_row = {normalized(key): value for key, value in row.items() if key}

        def get_value(aliases: tuple[str, ...]) -> str:
            for alias in aliases:
                key = normalized(alias)
                if key in normalized_row and str(normalized_row[key]).strip():
                    return str(normalized_row[key]).strip()
            raise BackendOutputError(
                f"missing columns {aliases} in SCALE-Sim report; "
                f"available={sorted(row)}"
            )

        cycles_text = get_value(("total cycles", "overall cycles", "cycles"))
        util_text = get_value(
            ("overall utilization %", "overall util %", "utilization %", "utilization")
        )
        try:
            total_cycles = int(float(cycles_text.replace(",", "")))
            utilization_raw = float(util_text.rstrip("%"))
        except ValueError as exc:
            raise BackendOutputError(
                f"invalid cycles/utilization in report: {cycles_text!r}, {util_text!r}"
            ) from exc
        utilization = utilization_raw / 100.0 if utilization_raw > 1.0 else utilization_raw
        return total_cycles, utilization, {str(k): str(v) for k, v in row.items()}

    def _parse_sram_traces(self, root: Path) -> list[OperandDemand]:
        definitions = (
            (Operand.INPUT, ("sram", "ifmap")),
            (Operand.WEIGHT, ("sram", "filter")),
            (Operand.OUTPUT, ("sram", "ofmap")),
        )
        demands: list[OperandDemand] = []
        operand_base = {
            Operand.INPUT: 0x1000_0000,
            Operand.WEIGHT: 0x2000_0000,
            Operand.OUTPUT: 0x3000_0000,
        }
        for operand, tokens in definitions:
            candidates = self._find_files(root, tokens)
            for path in candidates:
                demands.extend(
                    self._parse_one_trace(path, operand, operand_base[operand])
                )
        return sorted(
            demands, key=lambda demand: (demand.cycle, demand.operand.value, demand.address)
        )

    def _parse_one_trace(
        self, path: Path, operand: Operand, base_address: int
    ) -> list[OperandDemand]:
        demands: list[OperandDemand] = []
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.reader(handle)
            for row in reader:
                if not row:
                    continue
                try:
                    cycle = int(float(row[0].strip()))
                except ValueError:
                    # Header row.
                    continue
                for token in row[1:]:
                    text = token.strip()
                    if not text:
                        continue
                    try:
                        address = int(text, 0)
                    except ValueError:
                        try:
                            address = int(float(text))
                        except ValueError as exc:
                            raise BackendOutputError(
                                f"invalid address {text!r} in trace {path}"
                            ) from exc
                    if address < 0:
                        # SCALE-Sim traces may use negative values as idle sentinels.
                        continue
                    demands.append(
                        OperandDemand(
                            cycle=cycle,
                            operand=operand,
                            address=base_address + address * self.trace_word_bytes,
                            nbytes=self.trace_word_bytes,
                        )
                    )
        return demands
