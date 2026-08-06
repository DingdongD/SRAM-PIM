"""Public strict C-model facade."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .backends import RecordedSystolicBackend, ScaleSimBackend, SystolicBackend
from .config import CModelConfig
from .errors import ConfigurationError
from .ir import GemmOp
from .lowering import ArchitectureLowerer
from .resources import EventSimulator, ResourceGraph, SimulationResult


@dataclass(frozen=True, slots=True)
class CModelRun:
    operation: GemmOp
    result: SimulationResult
    metadata: Mapping[str, Any]

    def to_dict(self, frequency_hz: int) -> dict:
        report = self.result.to_dict(frequency_hz)
        report["operation"] = {
            "op_id": self.operation.op_id,
            "m": self.operation.m,
            "n": self.operation.n,
            "k": self.operation.k,
            "mac_count": self.operation.mac_count,
            "input_bits": self.operation.input_bits,
            "weight_bits": self.operation.weight_bits,
            "accumulator_bits": self.operation.accumulator_bits,
        }
        report["model"] = dict(self.metadata)
        return report


class StrictCModel:
    def __init__(self, config: CModelConfig, backend: SystolicBackend | None = None):
        self.config = config
        self.backend = backend if backend is not None else self._build_backend(config)
        self.resources = ResourceGraph.from_architecture(config.architecture)

    @staticmethod
    def _build_backend(config: CModelConfig) -> SystolicBackend:
        options = dict(config.backend.options)
        if config.backend.kind == "recorded":
            allowed = {"result_path"}
            unknown = set(options) - allowed
            if unknown:
                raise ConfigurationError(
                    f"unknown recorded backend options: {sorted(unknown)}"
                )
            if "result_path" not in options:
                raise ConfigurationError("recorded backend requires result_path")
            return RecordedSystolicBackend(Path(str(options["result_path"])))

        allowed = {
            "architecture_config",
            "python_executable",
            "module",
            "timeout_seconds",
            "trace_word_bytes",
            "expected_version",
            "keep_outputs",
        }
        unknown = set(options) - allowed
        if unknown:
            raise ConfigurationError(
                f"unknown SCALE-Sim backend options: {sorted(unknown)}"
            )
        if "architecture_config" not in options:
            raise ConfigurationError(
                "SCALE-Sim backend requires architecture_config; no fallback is available"
            )
        return ScaleSimBackend(
            architecture_config=str(options["architecture_config"]),
            python_executable=str(options.get("python_executable", "python3")),
            module=str(options.get("module", "scalesim.scale")),
            timeout_seconds=int(options.get("timeout_seconds", 600)),
            trace_word_bytes=int(options.get("trace_word_bytes", 1)),
            expected_version=(
                None
                if options.get("expected_version") is None
                else str(options["expected_version"])
            ),
            keep_outputs=(
                None
                if options.get("keep_outputs") is None
                else str(options["keep_outputs"])
            ),
        )

    def simulate_gemm(self, op: GemmOp) -> CModelRun:
        lowerer = ArchitectureLowerer(
            architecture=self.config.architecture,
            mapping=self.config.mapping,
            backend=self.backend,
        )
        program = lowerer.lower_gemm(op)
        result = EventSimulator(self.resources).run(program)
        return CModelRun(
            operation=op,
            result=result,
            metadata={
                "architecture": self.config.architecture.name,
                "compute_placement": self.config.architecture.placement.value,
                "backend": self.config.backend.kind,
                "backend_runs": program.metadata.get("backend_runs", 0),
                "backend_manifests": program.metadata.get("backend_manifests", []),
                "micro_op_count": len(program.operations),
            },
        )
