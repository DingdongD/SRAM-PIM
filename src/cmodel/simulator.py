"""Strict simulator facade."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .backends import BookSim2Backend, Ramulator2Backend, SRAMMacroBackend, ScaleSimBackend
from .backends.base import SRAMMacroResult
from .config import CModelConfig
from .errors import CModelError
from .lowering import ArchitectureLowerer
from .operators import Operator
from .scheduler import CycleScheduler, SimulationResult


@dataclass(frozen=True, slots=True)
class CModelRun:
    operation: Operator
    result: SimulationResult
    program_metadata: dict[str, Any]
    manifest: dict[str, Any]

    def to_dict(self, frequency_hz: int) -> dict[str, Any]:
        report = self.result.to_dict(frequency_hz)
        report["operation"] = asdict(self.operation)
        report["program"] = dict(self.program_metadata)
        report["model_manifest"] = dict(self.manifest)
        return report


class StrictCModel:
    def __init__(
        self,
        config: CModelConfig,
        scalesim,
        ramulator2,
        booksim2,
        macro_result: SRAMMacroResult,
    ):
        self.config = config
        self.scalesim = scalesim
        self.ramulator2 = ramulator2
        self.booksim2 = booksim2
        self.macro_result = macro_result
        self.used = False

    @classmethod
    def from_config(cls, config: CModelConfig) -> "StrictCModel":
        macro_result = SRAMMacroBackend(config.backends.sram_macro).run()
        return cls(
            config=config,
            scalesim=ScaleSimBackend(config.backends.scalesim, config.architecture.arrays),
            ramulator2=Ramulator2Backend(config.backends.ramulator2, config.architecture.dram),
            booksim2=BookSim2Backend(config.backends.booksim2, config.architecture.noc),
            macro_result=macro_result,
        )

    def simulate(self, operation: Operator) -> CModelRun:
        if self.used:
            raise CModelError("StrictCModel instances simulate exactly one lowered program")
        self.used = True
        lowerer = ArchitectureLowerer(self.config, self.scalesim, self.macro_result)
        program = lowerer.lower(operation)
        scheduler = CycleScheduler(
            self.config,
            self.macro_result,
            self.ramulator2,
            self.booksim2,
        )
        result = scheduler.run(program)
        self.ramulator2.close()
        self.booksim2.close()
        manifest = {
            "architecture": self.config.architecture.name,
            "compute_placement": self.config.architecture.compute_placement,
            "frequency_hz": self.config.architecture.frequency_hz,
            "array": asdict(self.config.architecture.arrays),
            "mapping": asdict(self.config.mapping),
            "backends": {
                "scalesim_commit": self.config.backends.scalesim.repository.expected_commit,
                "ramulator2_commit": self.config.backends.ramulator2.repository.expected_commit,
                "booksim2_commit": self.config.backends.booksim2.repository.expected_commit,
                "sram_macro_kind": self.config.backends.sram_macro.kind,
                "sram_macro_commit": self.config.backends.sram_macro.repository.expected_commit,
            },
            "sram_macro": asdict(self.macro_result),
        }
        return CModelRun(operation, result, dict(program.metadata), manifest)
