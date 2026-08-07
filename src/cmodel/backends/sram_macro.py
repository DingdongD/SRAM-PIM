"""Strict CACTI/DESTINY SRAM macro parameter import."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from ..config import SRAMMacroBackendSpec
from ..errors import BackendProtocolError
from .base import SRAMMacroResult
from .process import verify_repository


_NUMBER = r"([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)"


class SRAMMacroBackend:
    def __init__(self, spec: SRAMMacroBackendSpec):
        verify_repository(spec.repository)
        self.spec = spec
        self.repository = Path(spec.repository.path).resolve()
        self.executable = Path(spec.executable).resolve()
        self.config_path = Path(spec.config_path).resolve()
        if not self.executable.is_file():
            raise BackendProtocolError(f"SRAM macro executable does not exist: {self.executable}")
        if not self.config_path.is_file():
            raise BackendProtocolError(f"SRAM macro config does not exist: {self.config_path}")

    def run(self) -> SRAMMacroResult:
        command = (
            [str(self.executable), "-infile", str(self.config_path)]
            if self.spec.kind == "cacti"
            else [str(self.executable), str(self.config_path)]
        )
        completed = subprocess.run(
            command,
            cwd=self.repository,
            check=True,
            capture_output=True,
            text=True,
            timeout=self.spec.timeout_seconds,
        )
        text = completed.stdout + "\n" + completed.stderr
        if self.spec.kind == "cacti":
            return self._parse_cacti(text)
        if self.spec.kind == "destiny":
            return self._parse_destiny(text)
        raise BackendProtocolError(f"unsupported SRAM macro backend {self.spec.kind}")

    @staticmethod
    def _parse_cacti(text: str) -> SRAMMacroResult:
        access_ns = _match(text, rf"Access time \(ns\):\s*{_NUMBER}", "CACTI access time")
        cycle_ns = _match(text, rf"Cycle time \(ns\):\s*{_NUMBER}", "CACTI cycle time")
        read_nj = _match(
            text,
            rf"Total dynamic read energy per access \(nJ\):\s*{_NUMBER}",
            "CACTI read energy",
        )
        write_nj = _match(
            text,
            rf"Total dynamic write energy per access \(nJ\):\s*{_NUMBER}",
            "CACTI write energy",
        )
        leakage_mw = _match(
            text,
            rf"Total leakage power of a bank \(mW\):\s*{_NUMBER}",
            "CACTI leakage power",
        )
        height = _match(text, rf"Cache height x width \(mm\):\s*{_NUMBER}\s*x", "CACTI height")
        width = _match(text, rf"Cache height x width \(mm\):\s*{_NUMBER}\s*x\s*{_NUMBER}", "CACTI width", group=2)
        return SRAMMacroResult(
            access_time_ns=access_ns,
            cycle_time_ns=cycle_ns,
            read_energy_pj=read_nj * 1000.0,
            write_energy_pj=write_nj * 1000.0,
            leakage_mw=leakage_mw,
            area_mm2=height * width,
        )

    @staticmethod
    def _parse_destiny(text: str) -> SRAMMacroResult:
        read_ns = _match(text, rf"Read Latency[^:]*:\s*{_NUMBER}", "DESTINY read latency")
        write_ns = _match(text, rf"Write Latency[^:]*:\s*{_NUMBER}", "DESTINY write latency")
        read_pj = _match(text, rf"Read Dynamic Energy[^:]*:\s*{_NUMBER}", "DESTINY read energy")
        write_pj = _match(text, rf"Write Dynamic Energy[^:]*:\s*{_NUMBER}", "DESTINY write energy")
        leakage_mw = _match(text, rf"Leakage Power[^:]*:\s*{_NUMBER}", "DESTINY leakage power")
        area_mm2 = _match(text, rf"Area[^:]*:\s*{_NUMBER}", "DESTINY area")
        return SRAMMacroResult(
            access_time_ns=max(read_ns, write_ns),
            cycle_time_ns=max(read_ns, write_ns),
            read_energy_pj=read_pj,
            write_energy_pj=write_pj,
            leakage_mw=leakage_mw,
            area_mm2=area_mm2,
        )


def _match(text: str, pattern: str, name: str, group: int = 1) -> float:
    match = re.search(pattern, text)
    if match is None:
        raise BackendProtocolError(f"missing required {name} field")
    return float(match.group(group))
