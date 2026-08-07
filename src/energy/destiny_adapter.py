from __future__ import annotations

import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class SRAMParams:
    read_latency_ns: float
    write_latency_ns: float
    read_energy_pj: float
    write_energy_pj: float
    leakage_mw: float
    area_mm2: float
    tech_nm: int
    capacity_kb: int
    banks: int


class DestinyAdapter:
    def __init__(self, destiny_dir: str):
        self.destiny_dir = Path(destiny_dir).resolve()
        self.executable = self.destiny_dir / "destiny"
        self.cell_file = self.destiny_dir / "config" / "3D" / "SRAM" / "SRAM.cell"
        if not self.executable.is_file():
            raise FileNotFoundError(self.executable)
        if not self.cell_file.is_file():
            raise FileNotFoundError(self.cell_file)

    def generate_config(self, tech_nm: int, capacity_kb: int, banks: int, word_bits: int) -> str:
        if tech_nm <= 0 or capacity_kb <= 0 or banks <= 0 or word_bits <= 0:
            raise ValueError("DESTINY SRAM configuration values must be positive")
        return (
            "-DesignTarget: cache\n\n"
            "-CacheAccessMode: Normal\n"
            "-Associativity (for cache only): 1\n\n"
            f"-ProcessNode: {tech_nm}\n\n"
            f"-Capacity (KB): {capacity_kb}\n"
            f"-WordWidth (bit): {word_bits}\n\n"
            "-DeviceRoadmap: HP\n\n"
            "-LocalWireType: LocalAggressive\n"
            "-LocalWireRepeaterType: RepeatedNone\n"
            "-LocalWireUseLowSwing: No\n\n"
            "-GlobalWireType: GlobalAggressive\n"
            "-GlobalWireRepeaterType: RepeatedNone\n"
            "-GlobalWireUseLowSwing: No\n\n"
            "-Routing: H-tree\n\n"
            "-InternalSensing: true\n\n"
            f"-MemoryCellInputFile: {self.cell_file}\n\n"
            "-Temperature (K): 350\n\n"
            "-OptimizationTarget: ReadLatency\n"
            "-EnablePruning: Yes\n\n"
            "-BufferDesignOptimization: latency\n\n"
            "-StackedDieCount: 1\n"
            "-LocalTSVProjection: 0\n"
            "-GlobalTSVProjection: 0\n"
            "-TSVRedundancy: 1.0\n"
        )

    def run(self, tech_nm: int, capacity_kb: int, banks: int, word_bits: int) -> SRAMParams:
        config = self.generate_config(tech_nm, capacity_kb, banks, word_bits)
        with tempfile.TemporaryDirectory(prefix="strict_destiny_") as temp_dir:
            config_path = Path(temp_dir) / "sram.cfg"
            config_path.write_text(config, encoding="utf-8")
            completed = subprocess.run(
                [str(self.executable), str(config_path)],
                cwd=self.destiny_dir,
                check=True,
                capture_output=True,
                text=True,
            )
        parsed = self.parse_output(completed.stdout + "\n" + completed.stderr)
        return SRAMParams(
            parsed[0], parsed[1], parsed[2], parsed[3], parsed[4], parsed[5],
            tech_nm, capacity_kb, banks,
        )

    @staticmethod
    def parse_output(output: str) -> tuple[float, float, float, float, float, float]:
        patterns = (
            r"Read Latency[^:]*:\s*([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)",
            r"Write Latency[^:]*:\s*([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)",
            r"Read Dynamic Energy[^:]*:\s*([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)",
            r"Write Dynamic Energy[^:]*:\s*([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)",
            r"Leakage Power[^:]*:\s*([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)",
            r"Area[^:]*:\s*([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)",
        )
        values: list[float] = []
        for pattern in patterns:
            match = re.search(pattern, output)
            if match is None:
                raise ValueError(f"DESTINY output missing required field matching {pattern}")
            values.append(float(match.group(1)))
        return tuple(values)  # type: ignore[return-value]
