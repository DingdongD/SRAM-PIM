import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field


@dataclass
class SRAMParams:
    read_latency_ns: float = 0.534
    write_latency_ns: float = 0.612
    read_energy_pj: float = 3.2
    write_energy_pj: float = 3.8
    leakage_mw: float = 12.5
    area_mm2: float = 0.42
    tech_nm: int = 28
    capacity_kb: int = 256
    banks: int = 32


class DestinyAdapter:
    """Adapter that extracts SRAM macro parameters from DESTINY or falls back to defaults."""

    def __init__(self, destiny_dir: str = "/home/Destiny-Memory-simulator"):
        self.destiny_dir = destiny_dir
        self.cell_file = os.path.join(destiny_dir, "config", "3D", "SRAM", "SRAM.cell")

    def generate_config(self, tech_nm: int = 28, capacity_kb: int = 256,
                        banks: int = 32, word_bits: int = 128) -> str:
        """Generate a DESTINY config file for the given SRAM parameters."""
        cfg = (
            f"-DesignTarget: cache\n"
            f"\n"
            f"-CacheAccessMode: Normal\n"
            f"-Associativity (for cache only): 1\n"
            f"\n"
            f"-ProcessNode: {tech_nm}\n"
            f"\n"
            f"-Capacity (KB): {capacity_kb}\n"
            f"-WordWidth (bit): {word_bits}\n"
            f"\n"
            f"-DeviceRoadmap: HP\n"
            f"\n"
            f"-LocalWireType: LocalAggressive\n"
            f"-LocalWireRepeaterType: RepeatedNone\n"
            f"-LocalWireUseLowSwing: No\n"
            f"\n"
            f"-GlobalWireType: GlobalAggressive\n"
            f"-GlobalWireRepeaterType: RepeatedNone\n"
            f"-GlobalWireUseLowSwing: No\n"
            f"\n"
            f"-Routing: H-tree\n"
            f"\n"
            f"-InternalSensing: true\n"
            f"\n"
            f"-MemoryCellInputFile: {self.cell_file}\n"
            f"\n"
            f"-Temperature (K): 350\n"
            f"\n"
            f"-OptimizationTarget: ReadLatency\n"
            f"-EnablePruning: Yes\n"
            f"\n"
            f"-BufferDesignOptimization: latency\n"
            f"\n"
            f"-StackedDieCount: 1\n"
            f"-LocalTSVProjection: 0\n"
            f"-GlobalTSVProjection: 0\n"
            f"-TSVRedundancy: 1.0\n"
        )
        return cfg

    def run(self, tech_nm: int = 28, capacity_kb: int = 256,
            banks: int = 32, word_bits: int = 128) -> SRAMParams:
        """Run DESTINY and parse its output, falling back to analytical defaults."""
        cfg_content = self.generate_config(tech_nm, capacity_kb, banks, word_bits)

        exe = os.path.join(self.destiny_dir, "destiny")
        if not os.path.exists(exe):
            return SRAMParams(tech_nm=tech_nm, capacity_kb=capacity_kb, banks=banks)

        tmp_fd, cfg_path = tempfile.mkstemp(suffix=".cfg", dir=self.destiny_dir)
        try:
            with os.fdopen(tmp_fd, "w") as f:
                f.write(cfg_content)

            result = subprocess.run(
                [exe, cfg_path],
                capture_output=True,
                text=True,
                timeout=120,
                cwd=self.destiny_dir,
            )
            output = result.stdout + result.stderr
            params = self.parse_output(output)
            params.tech_nm = tech_nm
            params.capacity_kb = capacity_kb
            params.banks = banks
            return params
        except (subprocess.TimeoutExpired, subprocess.SubprocessError, OSError):
            return SRAMParams(tech_nm=tech_nm, capacity_kb=capacity_kb, banks=banks)
        finally:
            try:
                os.unlink(cfg_path)
            except OSError:
                pass

    def parse_output(self, output: str) -> SRAMParams:
        """Parse DESTINY stdout and return SRAMParams populated from the output."""
        params = SRAMParams()

        patterns = {
            "read_latency_ns": r"Read Latency[^:]*:\s*(\d+\.?\d*(?:e[+-]?\d+)?)",
            "write_latency_ns": r"Write Latency[^:]*:\s*(\d+\.?\d*(?:e[+-]?\d+)?)",
            "read_energy_pj": r"Read Dynamic Energy[^:]*:\s*(\d+\.?\d*(?:e[+-]?\d+)?)",
            "write_energy_pj": r"Write Dynamic Energy[^:]*:\s*(\d+\.?\d*(?:e[+-]?\d+)?)",
            "leakage_mw": r"Leakage Power[^:]*:\s*(\d+\.?\d*(?:e[+-]?\d+)?)",
            "area_mm2": r"Area[^:]*:\s*(\d+\.?\d*(?:e[+-]?\d+)?)",
        }

        for field_name, pattern in patterns.items():
            m = re.search(pattern, output)
            if m:
                setattr(params, field_name, float(m.group(1)))

        return params
