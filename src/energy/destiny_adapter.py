import os
import re
import subprocess
import tempfile
from dataclasses import dataclass


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
    read_bandwidth_gbps: float = 0.0


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
        """Parse DESTINY stdout, extracting DATA ARRAY results.

        DESTINY outputs both a SUMMARY section (cache-level, ns/nJ) and
        a CACHE DATA ARRAY DETAILS section (array-level, ps/pJ).
        We parse the data-array section for more accurate per-access values.
        """
        params = SRAMParams()

        # Extract the DATA ARRAY DETAILS section
        data_section = self._extract_data_array_section(output)
        if data_section:
            self._parse_data_array(data_section, params)
        else:
            # Fall back to cache-level SUMMARY parsing
            self._parse_summary(output, params)

        return params

    def _extract_data_array_section(self, output: str) -> str:
        """Extract text between 'CACHE DATA ARRAY DETAILS' and 'CACHE TAG ARRAY DETAILS'."""
        start = output.find("CACHE DATA ARRAY DETAILS")
        if start == -1:
            return ""
        end = output.find("CACHE TAG ARRAY DETAILS", start)
        if end == -1:
            end = len(output)
        return output[start:end]

    def _parse_data_array(self, section: str, params: SRAMParams) -> None:
        """Parse the DATA ARRAY RESULT section (values in ps and pJ)."""
        # Read Latency in ps — look for the top-level "Read Latency = XXXps"
        m = re.search(r'-\s+Read Latency\s*=\s*(\d+\.?\d*(?:e[+-]?\d+)?)ps', section)
        if m:
            params.read_latency_ns = float(m.group(1)) / 1000.0  # ps -> ns

        m = re.search(r'- Write Latency\s*=\s*(\d+\.?\d*(?:e[+-]?\d+)?)ps', section)
        if m:
            params.write_latency_ns = float(m.group(1)) / 1000.0

        # Read Dynamic Energy in pJ
        m = re.search(r'-\s+Read Dynamic Energy\s*=\s*(\d+\.?\d*(?:e[+-]?\d+)?)pJ', section)
        if m:
            params.read_energy_pj = float(m.group(1))

        m = re.search(r'- Write Dynamic Energy\s*=\s*(\d+\.?\d*(?:e[+-]?\d+)?)pJ', section)
        if m:
            params.write_energy_pj = float(m.group(1))

        # Leakage Power in mW
        m = re.search(r'- Leakage Power\s*=\s*(\d+\.?\d*(?:e[+-]?\d+)?)mW', section)
        if m:
            params.leakage_mw = float(m.group(1))

        # Total Area — look for um^2 and convert to mm^2
        m = re.search(r'- Total Area\s*=.*?=\s*(\d+\.?\d*(?:e[+-]?\d+)?)um\^2', section)
        if m:
            params.area_mm2 = float(m.group(1)) / 1e6  # um^2 -> mm^2

        # Read Bandwidth in GB/s
        m = re.search(r'- Read Bandwidth\s*=\s*(\d+\.?\d*(?:e[+-]?\d+)?)GB/s', section)
        if m:
            params.read_bandwidth_gbps = float(m.group(1))

    def _parse_summary(self, output: str, params: SRAMParams) -> None:
        """Fallback: parse the cache-level SUMMARY section (values in ns and nJ)."""
        # Cache Hit Latency in ns
        m = re.search(r'Cache Hit Latency\s*=\s*(\d+\.?\d*(?:e[+-]?\d+)?)ns', output)
        if m:
            params.read_latency_ns = float(m.group(1))

        m = re.search(r'Cache Write Latency\s*=\s*(\d+\.?\d*(?:e[+-]?\d+)?)ns', output)
        if m:
            params.write_latency_ns = float(m.group(1))

        # Cache dynamic energy in nJ -> convert to pJ
        m = re.search(r'Cache Hit Dynamic Energy\s*=\s*(\d+\.?\d*(?:e[+-]?\d+)?)nJ', output)
        if m:
            params.read_energy_pj = float(m.group(1)) * 1000.0  # nJ -> pJ

        m = re.search(r'Cache Write Dynamic Energy\s*=\s*(\d+\.?\d*(?:e[+-]?\d+)?)nJ', output)
        if m:
            params.write_energy_pj = float(m.group(1)) * 1000.0

        # Data Array Leakage Power in mW
        m = re.search(r'Cache Data Array Leakage Power\s*=\s*(\d+\.?\d*(?:e[+-]?\d+)?)mW', output)
        if m:
            params.leakage_mw = float(m.group(1))

        # Data Array Area in mm^2
        m = re.search(r'Data Array Area\s*=.*?=\s*(\d+\.?\d*(?:e[+-]?\d+)?)mm\^2', output)
        if m:
            params.area_mm2 = float(m.group(1))


def ns_to_cycles(ns: float, freq_hz: int) -> int:
    """Convert nanoseconds to cycles at the given frequency, rounding up."""
    import math
    return max(1, math.ceil(ns * freq_hz / 1e9))


def apply_destiny_params(config) -> str:
    """Apply DESTINY-derived SRAM parameters to a SimConfig.

    Returns the source string ("destiny" or "analytical_fallback").
    """
    if config.energy.sram.source != "destiny":
        return config.energy.sram.source

    adapter = DestinyAdapter()
    tile_capacity_kb = (config.sram_pim.bank_capacity_kb
                        * config.sram_pim.banks_per_tile)
    params = adapter.run(
        tech_nm=28,
        capacity_kb=tile_capacity_kb,
        banks=config.sram_pim.banks_per_tile,
        word_bits=config.sram_pim.word_bytes * 8,
    )

    # Check if DESTINY actually ran (non-default values)
    default = SRAMParams()
    if (params.read_energy_pj == default.read_energy_pj
            and params.read_latency_ns == default.read_latency_ns):
        return "analytical_fallback"

    # Apply DESTINY results to config
    config.energy.sram.read_pj_per_access = params.read_energy_pj
    config.energy.sram.write_pj_per_access = params.write_energy_pj
    config.energy.sram.leakage_mw_per_bank = (
        params.leakage_mw / config.sram_pim.banks_per_tile
    )
    config.energy.sram.source = "destiny"

    return "destiny"
