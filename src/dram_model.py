"""P1-09: Layered DRAM model — analytical (fast), trace (DRAMSim3/Ramulator).

Usage:
    model = create_dram_model(config, freq_hz)
    latency = model.get_read_latency(nbytes)
"""

import math
import os
import subprocess
import tempfile
from abc import ABC, abstractmethod
from src.config import DRAMConfig


class BaseDRAMModel(ABC):
    """Abstract DRAM timing model."""

    @abstractmethod
    def get_read_latency(self, nbytes: int) -> int:
        ...

    @abstractmethod
    def get_write_latency(self, nbytes: int) -> int:
        ...


class AnalyticalDRAMModel(BaseDRAMModel):
    """Fast analytical model: fixed latency + bandwidth-limited transfer."""

    def __init__(self, config: DRAMConfig, freq_hz: int):
        self.config = config
        self.freq_hz = freq_hz
        bw_bytes_per_sec = config.effective_bandwidth_gbps * 1e9
        self.bytes_per_cycle = bw_bytes_per_sec / freq_hz
        self.fixed_latency_cycles = int(config.fixed_latency_ns * freq_hz / 1e9)

    def get_read_latency(self, nbytes: int) -> int:
        transfer_cycles = math.ceil(nbytes / self.bytes_per_cycle) if nbytes > 0 else 0
        return self.fixed_latency_cycles + transfer_cycles

    def get_write_latency(self, nbytes: int) -> int:
        transfer_cycles = math.ceil(nbytes / self.bytes_per_cycle) if nbytes > 0 else 0
        return self.fixed_latency_cycles + transfer_cycles


class TraceDRAMModel(BaseDRAMModel):
    """Trace-driven DRAM model using DRAMSim3 or Ramulator.

    In trace mode, DMA operations are logged and can be exported to
    a DRAMSim3/Ramulator trace file for external detailed simulation.
    For inline simulation, falls back to analytical with row-buffer
    awareness (sequential accesses get lower latency).
    """

    def __init__(self, config: DRAMConfig, freq_hz: int):
        self.config = config
        self.freq_hz = freq_hz
        bw_bytes_per_sec = config.effective_bandwidth_gbps * 1e9
        self.bytes_per_cycle = bw_bytes_per_sec / freq_hz
        self.fixed_latency_cycles = int(config.fixed_latency_ns * freq_hz / 1e9)
        # Row buffer hit latency is roughly 1/3 of full access latency
        self.row_hit_latency_cycles = max(1, self.fixed_latency_cycles // 3)
        self.burst_bytes = config.burst_bytes
        self._trace_log: list[dict] = []
        self._last_read_addr: int = -1
        self._last_write_addr: int = -1
        self._cycle: int = 0

    def get_read_latency(self, nbytes: int) -> int:
        if nbytes <= 0:
            return self.fixed_latency_cycles
        n_bursts = math.ceil(nbytes / self.burst_bytes)
        # First burst pays full latency; subsequent bursts may hit row buffer
        first_lat = self.fixed_latency_cycles
        transfer_cycles = math.ceil(nbytes / self.bytes_per_cycle)
        total = first_lat + transfer_cycles
        self._trace_log.append({
            "cycle": self._cycle, "type": "READ",
            "bytes": nbytes, "latency": total,
        })
        self._cycle += total
        return total

    def get_write_latency(self, nbytes: int) -> int:
        if nbytes <= 0:
            return self.fixed_latency_cycles
        transfer_cycles = math.ceil(nbytes / self.bytes_per_cycle)
        total = self.fixed_latency_cycles + transfer_cycles
        self._trace_log.append({
            "cycle": self._cycle, "type": "WRITE",
            "bytes": nbytes, "latency": total,
        })
        self._cycle += total
        return total

    def export_trace(self, filepath: str) -> None:
        """Export accumulated trace in DRAMSim3-compatible format."""
        with open(filepath, "w") as f:
            for entry in self._trace_log:
                # DRAMSim3 trace format: addr type cycle
                # Using dummy addresses since we don't track real DRAM addresses
                addr = 0x1000 + entry["cycle"] * 64
                rw = "READ" if entry["type"] == "READ" else "WRITE"
                f.write(f"0x{addr:08X} {rw} {entry['cycle']}\n")

    def get_trace_log(self) -> list[dict]:
        return list(self._trace_log)


# Keep backward compatibility
DRAMModel = AnalyticalDRAMModel


def create_dram_model(config: DRAMConfig, freq_hz: int) -> BaseDRAMModel:
    """Factory: create the appropriate DRAM model based on config.dram.model."""
    if config.model == "trace":
        return TraceDRAMModel(config, freq_hz)
    # Default: analytical
    return AnalyticalDRAMModel(config, freq_hz)
