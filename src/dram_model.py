import math
from src.config import DRAMConfig


class DRAMModel:
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
