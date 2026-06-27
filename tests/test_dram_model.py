from src.dram_model import DRAMModel
from src.config import DRAMConfig


def test_analytical_read_latency():
    config = DRAMConfig(effective_bandwidth_gbps=25.6, fixed_latency_ns=80)
    model = DRAMModel(config, freq_hz=1_000_000_000)
    # 64KB at 25.6 GB/s = 65536 / 25.6e9 * 1e9 = 2.56 us = 2560 ns = 2560 cycles at 1GHz
    # plus 80ns fixed = 2640 cycles
    lat = model.get_read_latency(65536)
    assert lat == 2640


def test_analytical_write_latency():
    config = DRAMConfig(effective_bandwidth_gbps=25.6, fixed_latency_ns=80)
    model = DRAMModel(config, freq_hz=1_000_000_000)
    lat = model.get_write_latency(65536)
    assert lat == 2640


def test_zero_bytes_read():
    config = DRAMConfig(effective_bandwidth_gbps=25.6, fixed_latency_ns=80)
    model = DRAMModel(config, freq_hz=1_000_000_000)
    lat = model.get_read_latency(0)
    assert lat == 80  # only fixed latency


def test_small_bytes_rounds_up():
    config = DRAMConfig(effective_bandwidth_gbps=25.6, fixed_latency_ns=0)
    model = DRAMModel(config, freq_hz=1_000_000_000)
    # 1 byte at 25.6 GB/s = 1/25.6e9 s -> tiny fraction of a cycle, ceil -> 1 cycle
    lat = model.get_read_latency(1)
    assert lat == 1


def test_different_frequency():
    config = DRAMConfig(effective_bandwidth_gbps=8.0, fixed_latency_ns=100)
    model = DRAMModel(config, freq_hz=2_000_000_000)  # 2 GHz
    # fixed latency: 100ns * 2GHz = 200 cycles
    # 1024 bytes at 8 GB/s = 128ns = 256 cycles at 2GHz
    lat = model.get_read_latency(1024)
    assert lat == 200 + 256
