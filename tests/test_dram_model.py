import os
import tempfile
from src.dram_model import (AnalyticalDRAMModel, TraceDRAMModel,
                             create_dram_model, DRAMModel)
from src.config import DRAMConfig


def test_analytical_read_latency():
    config = DRAMConfig(effective_bandwidth_gbps=25.6, fixed_latency_ns=80)
    model = AnalyticalDRAMModel(config, freq_hz=1_000_000_000)
    lat = model.get_read_latency(65536)
    assert lat == 2640


def test_analytical_write_latency():
    config = DRAMConfig(effective_bandwidth_gbps=25.6, fixed_latency_ns=80)
    model = AnalyticalDRAMModel(config, freq_hz=1_000_000_000)
    lat = model.get_write_latency(65536)
    assert lat == 2640


def test_zero_bytes_read():
    config = DRAMConfig(effective_bandwidth_gbps=25.6, fixed_latency_ns=80)
    model = AnalyticalDRAMModel(config, freq_hz=1_000_000_000)
    lat = model.get_read_latency(0)
    assert lat == 80


def test_small_bytes_rounds_up():
    config = DRAMConfig(effective_bandwidth_gbps=25.6, fixed_latency_ns=0)
    model = AnalyticalDRAMModel(config, freq_hz=1_000_000_000)
    lat = model.get_read_latency(1)
    assert lat == 1


def test_different_frequency():
    config = DRAMConfig(effective_bandwidth_gbps=8.0, fixed_latency_ns=100)
    model = AnalyticalDRAMModel(config, freq_hz=2_000_000_000)
    lat = model.get_read_latency(1024)
    assert lat == 200 + 256


def test_backward_compat_alias():
    """DRAMModel should still work as an alias for AnalyticalDRAMModel."""
    config = DRAMConfig(effective_bandwidth_gbps=25.6, fixed_latency_ns=80)
    model = DRAMModel(config, freq_hz=1_000_000_000)
    assert isinstance(model, AnalyticalDRAMModel)
    assert model.get_read_latency(65536) == 2640


def test_factory_analytical():
    config = DRAMConfig(model="analytical")
    model = create_dram_model(config, 1_000_000_000)
    assert isinstance(model, AnalyticalDRAMModel)


def test_factory_trace():
    config = DRAMConfig(model="trace")
    model = create_dram_model(config, 1_000_000_000)
    assert isinstance(model, TraceDRAMModel)


def test_trace_model_latency():
    config = DRAMConfig(effective_bandwidth_gbps=25.6, fixed_latency_ns=80,
                        model="trace")
    model = TraceDRAMModel(config, freq_hz=1_000_000_000)
    lat = model.get_read_latency(65536)
    assert lat > 0
    assert len(model.get_trace_log()) == 1
    assert model.get_trace_log()[0]["type"] == "READ"


def test_trace_export():
    config = DRAMConfig(effective_bandwidth_gbps=25.6, fixed_latency_ns=80,
                        model="trace")
    model = TraceDRAMModel(config, freq_hz=1_000_000_000)
    model.get_read_latency(4096)
    model.get_write_latency(2048)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".trace",
                                     delete=False) as f:
        path = f.name
    try:
        model.export_trace(path)
        with open(path) as f:
            lines = f.readlines()
        assert len(lines) == 2
        assert "READ" in lines[0]
        assert "WRITE" in lines[1]
    finally:
        os.unlink(path)
