import os
import pytest
from src.config import DRAMConfig
from src.dram_model import (
    create_dram_model, AnalyticalDRAMModel, TraceDRAMModel, DRAMSim3Model,
)


def test_create_dramsim3_model():
    """Factory should create DRAMSim3Model when model='dramsim3'."""
    cfg = DRAMConfig(model="dramsim3")
    model = create_dram_model(cfg, freq_hz=1_000_000_000)
    assert isinstance(model, DRAMSim3Model)


def test_dramsim3_model_fallback():
    """DRAMSim3Model should fall back to analytical when binary is unavailable."""
    cfg = DRAMConfig(
        model="dramsim3",
        dramsim3_dir="/nonexistent/path",
        dramsim3_config="/nonexistent/config.ini",
    )
    model = create_dram_model(cfg, freq_hz=1_000_000_000)
    # Should fall back to analytical and return > 0
    lat = model.get_read_latency(1024)
    assert lat > 0
    lat_w = model.get_write_latency(1024)
    assert lat_w > 0


def test_dramsim3_model_with_real_binary():
    """If DRAMSim3 binary exists, model should return cycle-accurate timing."""
    dramsim3_dir = "/home/NPU-PIM-co-simulator/DRAMsim3"
    exe = os.path.join(dramsim3_dir, "dramsim3main.out")
    if not os.path.isfile(exe):
        pytest.skip("DRAMSim3 binary not available")

    cfg = DRAMConfig(
        model="dramsim3",
        dramsim3_dir=dramsim3_dir,
        dramsim3_config=os.path.join(dramsim3_dir, "configs/DDR4_8Gb_x8_2400.ini"),
    )
    model = create_dram_model(cfg, freq_hz=1_000_000_000)
    lat = model.get_read_latency(4096)
    assert lat > 0


def test_factory_returns_correct_types():
    """Factory should return the right model type for each config."""
    assert isinstance(
        create_dram_model(DRAMConfig(model="analytical"), 1_000_000_000),
        AnalyticalDRAMModel)
    assert isinstance(
        create_dram_model(DRAMConfig(model="trace"), 1_000_000_000),
        TraceDRAMModel)
    assert isinstance(
        create_dram_model(DRAMConfig(model="dramsim3"), 1_000_000_000),
        DRAMSim3Model)
