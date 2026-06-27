import pytest
from src.config import load_config, SimConfig


def test_load_config_returns_simconfig():
    config = load_config("configs")
    assert isinstance(config, SimConfig)
    assert config.system.frequency_hz == 1_000_000_000
    assert config.system.mode == "warm_resident"


def test_sram_pim_config():
    config = load_config("configs")
    assert config.sram_pim.tiles == 16
    assert config.sram_pim.banks_per_tile == 32
    assert config.sram_pim.bank_capacity_kb == 8
    assert config.sram_pim.total_capacity_kb == 4096


def test_dram_config():
    config = load_config("configs")
    assert config.dram.model == "analytical"
    assert config.dram.effective_bandwidth_gbps == 25.6


def test_energy_config():
    config = load_config("configs")
    assert config.energy.sram.read_pj_per_access == 3.2
    assert config.energy.pim.mac_pj_per_op == 0.08
