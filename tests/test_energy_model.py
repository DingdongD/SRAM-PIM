from src.energy_model import EnergyModel
from src.config import EnergyConfig, SRAMPIMConfig, DRAMConfig


def test_dram_energy():
    energy = EnergyModel(EnergyConfig(), SRAMPIMConfig(), DRAMConfig(), freq_hz=1_000_000_000)
    energy.add_dram_read(1000)
    bd = energy.get_breakdown()
    # 1000 * 15.0 = 15000 pJ read + 1000 * 8.0 = 8000 pJ io
    assert bd["dram_read_pj"] == 15000.0
    assert bd["offchip_io_pj"] == 8000.0


def test_sram_energy():
    energy = EnergyModel(EnergyConfig(), SRAMPIMConfig(), DRAMConfig(), freq_hz=1_000_000_000)
    energy.add_sram_read(100)
    energy.add_sram_write(50)
    bd = energy.get_breakdown()
    assert bd["sram_read_pj"] == 100 * 3.2
    assert bd["sram_write_pj"] == 50 * 3.8


def test_pim_energy():
    energy = EnergyModel(EnergyConfig(), SRAMPIMConfig(), DRAMConfig(), freq_hz=1_000_000_000)
    energy.add_pim_mac(1000)
    energy.add_pim_reduce(500)
    bd = energy.get_breakdown()
    assert bd["pim_mac_pj"] == 1000 * 0.08
    assert bd["pim_reduce_pj"] == 500 * 0.04


def test_leakage_energy():
    cfg = EnergyConfig()
    sram_cfg = SRAMPIMConfig(tiles=2, banks_per_tile=4)
    energy = EnergyModel(cfg, sram_cfg, DRAMConfig(), freq_hz=1_000_000_000)
    energy.add_leakage(1000)  # 1000 cycles at 1GHz = 1us
    bd = energy.get_breakdown()
    # 8 banks * 0.39 mW * 1us = 8 * 0.39 * 1e-3 * 1e-6 W*s = 8 * 0.39e-9 J = 3.12e-9 J = 3120 pJ
    assert abs(bd["sram_leakage_pj"] - 3120.0) < 1.0


def test_total_energy():
    energy = EnergyModel(EnergyConfig(), SRAMPIMConfig(), DRAMConfig(), freq_hz=1_000_000_000)
    energy.add_dram_read(100)
    energy.add_pim_mac(100)
    bd = energy.get_breakdown()
    assert bd["total_pj"] > 0


def test_dram_write_energy():
    energy = EnergyModel(EnergyConfig(), SRAMPIMConfig(), DRAMConfig(), freq_hz=1_000_000_000)
    energy.add_dram_write(500)
    bd = energy.get_breakdown()
    assert bd["dram_write_pj"] == 500 * 18.0
    assert bd["offchip_io_pj"] == 500 * 8.0


def test_noc_energy():
    energy = EnergyModel(EnergyConfig(), SRAMPIMConfig(), DRAMConfig(), freq_hz=1_000_000_000)
    energy.add_noc(200)
    bd = energy.get_breakdown()
    # 200 bytes * 0.2 pJ/byte/hop * 2 hops = 80 pJ
    assert bd["noc_pj"] == 200 * 0.2 * 2


def test_reset():
    energy = EnergyModel(EnergyConfig(), SRAMPIMConfig(), DRAMConfig(), freq_hz=1_000_000_000)
    energy.add_dram_read(1000)
    energy.add_pim_mac(500)
    energy.reset()
    bd = energy.get_breakdown()
    assert bd["total_pj"] == 0.0
    assert bd["dram_read_pj"] == 0.0
