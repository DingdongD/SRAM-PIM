"""P2-03: Verify energy accounting separates MAC vs EW and includes SRAM energy."""

from src.energy_model import EnergyModel
from src.config import SRAMPIMConfig, EnergyConfig, DRAMConfig


def make_energy_model():
    sram_cfg = SRAMPIMConfig(tiles=4, banks_per_tile=8,
                              bank_capacity_kb=8, total_capacity_kb=256)
    energy_cfg = EnergyConfig()
    dram_cfg = DRAMConfig()
    return EnergyModel(energy_cfg, sram_cfg, dram_cfg, freq_hz=1_000_000_000)


def test_pim_mac_energy():
    model = make_energy_model()
    model.add_pim_mac(1000)
    breakdown = model.get_breakdown()
    assert breakdown["pim_mac_pj"] > 0


def test_pim_ew_energy_separate():
    """P1-06: EW_OP uses separate counter from MAC."""
    model = make_energy_model()
    model.add_pim_mac(100)
    model.add_pim_ew(100)
    breakdown = model.get_breakdown()
    assert breakdown["pim_mac_pj"] > 0
    assert breakdown["pim_ew_pj"] > 0
    assert breakdown["pim_mac_pj"] != breakdown["pim_ew_pj"]


def test_sram_read_write_energy():
    """P1-05: SRAM read/write energy tracked."""
    model = make_energy_model()
    model.add_sram_read(100)
    model.add_sram_write(100)
    breakdown = model.get_breakdown()
    assert breakdown["sram_read_pj"] > 0
    assert breakdown["sram_write_pj"] > 0


def test_dram_energy():
    model = make_energy_model()
    model.add_dram_read(4096)
    model.add_dram_write(2048)
    breakdown = model.get_breakdown()
    assert breakdown["dram_read_pj"] > 0
    assert breakdown["dram_write_pj"] > 0


def test_total_energy():
    model = make_energy_model()
    model.add_pim_mac(100)
    model.add_sram_read(50)
    model.add_dram_read(1024)
    breakdown = model.get_breakdown()
    assert breakdown["total_pj"] > 0
    assert breakdown["total_pj"] >= (breakdown["pim_mac_pj"]
                                      + breakdown["sram_read_pj"]
                                      + breakdown["dram_read_pj"])
