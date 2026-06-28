from src.energy.destiny_adapter import DestinyAdapter, SRAMParams, apply_destiny_params
from src.config import SimConfig, SRAMPIMConfig, EnergyConfig, SRAMEnergyConfig


SAMPLE_DATA_ARRAY_OUTPUT = """
CACHE DATA ARRAY DETAILS
    =============
       RESULT
    =============
    Area:
     - Total Area = 979.676um x 350.503um = 343380um^2
    Timing:
     -  Read Latency = 180.838ps
     - Write Latency = 146.431ps
    Power:
     -  Read Dynamic Energy = 31.4234pJ
     - Write Dynamic Energy = 31.2745pJ
     - Leakage Power = 459.244mW
     - Read Bandwidth  = 135.844GB/s

CACHE TAG ARRAY DETAILS
"""

SAMPLE_SUMMARY_OUTPUT = """
=======================
CACHE DESIGN -- SUMMARY
=======================
Timing:
 - Cache Hit Latency   = 0.195973ns
 - Cache Write Latency = 0.146431ns
Power:
 - Cache Hit Dynamic Energy   = 0.0378698nJ per access
 - Cache Write Dynamic Energy = 0.0376833nJ per access
 |--- Cache Data Array Leakage Power = 459.244mW
Area:
 |--- Data Array Area = 979.676um x 350.503um = 0.34338mm^2
"""


def test_generate_sram_config():
    adapter = DestinyAdapter(destiny_dir="/home/Destiny-Memory-simulator")
    cfg = adapter.generate_config(tech_nm=28, capacity_kb=256, banks=32, word_bits=128)
    assert "-MemoryCellInputFile:" in cfg
    assert "SRAM" in cfg
    assert "-Capacity (KB): 256" in cfg


def test_parse_data_array_section():
    """Parse DATA ARRAY DETAILS with ps/pJ values."""
    adapter = DestinyAdapter(destiny_dir="/home/Destiny-Memory-simulator")
    params = adapter.parse_output(SAMPLE_DATA_ARRAY_OUTPUT)
    assert isinstance(params, SRAMParams)
    assert abs(params.read_latency_ns - 0.180838) < 0.001  # 180.838ps -> ns
    assert abs(params.write_latency_ns - 0.146431) < 0.001
    assert abs(params.read_energy_pj - 31.4234) < 0.01
    assert abs(params.write_energy_pj - 31.2745) < 0.01
    assert abs(params.leakage_mw - 459.244) < 0.01
    assert abs(params.area_mm2 - 0.34338) < 0.001
    assert abs(params.read_bandwidth_gbps - 135.844) < 0.1


def test_parse_summary_fallback():
    """When DATA ARRAY section absent, fall back to summary parsing."""
    adapter = DestinyAdapter(destiny_dir="/home/Destiny-Memory-simulator")
    params = adapter.parse_output(SAMPLE_SUMMARY_OUTPUT)
    assert abs(params.read_latency_ns - 0.195973) < 0.001
    assert abs(params.write_latency_ns - 0.146431) < 0.001
    # nJ -> pJ conversion
    assert abs(params.read_energy_pj - 37.8698) < 0.1
    assert abs(params.write_energy_pj - 37.6833) < 0.1
    assert abs(params.leakage_mw - 459.244) < 0.01


def test_default_params():
    params = SRAMParams()
    assert params.read_energy_pj > 0


def test_generate_config_tech_node():
    adapter = DestinyAdapter(destiny_dir="/home/Destiny-Memory-simulator")
    cfg = adapter.generate_config(tech_nm=65, capacity_kb=512, banks=16, word_bits=64)
    assert "-ProcessNode: 65" in cfg
    assert "-Capacity (KB): 512" in cfg
    assert "-WordWidth (bit): 64" in cfg


def test_run_returns_sram_params():
    adapter = DestinyAdapter(destiny_dir="/home/Destiny-Memory-simulator")
    params = adapter.run(tech_nm=28, capacity_kb=256, banks=32, word_bits=128)
    assert isinstance(params, SRAMParams)
    assert params.read_latency_ns > 0
    assert params.read_energy_pj > 0
    assert params.area_mm2 > 0


def test_sram_params_fields():
    p = SRAMParams(read_latency_ns=1.0, write_latency_ns=1.2,
                   read_energy_pj=5.0, write_energy_pj=5.5,
                   leakage_mw=20.0, area_mm2=0.8)
    assert p.write_latency_ns == 1.2
    assert p.write_energy_pj == 5.5
    assert p.leakage_mw == 20.0


def test_apply_destiny_params_skips_analytical():
    """When source is 'analytical', apply_destiny_params is a no-op."""
    config = SimConfig()
    config.energy.sram.source = "analytical"
    result = apply_destiny_params(config)
    assert result == "analytical"


def test_apply_destiny_params_updates_config():
    """When source is 'destiny', config should get DESTINY values."""
    config = SimConfig()
    config.energy.sram.source = "destiny"
    config.sram_pim.banks_per_tile = 8
    config.sram_pim.bank_capacity_kb = 8
    config.sram_pim.word_bytes = 16

    result = apply_destiny_params(config)
    # DESTINY binary exists at /home/Destiny-Memory-simulator
    assert result == "destiny"
    # Values should differ from defaults (3.2/3.8 pJ)
    assert config.energy.sram.read_pj_per_access != 3.2
    assert config.energy.sram.source == "destiny"
