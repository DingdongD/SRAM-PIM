from src.energy.destiny_adapter import DestinyAdapter, SRAMParams


def test_generate_sram_config():
    adapter = DestinyAdapter(destiny_dir="/home/Destiny-Memory-simulator")
    cfg = adapter.generate_config(tech_nm=28, capacity_kb=256, banks=32, word_bits=128)
    assert "-MemoryCellInputFile:" in cfg
    assert "SRAM" in cfg
    assert "-Capacity (KB): 256" in cfg


def test_parse_output():
    # Simulated DESTINY output
    sample = """
 - Read Latency (ns): 0.534
 - Write Latency (ns): 0.612
 - Read Dynamic Energy (pJ): 3.21
 - Write Dynamic Energy (pJ): 3.85
 - Leakage Power (mW): 12.5
 - Area (mm2): 0.42
"""
    adapter = DestinyAdapter(destiny_dir="/home/Destiny-Memory-simulator")
    params = adapter.parse_output(sample)
    assert isinstance(params, SRAMParams)
    assert abs(params.read_latency_ns - 0.534) < 0.01
    assert abs(params.read_energy_pj - 3.21) < 0.01
    assert abs(params.leakage_mw - 12.5) < 0.01


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
    # run() should always return SRAMParams (fallback if binary missing/fails)
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
