import subprocess
import json
import yaml
from src.config import load_config, SimConfig, SystemConfig, SRAMPIMConfig
from src.tracegen.gen_gemm_trace import gen_gemm_trace
from src.simulator import Simulator
from src.report import generate_report


def test_gemm_e2e_pipeline():
    config = SimConfig(
        system=SystemConfig(frequency_hz=1_000_000_000, mode="cold_start"),
        sram_pim=SRAMPIMConfig(tiles=4, banks_per_tile=8, bank_capacity_kb=8, total_capacity_kb=256),
    )
    cmds = gen_gemm_trace(M=128, N=128, K=128, Tm=64, Tn=64, Tk=128,
                          sram_config=config.sram_pim, mode="cold_start")
    sim = Simulator(config)
    sim.load_trace(cmds)
    result = sim.run()
    report = generate_report(result, config)
    assert "total_cycles" in report
    assert "total_pj" in report
    assert "dram_read_bytes" in report


def test_report_yaml_parseable():
    config = SimConfig(
        system=SystemConfig(frequency_hz=1_000_000_000, mode="cold_start"),
        sram_pim=SRAMPIMConfig(tiles=4, banks_per_tile=8, bank_capacity_kb=8, total_capacity_kb=256),
    )
    cmds = gen_gemm_trace(M=64, N=64, K=64, Tm=64, Tn=64, Tk=64,
                          sram_config=config.sram_pim, mode="cold_start")
    sim = Simulator(config)
    sim.load_trace(cmds)
    result = sim.run()
    report = generate_report(result, config)
    parsed = yaml.safe_load(report)
    assert parsed["latency"]["total_cycles"] > 0
    assert parsed["energy"]["total_pj"] > 0
    assert parsed["correctness"]["illegal_read_unresident_object"] == 0


def test_cli_smoke():
    result = subprocess.run(
        ["python", "main.py", "--workload", "gemm",
         "--M", "64", "--N", "64", "--K", "64", "--mode", "cold_start"],
        capture_output=True, text=True, cwd="/home/sram_pim"
    )
    assert result.returncode == 0, result.stderr
    assert "total_cycles" in result.stdout
    assert "total_pj" in result.stdout
