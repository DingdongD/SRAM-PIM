from src.simulator import Simulator
from src.config import SimConfig, SRAMPIMConfig, SystemConfig
from src.trace_ir import TraceCommand, OpCode
from src.report import generate_report
import yaml


def make_config():
    return SimConfig(
        system=SystemConfig(
            frequency_hz=1_000_000_000,
            mode="cold_start",
            correctness_mode="strict",
            final_dirty_policy="report",
        ),
        sram_pim=SRAMPIMConfig(tiles=4, banks_per_tile=8,
                               bank_capacity_kb=8, total_capacity_kb=256),
    )


def _run_simple_trace():
    cmds = [
        TraceCommand(0, OpCode.SRAM_ALLOC, "W0", "-", "SRAM:T0:B0-3",
                     8192, {"pinned": True, "type": "WEIGHT"}, []),
        TraceCommand(1, OpCode.DMA_LOAD, "W0", "DRAM:0x0",
                     "SRAM:T0:B0-3", 8192, {}, [0]),
        TraceCommand(2, OpCode.PIM_MAC, "Y0", "W0", "SRAM:T1:B0-1",
                     0, {"mac_count": 64, "out_bytes": 2048,
                         "out_type": "OUTPUT"}, [1]),
        TraceCommand(3, OpCode.DMA_STORE, "Y0", "SRAM:T1:B0-1",
                     "DRAM:0x8000", 2048, {}, [2]),
    ]
    config = make_config()
    sim = Simulator(config)
    sim.load_trace(cmds)
    return sim.run(), config


def test_report_has_final_state():
    """Report must include final_state section."""
    report, _ = _run_simple_trace()
    assert "final_state" in report
    fs = report["final_state"]
    assert "final_dirty_policy" in fs
    assert "final_dirty_objects" in fs
    assert "final_resident_objects" in fs


def test_report_has_model_provenance():
    """Report must include model_provenance section."""
    report, _ = _run_simple_trace()
    assert "model_provenance" in report
    mp = report["model_provenance"]
    assert mp["correctness_mode"] == "strict"
    assert mp["spill_model"] == "blocking"
    assert mp["dram_model"] == "analytical"
    assert mp["sram_param_source"] == "analytical"
    assert "workload_trace_semantics" in mp


def test_yaml_report_includes_new_sections():
    """YAML report via generate_report must include final_state and provenance."""
    report, config = _run_simple_trace()
    yaml_str = generate_report(report, config)
    parsed = yaml.safe_load(yaml_str)
    assert "final_state" in parsed
    assert "model_provenance" in parsed
    assert parsed["model_provenance"]["correctness_mode"] == "strict"
