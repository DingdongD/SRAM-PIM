import pytest
from src.simulator import Simulator
from src.config import SimConfig, SRAMPIMConfig, SystemConfig
from src.trace_ir import TraceCommand, OpCode


def make_config(correctness_mode="strict", final_dirty_policy="report"):
    return SimConfig(
        system=SystemConfig(
            frequency_hz=1_000_000_000,
            mode="cold_start",
            correctness_mode=correctness_mode,
            final_dirty_policy=final_dirty_policy,
        ),
        sram_pim=SRAMPIMConfig(tiles=4, banks_per_tile=8,
                               bank_capacity_kb=8, total_capacity_kb=256),
    )


def _dirty_output_trace():
    """Produce a dirty PIM output with no DMA_STORE."""
    return [
        TraceCommand(0, OpCode.SRAM_ALLOC, "W0", "-", "SRAM:T0:B0-3",
                     8192, {"pinned": True, "type": "WEIGHT"}, []),
        TraceCommand(1, OpCode.DMA_LOAD, "W0", "DRAM:0x0",
                     "SRAM:T0:B0-3", 8192, {}, [0]),
        TraceCommand(2, OpCode.PIM_MAC, "Y0", "W0", "SRAM:T1:B0-1",
                     0, {"mac_count": 64, "out_bytes": 2048,
                         "out_type": "OUTPUT"}, [1]),
    ]


def test_final_dirty_output_error_policy_raises():
    """error policy + strict mode must raise on dirty persistent output."""
    sim = Simulator(make_config(
        correctness_mode="strict", final_dirty_policy="error"))
    sim.load_trace(_dirty_output_trace())
    with pytest.raises(RuntimeError, match="dirty persistent"):
        sim.run()


def test_final_dirty_auto_writeback_counts_energy():
    """auto_writeback policy should write dirty objects back to DRAM."""
    sim = Simulator(make_config(
        correctness_mode="strict", final_dirty_policy="auto_writeback"))
    sim.load_trace(_dirty_output_trace())
    report = sim.run()
    # Y0 (2048 bytes) should be written back
    assert report["traffic"]["dram_write_bytes"] >= 2048
    assert report["latency"]["final_writeback_cycles"] > 0
    assert report["final_state"]["final_dirty_objects"] == []
    assert report["valid_simulation"] is True


def test_final_dirty_report_policy_records_objects():
    """report policy records dirty objects but doesn't raise."""
    sim = Simulator(make_config(
        correctness_mode="strict", final_dirty_policy="report"))
    sim.load_trace(_dirty_output_trace())
    report = sim.run()
    assert len(report["final_state"]["final_dirty_objects"]) > 0
    assert report["correctness"]["final_dirty_objects"] > 0
    assert report["valid_simulation"] is False
