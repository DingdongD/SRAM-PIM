from src.simulator import Simulator
from src.config import SimConfig, SRAMPIMConfig, SystemConfig
from src.trace_ir import TraceCommand, OpCode


def make_config(final_dirty_policy="auto_writeback"):
    return SimConfig(
        system=SystemConfig(
            frequency_hz=1_000_000_000,
            mode="cold_start",
            correctness_mode="strict",
            final_dirty_policy=final_dirty_policy,
        ),
        sram_pim=SRAMPIMConfig(tiles=4, banks_per_tile=8,
                               bank_capacity_kb=8, total_capacity_kb=256),
    )


def _dirty_output_trace():
    return [
        TraceCommand(0, OpCode.SRAM_ALLOC, "W0", "-", "SRAM:T0:B0-3",
                     8192, {"pinned": True, "type": "WEIGHT"}, []),
        TraceCommand(1, OpCode.DMA_LOAD, "W0", "DRAM:0x0",
                     "SRAM:T0:B0-3", 8192, {}, [0]),
        TraceCommand(2, OpCode.PIM_MAC, "Y0", "W0", "SRAM:T1:B0-1",
                     0, {"mac_count": 64, "out_bytes": 2048,
                         "out_type": "OUTPUT"}, [1]),
        # No DMA_STORE — Y0 remains dirty
    ]


def test_final_auto_writeback_not_counted_as_spill():
    """Final auto-writeback should increment final_writeback_count, not spill_count."""
    sim = Simulator(make_config(final_dirty_policy="auto_writeback"))
    sim.load_trace(_dirty_output_trace())
    report = sim.run()

    # spill_count should be 0 (no capacity-driven spills happened)
    assert report["memory_lifecycle"]["spill_count"] == 0
    # final_writeback_count should be > 0
    assert report["memory_lifecycle"]["final_writeback_count"] >= 1
    # final_writeback_cycles should be tracked
    assert report["latency"]["final_writeback_cycles"] > 0
    # simulation should be valid (auto_writeback handles dirty objects)
    assert report["valid_simulation"] is True


def test_final_writeback_includes_leakage():
    """Final auto-writeback period should include leakage energy."""
    sim = Simulator(make_config(final_dirty_policy="auto_writeback"))
    sim.load_trace(_dirty_output_trace())
    report = sim.run()

    # Leakage energy should be positive (includes final writeback period)
    assert report["energy"]["sram_leakage_pj"] > 0
    assert report["latency"]["final_writeback_cycles"] > 0
