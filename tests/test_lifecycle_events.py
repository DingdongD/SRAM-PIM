from src.simulator import Simulator
from src.config import SimConfig, SRAMPIMConfig, SystemConfig
from src.trace_ir import TraceCommand, OpCode


def make_config(correctness_mode="strict", final_dirty_policy="ignore"):
    return SimConfig(
        system=SystemConfig(
            frequency_hz=1_000_000_000,
            mode="cold_start",
            correctness_mode=correctness_mode,
            final_dirty_policy=final_dirty_policy,
        ),
        sram_pim=SRAMPIMConfig(
            tiles=2, banks_per_tile=4,
            bank_capacity_kb=2, total_capacity_kb=16,
        ),
    )


def test_lifecycle_events_in_report():
    """Report should contain lifecycle_events dict with unresident and reload counters."""
    cmds = [
        TraceCommand(0, OpCode.SRAM_ALLOC, "W0", "-", "SRAM:T0:B0-1",
                     2048, {"pinned": True, "type": "WEIGHT"}, []),
        TraceCommand(1, OpCode.DMA_LOAD, "W0", "DRAM:0x0",
                     "SRAM:T0:B0-1", 2048, {}, [0]),
        TraceCommand(2, OpCode.SRAM_ALLOC, "X0", "-", "SRAM:T0:B2-3",
                     2048, {"type": "ACTIVATION"}, []),
        TraceCommand(3, OpCode.DMA_LOAD, "X0", "DRAM:0x1000",
                     "SRAM:T0:B2-3", 2048, {}, [2]),
        TraceCommand(4, OpCode.PIM_MAC, "Y0", "W0,X0", "SRAM:T1:B0-1",
                     0, {"mac_count": 64, "out_bytes": 2048}, [1, 3]),
        TraceCommand(5, OpCode.DMA_STORE, "Y0", "SRAM:T1:B0-1",
                     "DRAM:0xA000", 2048, {}, [4]),
    ]
    sim = Simulator(make_config())
    sim.load_trace(cmds)
    report = sim.run()

    assert "lifecycle_events" in report
    le = report["lifecycle_events"]
    assert "unresident_input_events" in le
    assert "auto_reload_events" in le
    assert le["unresident_input_events"] == 0
    assert le["auto_reload_events"] == 0


def test_unresident_events_tracked_in_auto_reload():
    """In auto_reload mode, unresident accesses should be lifecycle events, not errors."""
    cmds = [
        TraceCommand(0, OpCode.SRAM_ALLOC, "W0", "-", "SRAM:T0:B0-1",
                     2048, {"pinned": True, "type": "WEIGHT"}, []),
        TraceCommand(1, OpCode.DMA_LOAD, "W0", "DRAM:0x0",
                     "SRAM:T0:B0-1", 2048, {}, [0]),
        TraceCommand(2, OpCode.SRAM_ALLOC, "X0", "-", "SRAM:T0:B2-3",
                     2048, {"type": "ACTIVATION"}, []),
        TraceCommand(3, OpCode.DMA_LOAD, "X0", "DRAM:0x1000",
                     "SRAM:T0:B2-3", 2048, {}, [2]),
        TraceCommand(4, OpCode.PIM_MAC, "Y0", "W0,X0", "SRAM:T1:B0-1",
                     0, {"mac_count": 64, "out_bytes": 2048}, [1, 3]),
        TraceCommand(5, OpCode.DMA_STORE, "Y0", "SRAM:T1:B0-1",
                     "DRAM:0xA000", 2048, {}, [4]),
        # Free X0 then reuse
        TraceCommand(6, OpCode.SRAM_FREE, "X0", "-", "-", 0, {}, [5]),
        TraceCommand(7, OpCode.PIM_MAC, "Y1", "W0,X0", "SRAM:T1:B2-3",
                     0, {"mac_count": 64, "out_bytes": 2048}, [6]),
        TraceCommand(8, OpCode.DMA_STORE, "Y1", "SRAM:T1:B2-3",
                     "DRAM:0xB000", 2048, {}, [7]),
    ]
    sim = Simulator(make_config(correctness_mode="auto_reload"))
    sim.load_trace(cmds)
    report = sim.run()

    le = report["lifecycle_events"]
    assert le["unresident_input_events"] >= 1
    assert le["auto_reload_events"] >= 1
    # Backward compat: correctness dict still has the alias
    assert report["correctness"]["illegal_read_unresident_object"] >= 1
    # But simulation is still valid in auto_reload mode
    assert report["valid_simulation"] is True
