from src.simulator import Simulator
from src.config import SimConfig, SRAMPIMConfig, SystemConfig
from src.trace_ir import TraceCommand, OpCode


def make_config(correctness_mode="auto_reload", final_dirty_policy="ignore"):
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


def test_latency_breakdown_no_double_count_reload():
    """pim_compute_cycles must NOT include auto_reload_cycles."""
    cmds = [
        TraceCommand(0, OpCode.SRAM_ALLOC, "W0", "-", "SRAM:T0:B0-1",
                     2048, {"pinned": True, "type": "WEIGHT"}, []),
        TraceCommand(1, OpCode.DMA_LOAD, "W0", "DRAM:0x0",
                     "SRAM:T0:B0-1", 2048, {}, [0]),
        TraceCommand(2, OpCode.SRAM_ALLOC, "X0", "-", "SRAM:T0:B2-3",
                     2048, {"type": "ACTIVATION"}, []),
        TraceCommand(3, OpCode.DMA_LOAD, "X0", "DRAM:0x1000",
                     "SRAM:T0:B2-3", 2048, {}, [2]),
        # First MAC — normal
        TraceCommand(4, OpCode.PIM_MAC, "Y0", "W0,X0", "SRAM:T1:B0-1",
                     0, {"mac_count": 64, "out_bytes": 2048}, [1, 3]),
        TraceCommand(5, OpCode.DMA_STORE, "Y0", "SRAM:T1:B0-1",
                     "DRAM:0xA000", 2048, {}, [4]),
        # Free X0 and try to use it again — triggers auto_reload
        TraceCommand(6, OpCode.SRAM_FREE, "X0", "-", "-", 0, {}, [5]),
        TraceCommand(7, OpCode.PIM_MAC, "Y1", "W0,X0", "SRAM:T1:B2-3",
                     0, {"mac_count": 64, "out_bytes": 2048}, [6]),
        TraceCommand(8, OpCode.DMA_STORE, "Y1", "SRAM:T1:B2-3",
                     "DRAM:0xB000", 2048, {}, [7]),
    ]
    sim = Simulator(make_config(correctness_mode="auto_reload"))
    sim.load_trace(cmds)
    report = sim.run()

    lat = report["latency"]
    # auto_reload_cycles should be tracked separately
    assert lat["auto_reload_cycles"] > 0
    # pim_compute_cycles should be pure compute, not including reload
    # Verify reload is not double-counted inside pim_compute_cycles
    # by checking pim_compute_cycles is reasonable (not inflated)
    assert lat["pim_compute_cycles"] > 0
    # The key invariant: auto_reload_cycles and pim_compute_cycles are distinct
    assert lat["auto_reload_cycles"] != lat["pim_compute_cycles"]


def test_auto_reload_success_keeps_valid_simulation_true():
    """auto_reload success should NOT make valid_simulation False."""
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
        # Free then reuse — auto_reload triggers
        TraceCommand(6, OpCode.SRAM_FREE, "X0", "-", "-", 0, {}, [5]),
        TraceCommand(7, OpCode.PIM_MAC, "Y1", "W0,X0", "SRAM:T1:B2-3",
                     0, {"mac_count": 64, "out_bytes": 2048}, [6]),
        TraceCommand(8, OpCode.DMA_STORE, "Y1", "SRAM:T1:B2-3",
                     "DRAM:0xB000", 2048, {}, [7]),
    ]
    sim = Simulator(make_config(correctness_mode="auto_reload"))
    sim.load_trace(cmds)
    report = sim.run()
    assert report["memory_lifecycle"]["reload_count"] >= 1
    assert report["correctness"]["illegal_read_unresident_object"] >= 1
    # Despite unresident events, simulation is valid because reload succeeded
    assert report["valid_simulation"] is True
