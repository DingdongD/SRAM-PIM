from src.simulator import Simulator
from src.config import SimConfig, SRAMPIMConfig, SystemConfig
from src.trace_ir import TraceCommand, OpCode


def make_config(spill_model="event_level"):
    return SimConfig(
        system=SystemConfig(
            frequency_hz=1_000_000_000,
            mode="cold_start",
            correctness_mode="strict",
            final_dirty_policy="ignore",
            spill_model=spill_model,
        ),
        sram_pim=SRAMPIMConfig(
            tiles=2, banks_per_tile=4,
            bank_capacity_kb=2, total_capacity_kb=16,
        ),
    )


def test_event_level_writeback_non_blocking():
    """Event-level writeback should return 0 latency (non-blocking)."""
    # Fill tile 0 completely, then force eviction of a dirty object
    cmds = [
        # Weight on tile 1
        TraceCommand(0, OpCode.SRAM_ALLOC, "W0", "-", "SRAM:T1:B0-1",
                     2048, {"pinned": True, "type": "WEIGHT"}, []),
        TraceCommand(1, OpCode.DMA_LOAD, "W0", "DRAM:0x0",
                     "SRAM:T1:B0-1", 2048, {}, [0]),
        # Input X0 on tile 0 banks 2-3
        TraceCommand(2, OpCode.SRAM_ALLOC, "X0", "-", "SRAM:T0:B2-3",
                     4096, {"type": "ACTIVATION"}, []),
        TraceCommand(3, OpCode.DMA_LOAD, "X0", "DRAM:0x1000",
                     "SRAM:T0:B2-3", 4096, {}, [2]),
        # PIM produces D0 on banks 0-1 (dirty)
        TraceCommand(4, OpCode.PIM_MAC, "D0", "W0,X0", "SRAM:T0:B0-1",
                     0, {"mac_count": 64, "out_bytes": 4096,
                         "out_type": "OUTPUT"}, [1, 3]),
        # tile 0 now full: D0(dirty,0-1) + X0(clean,2-3)
        # PIM output Y0 needs banks 0-1 → evicts D0 (dirty → event-level writeback)
        TraceCommand(5, OpCode.PIM_MAC, "Y0", "W0,X0", "SRAM:T0:B0-1",
                     0, {"mac_count": 64, "out_bytes": 4096,
                         "out_type": "OUTPUT"}, [4]),
        TraceCommand(6, OpCode.DMA_STORE, "Y0", "SRAM:T0:B0-1",
                     "DRAM:0xA000", 4096, {}, [5]),
    ]
    sim = Simulator(make_config(spill_model="event_level"))
    sim.load_trace(cmds)
    report = sim.run()

    assert report["memory_lifecycle"]["spill_count"] >= 1
    assert report["memory_lifecycle"]["eviction_count"] >= 1
    # Event-level writeback should create internal DMA_STORE events
    assert report["latency"]["dram_store_cycles"] > 0
    assert report["valid_simulation"] is True


def test_event_level_vs_blocking_latency():
    """Event-level writeback should have lower or equal effective latency than blocking."""
    # Same trace, compare blocking vs event-level
    cmds = [
        TraceCommand(0, OpCode.SRAM_ALLOC, "W0", "-", "SRAM:T1:B0-1",
                     2048, {"pinned": True, "type": "WEIGHT"}, []),
        TraceCommand(1, OpCode.DMA_LOAD, "W0", "DRAM:0x0",
                     "SRAM:T1:B0-1", 2048, {}, [0]),
        TraceCommand(2, OpCode.SRAM_ALLOC, "X0", "-", "SRAM:T0:B2-3",
                     4096, {"type": "ACTIVATION"}, []),
        TraceCommand(3, OpCode.DMA_LOAD, "X0", "DRAM:0x1000",
                     "SRAM:T0:B2-3", 4096, {}, [2]),
        TraceCommand(4, OpCode.PIM_MAC, "D0", "W0,X0", "SRAM:T0:B0-1",
                     0, {"mac_count": 64, "out_bytes": 4096,
                         "out_type": "OUTPUT"}, [1, 3]),
        TraceCommand(5, OpCode.PIM_MAC, "Y0", "W0,X0", "SRAM:T0:B0-1",
                     0, {"mac_count": 64, "out_bytes": 4096,
                         "out_type": "OUTPUT"}, [4]),
        TraceCommand(6, OpCode.DMA_STORE, "Y0", "SRAM:T0:B0-1",
                     "DRAM:0xA000", 4096, {}, [5]),
    ]

    sim_blocking = Simulator(make_config(spill_model="blocking"))
    sim_blocking.load_trace(cmds)
    report_blocking = sim_blocking.run()

    sim_event = Simulator(make_config(spill_model="event_level"))
    sim_event.load_trace(cmds)
    report_event = sim_event.run()

    # Event-level should overlap writeback with compute → fewer or equal total cycles
    assert report_event["latency"]["total_cycles"] <= report_blocking["latency"]["total_cycles"]
