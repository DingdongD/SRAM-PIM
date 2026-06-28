import pytest
from src.simulator import Simulator
from src.config import SimConfig, SRAMPIMConfig, SystemConfig


from src.trace_ir import TraceCommand, OpCode


def make_simple_config(correctness_mode="strict"):
    return SimConfig(
        system=SystemConfig(frequency_hz=1_000_000_000, mode="cold_start",
                            correctness_mode=correctness_mode),
        sram_pim=SRAMPIMConfig(tiles=4, banks_per_tile=8,
                               bank_capacity_kb=8, total_capacity_kb=256),
    )


def test_empty_trace():
    sim = Simulator(make_simple_config())
    sim.load_trace([])
    report = sim.run()
    assert report["latency"]["total_cycles"] == 0


def test_dma_load_then_pim_mac():
    """Basic: alloc -> load weight -> load activation -> PIM MAC -> store output"""
    cmds = [
        TraceCommand(0, OpCode.SRAM_ALLOC, "W0", "-", "SRAM:T0:B0-3",
                     8192, {"pinned": True, "type": "WEIGHT"}, []),
        TraceCommand(1, OpCode.DMA_LOAD, "W0", "DRAM:0x1000",
                     "SRAM:T0:B0-3", 8192, {}, [0]),
        TraceCommand(2, OpCode.SRAM_ALLOC, "X0", "-", "SRAM:T0:B4-7",
                     4096, {"type": "ACTIVATION"}, []),
        TraceCommand(3, OpCode.DMA_LOAD, "X0", "DRAM:0x9000",
                     "SRAM:T0:B4-7", 4096, {}, [2]),
        TraceCommand(4, OpCode.PIM_MAC, "Y0", "W0,X0", "SRAM:T1:B0-1",
                     0, {"mac_count": 1024}, [1, 3]),
        TraceCommand(5, OpCode.DMA_STORE, "Y0", "SRAM:T1:B0-1",
                     "DRAM:0xA000", 2048, {}, [4]),
    ]
    sim = Simulator(make_simple_config())
    sim.load_trace(cmds)
    report = sim.run()
    assert report["latency"]["total_cycles"] > 0
    assert report["energy"]["total_pj"] > 0
    assert report["traffic"]["dram_read_bytes"] == 8192 + 4096
    assert report["traffic"]["dram_write_bytes"] == 2048
    assert report["valid_simulation"] is True


def test_dependency_enforcement():
    """PIM_MAC cannot run before its DMA_LOAD dependency completes"""
    cmds = [
        TraceCommand(0, OpCode.SRAM_ALLOC, "W0", "-", "SRAM:T0:B0-3",
                     8192, {"pinned": True, "type": "WEIGHT"}, []),
        TraceCommand(1, OpCode.DMA_LOAD, "W0", "DRAM:0x1000",
                     "SRAM:T0:B0-3", 8192, {}, [0]),
        TraceCommand(2, OpCode.PIM_MAC, "Y0", "W0", "SRAM:T1:B0-1",
                     0, {"mac_count": 100}, [1]),
    ]
    sim = Simulator(make_simple_config())
    sim.load_trace(cmds)
    report = sim.run()
    assert report["latency"]["total_cycles"] > report["latency"]["dram_load_cycles"]


def test_unresident_read_strict_raises():
    """P0-02: In strict mode, reading non-resident object raises error."""
    cmds = [
        TraceCommand(0, OpCode.PIM_MAC, "Y0", "W0_missing", "SRAM:T0:B0-0",
                     0, {"mac_count": 100}, []),
    ]
    sim = Simulator(make_simple_config(correctness_mode="strict"))
    sim.load_trace(cmds)
    with pytest.raises(RuntimeError, match="missing input object"):
        sim.run()


def test_unresident_read_warn_mode():
    """In warn mode, non-resident read is counted but doesn't raise."""
    cmds = [
        TraceCommand(0, OpCode.PIM_MAC, "Y0", "W0_missing", "SRAM:T0:B0-0",
                     0, {"mac_count": 100}, []),
    ]
    sim = Simulator(make_simple_config(correctness_mode="warn"))
    sim.load_trace(cmds)
    report = sim.run()
    assert report["correctness"]["missing_input_object"] > 0
    assert report["valid_simulation"] is False


def test_warm_resident_mode():
    """In warm_resident mode, pre-loaded weights skip DMA cost"""
    cmds = [
        TraceCommand(0, OpCode.SRAM_ALLOC, "W0", "-", "SRAM:T0:B0-3",
                     8192, {"pinned": True, "type": "WEIGHT", "preloaded": True}, []),
        TraceCommand(1, OpCode.SRAM_ALLOC, "X0", "-", "SRAM:T1:B0-3",
                     4096, {"type": "ACTIVATION"}, []),
        TraceCommand(2, OpCode.DMA_LOAD, "X0", "DRAM:0x9000",
                     "SRAM:T1:B0-3", 4096, {}, [1]),
        TraceCommand(3, OpCode.PIM_MAC, "Y0", "W0,X0", "SRAM:T2:B0-1",
                     0, {"mac_count": 1024}, [0, 2]),
    ]
    config = make_simple_config()
    config.system.mode = "warm_resident"
    sim = Simulator(config)
    sim.load_trace(cmds)
    report = sim.run()
    assert report["traffic"]["dram_read_bytes"] == 4096


def test_alloc_not_valid():
    """P0-01: SRAM_ALLOC alone does not make data valid for PIM."""
    cmds = [
        TraceCommand(0, OpCode.SRAM_ALLOC, "X0", "-", "SRAM:T0:B0-1",
                     1024, {"type": "ACTIVATION"}, []),
        TraceCommand(1, OpCode.PIM_MAC, "Y0", "X0", "SRAM:T1:B0-1",
                     0, {"mac_count": 128}, [0]),
    ]
    sim = Simulator(make_simple_config(correctness_mode="strict"))
    sim.load_trace(cmds)
    with pytest.raises(RuntimeError, match="not valid in SRAM"):
        sim.run()


def test_dirty_free_strict_raises():
    """P0-05: Cannot free dirty object in strict mode."""
    cmds = [
        TraceCommand(0, OpCode.SRAM_ALLOC, "W0", "-", "SRAM:T0:B0-1",
                     1024, {"type": "WEIGHT", "preloaded": True}, []),
        TraceCommand(1, OpCode.SRAM_ALLOC, "X0", "-", "SRAM:T0:B2-3",
                     1024, {"type": "ACTIVATION"}, []),
        TraceCommand(2, OpCode.DMA_LOAD, "X0", "DRAM:0x1000",
                     "SRAM:T0:B2-3", 1024, {}, [1]),
        TraceCommand(3, OpCode.PIM_MAC, "Y0", "W0,X0", "SRAM:T1:B0-1",
                     0, {"mac_count": 64}, [0, 2]),
        # Try to free dirty output without DMA_STORE
        TraceCommand(4, OpCode.SRAM_FREE, "Y0", "SRAM:T1", "-", 0, {}, [3]),
    ]
    sim = Simulator(make_simple_config(correctness_mode="strict"))
    sim.load_trace(cmds)
    with pytest.raises(RuntimeError, match="Cannot free dirty"):
        sim.run()


def test_deadlock_strict_raises():
    """P0-09: Deadlock raises error in strict mode."""
    cmds = [
        TraceCommand(0, OpCode.PIM_MAC, "Y0", "-", "SRAM:T0:B0-0",
                     0, {"mac_count": 10}, [999]),  # dep on non-existent
    ]
    sim = Simulator(make_simple_config(correctness_mode="strict"))
    sim.load_trace(cmds)
    with pytest.raises(RuntimeError, match="Deadlock"):
        sim.run()


def test_issue_complete_separation():
    """P0-03: Data should not be valid until DMA_LOAD completes."""
    cmds = [
        TraceCommand(0, OpCode.SRAM_ALLOC, "X0", "-", "SRAM:T0:B0-1",
                     1024, {"type": "ACTIVATION"}, []),
        TraceCommand(1, OpCode.DMA_LOAD, "X0", "DRAM:0x1000",
                     "SRAM:T0:B0-1", 1024, {}, [0]),
        # PIM_MAC depends on DMA_LOAD, so it waits for completion
        TraceCommand(2, OpCode.PIM_MAC, "Y0", "X0", "SRAM:T1:B0-1",
                     0, {"mac_count": 64}, [1]),
    ]
    sim = Simulator(make_simple_config())
    sim.load_trace(cmds)
    report = sim.run()
    # PIM should complete after DMA
    assert report["latency"]["total_cycles"] > report["latency"]["dram_load_cycles"]
    assert report["valid_simulation"] is True
