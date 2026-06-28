import pytest
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


def test_strict_unresident_input_raises():
    """strict mode must raise when PIM reads an evicted object."""
    cmds = [
        TraceCommand(0, OpCode.SRAM_ALLOC, "W0", "-", "SRAM:T0:B0-1",
                     2048, {"pinned": True, "type": "WEIGHT"}, []),
        TraceCommand(1, OpCode.DMA_LOAD, "W0", "DRAM:0x0",
                     "SRAM:T0:B0-1", 2048, {}, [0]),
        TraceCommand(2, OpCode.SRAM_ALLOC, "X0", "-", "SRAM:T0:B2-3",
                     2048, {"type": "ACTIVATION"}, []),
        TraceCommand(3, OpCode.DMA_LOAD, "X0", "DRAM:0x1000",
                     "SRAM:T0:B2-3", 2048, {}, [2]),
        # Produce Y0 dirty on tile 1
        TraceCommand(4, OpCode.PIM_MAC, "Y0", "W0,X0", "SRAM:T1:B0-1",
                     0, {"mac_count": 64, "out_bytes": 2048}, [1, 3]),
        # Free X0 to create an evicted object
        TraceCommand(5, OpCode.DMA_STORE, "Y0", "SRAM:T1:B0-1",
                     "DRAM:0xA000", 2048, {}, [4]),
        TraceCommand(6, OpCode.SRAM_FREE, "X0", "-", "-", 0, {}, [5]),
        # Try to use X0 again after free — it's no longer in SRAM
        TraceCommand(7, OpCode.PIM_MAC, "Y1", "W0,X0", "SRAM:T1:B2-3",
                     0, {"mac_count": 64, "out_bytes": 2048}, [6]),
    ]
    sim = Simulator(make_config(correctness_mode="strict"))
    sim.load_trace(cmds)
    with pytest.raises(RuntimeError, match="not valid in SRAM"):
        sim.run()


def test_auto_reload_reads_from_dram():
    """auto_reload mode should reload evicted object from DRAM."""
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
        # Free X0, then try to reuse it — auto_reload should kick in
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


def test_auto_reload_invalid_dram_copy_raises():
    """auto_reload must raise if DRAM copy is also invalid."""
    cmds = [
        # Alloc X0 but never load from DRAM — valid_in_dram stays False
        TraceCommand(0, OpCode.SRAM_ALLOC, "W0", "-", "SRAM:T0:B0-1",
                     2048, {"pinned": True, "type": "WEIGHT"}, []),
        TraceCommand(1, OpCode.DMA_LOAD, "W0", "DRAM:0x0",
                     "SRAM:T0:B0-1", 2048, {}, [0]),
        # Produce X0 via PIM (dirty, never stored to DRAM)
        TraceCommand(2, OpCode.PIM_MAC, "X0", "W0", "SRAM:T0:B2-3",
                     0, {"mac_count": 32, "out_bytes": 1024,
                         "out_type": "ACTIVATION"}, [1]),
        # Free X0 (dirty — should get writeback in _issue_sram_free)
        # Actually: dirty free in strict raises, so let's store it first
        TraceCommand(3, OpCode.DMA_STORE, "X0", "SRAM:T0:B2-3",
                     "DRAM:0x5000", 1024, {}, [2]),
        TraceCommand(4, OpCode.SRAM_FREE, "X0", "-", "-", 0, {}, [3]),
        # Mark X0 as invalid in DRAM by modifying it was never truly
        # a DRAM-backed object — but DMA_STORE makes valid_in_dram True.
        # So instead: create an object that was produced in SRAM only
        TraceCommand(5, OpCode.PIM_MAC, "Z0", "W0", "SRAM:T0:B2-3",
                     0, {"mac_count": 32, "out_bytes": 1024,
                         "out_type": "ACTIVATION"}, [4]),
        # Free Z0 without DMA_STORE — dirty
        # dirty_free in strict will raise... let's use auto_reload mode
        # In auto_reload, dirty free is checked differently. Let's just
        # test the no-DRAM-copy path differently.
    ]
    # Simplify: just create a missing object reference
    simple_cmds = [
        TraceCommand(0, OpCode.SRAM_ALLOC, "W0", "-", "SRAM:T0:B0-1",
                     2048, {"pinned": True, "type": "WEIGHT"}, []),
        TraceCommand(1, OpCode.DMA_LOAD, "W0", "DRAM:0x0",
                     "SRAM:T0:B0-1", 2048, {}, [0]),
        # Alloc X0, produce it via PIM (never loaded from DRAM)
        TraceCommand(2, OpCode.PIM_MAC, "X0", "W0", "SRAM:T0:B2-3",
                     0, {"mac_count": 32, "out_bytes": 1024,
                         "out_type": "ACTIVATION"}, [1]),
        # Store and free X0
        TraceCommand(3, OpCode.DMA_STORE, "X0", "SRAM:T0:B2-3",
                     "DRAM:0x5000", 1024, {}, [2]),
        TraceCommand(4, OpCode.SRAM_FREE, "X0", "-", "-", 0, {}, [3]),
    ]
    # After SRAM_FREE + DMA_STORE, X0 has valid_in_dram=True.
    # To test invalid DRAM copy, we need to manually manipulate state.
    # Use a unit-test approach: directly set up the state.
    from src.memory_object import MemoryObject, ObjType, ObjState
    from src.memory_manager import MemoryManager

    config = make_config(correctness_mode="auto_reload")
    sim = Simulator(config)

    # Manually register an object that's evicted with no DRAM copy
    obj = MemoryObject("X_no_dram", ObjType.ACTIVATION, 1024, "int8",
                       valid_in_dram=False, valid_in_sram=False)
    obj.state = ObjState.EVICTED
    sim.mem_mgr.register_object(obj)

    # Weight still needs proper setup
    w_cmds = [
        TraceCommand(0, OpCode.SRAM_ALLOC, "W0", "-", "SRAM:T0:B0-1",
                     2048, {"pinned": True, "type": "WEIGHT"}, []),
        TraceCommand(1, OpCode.DMA_LOAD, "W0", "DRAM:0x0",
                     "SRAM:T0:B0-1", 2048, {}, [0]),
        TraceCommand(2, OpCode.PIM_MAC, "Y0", "W0,X_no_dram", "SRAM:T1:B0-1",
                     0, {"mac_count": 32, "out_bytes": 1024}, [1]),
    ]
    sim.load_trace(w_cmds)
    with pytest.raises(RuntimeError, match="DRAM copy invalid"):
        sim.run()
