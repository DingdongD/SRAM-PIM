from src.trace_validator import TraceValidator
from src.trace_ir import TraceCommand, OpCode


def test_valid_trace_no_errors():
    cmds = [
        TraceCommand(0, OpCode.SRAM_ALLOC, "W0", "-", "SRAM:T0:B0-1", 1024,
                     {"pinned": True, "type": "WEIGHT", "preloaded": True}, []),
        TraceCommand(1, OpCode.SRAM_ALLOC, "X0", "-", "SRAM:T0:B2-3", 1024,
                     {"type": "ACTIVATION"}, []),
        TraceCommand(2, OpCode.DMA_LOAD, "X0", "DRAM:0x1000", "SRAM:T0:B2-3",
                     1024, {}, [1]),
        TraceCommand(3, OpCode.PIM_MAC, "Y0", "W0,X0", "SRAM:T1:B0-1",
                     0, {"mac_count": 64}, [0, 2]),
        TraceCommand(4, OpCode.DMA_STORE, "Y0", "SRAM:T1:B0-1", "DRAM:0x2000",
                     256, {}, [3]),
        TraceCommand(5, OpCode.SRAM_FREE, "Y0", "SRAM:T1", "-", 0, {}, [4]),
    ]
    errors = TraceValidator().validate(cmds)
    assert len([e for e in errors if e.severity == "error"]) == 0


def test_unknown_dependency():
    cmds = [
        TraceCommand(0, OpCode.PIM_MAC, "Y0", "-", "SRAM:T0:B0-0",
                     0, {"mac_count": 10}, [999]),
    ]
    errors = TraceValidator().validate(cmds)
    assert any(e.severity == "error" and "Unknown dependency" in e.message
               for e in errors)


def test_missing_producer():
    cmds = [
        TraceCommand(0, OpCode.PIM_MAC, "Y0", "W_missing", "SRAM:T0:B0-0",
                     0, {"mac_count": 10}, []),
    ]
    errors = TraceValidator().validate(cmds)
    assert any("no producer" in e.message for e in errors)


def test_dirty_free_warning():
    cmds = [
        TraceCommand(0, OpCode.SRAM_ALLOC, "W0", "-", "SRAM:T0:B0-1", 1024,
                     {"pinned": True, "type": "WEIGHT", "preloaded": True}, []),
        TraceCommand(1, OpCode.SRAM_ALLOC, "X0", "-", "SRAM:T0:B2-3", 1024,
                     {"type": "ACTIVATION"}, []),
        TraceCommand(2, OpCode.DMA_LOAD, "X0", "DRAM:0x1000", "SRAM:T0:B2-3",
                     1024, {}, [1]),
        TraceCommand(3, OpCode.PIM_MAC, "Y0", "W0,X0", "SRAM:T1:B0-1",
                     0, {"mac_count": 64}, [0, 2]),
        # Free dirty Y0 without DMA_STORE
        TraceCommand(4, OpCode.SRAM_FREE, "Y0", "SRAM:T1", "-", 0, {}, [3]),
    ]
    errors = TraceValidator().validate(cmds)
    assert any("dirty" in e.message.lower() for e in errors)


def test_cycle_detection():
    cmds = [
        TraceCommand(0, OpCode.PIM_MAC, "Y0", "-", "SRAM:T0:B0-0",
                     0, {"mac_count": 10}, [1]),
        TraceCommand(1, OpCode.PIM_MAC, "Y1", "-", "SRAM:T0:B1-1",
                     0, {"mac_count": 10}, [0]),
    ]
    errors = TraceValidator().validate(cmds)
    assert any("cycle" in e.message.lower() for e in errors)


def test_waw_hazard():
    """Two writers to same object without dependency should warn."""
    cmds = [
        TraceCommand(0, OpCode.SRAM_ALLOC, "W0", "-", "SRAM:T0:B0-1", 1024,
                     {"pinned": True, "type": "WEIGHT", "preloaded": True}, []),
        TraceCommand(1, OpCode.SRAM_ALLOC, "X0", "-", "SRAM:T0:B2-3", 1024,
                     {"type": "ACTIVATION"}, []),
        TraceCommand(2, OpCode.DMA_LOAD, "X0", "DRAM:0x1000", "SRAM:T0:B2-3",
                     1024, {}, [1]),
        # Two MACs write to same Y0 without chaining
        TraceCommand(3, OpCode.PIM_MAC, "Y0", "W0,X0", "SRAM:T1:B0-1",
                     0, {"mac_count": 64}, [0, 2]),
        TraceCommand(4, OpCode.PIM_MAC, "Y0", "W0,X0", "SRAM:T1:B0-1",
                     0, {"mac_count": 64}, [0, 2]),
    ]
    errors = TraceValidator().validate(cmds)
    assert any("WAW" in e.message for e in errors)
