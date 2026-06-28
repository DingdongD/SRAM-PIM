import pytest
from src.simulator import Simulator
from src.config import SimConfig, SRAMPIMConfig, SystemConfig
from src.trace_ir import TraceCommand, OpCode
from src.memory_object import ObjType


def make_config(final_dirty_policy="ignore"):
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


def _base_weight_trace():
    """Common prefix: allocate and load a weight."""
    return [
        TraceCommand(0, OpCode.SRAM_ALLOC, "W0", "-", "SRAM:T0:B0-3",
                     8192, {"pinned": True, "type": "WEIGHT"}, []),
        TraceCommand(1, OpCode.DMA_LOAD, "W0", "DRAM:0x0",
                     "SRAM:T0:B0-3", 8192, {}, [0]),
    ]


def test_pim_mac_accumulate_outputs_psum():
    """PIM_MAC with accumulate=True should default to PSUM type."""
    cmds = _base_weight_trace() + [
        TraceCommand(2, OpCode.PIM_MAC, "Y0", "W0", "SRAM:T1:B0-1",
                     0, {"mac_count": 64, "out_bytes": 2048,
                         "accumulate": True}, [1]),
        TraceCommand(3, OpCode.DMA_STORE, "Y0", "SRAM:T1:B0-1",
                     "DRAM:0x8000", 2048, {}, [2]),
    ]
    sim = Simulator(make_config())
    sim.load_trace(cmds)
    report = sim.run()
    obj = sim.mem_mgr.objects["Y0"]
    assert obj.obj_type == ObjType.PSUM


def test_pim_ew_op_outputs_state():
    """PIM_EW_OP with out_type=STATE should produce STATE-typed output."""
    cmds = _base_weight_trace() + [
        TraceCommand(2, OpCode.SRAM_ALLOC, "S0", "-", "SRAM:T1:B0-1",
                     2048, {"type": "ACTIVATION"}, []),
        TraceCommand(3, OpCode.DMA_LOAD, "S0", "DRAM:0x2000",
                     "SRAM:T1:B0-1", 2048, {}, [2]),
        TraceCommand(4, OpCode.PIM_EW_OP, "S_out", "S0", "SRAM:T1:B2-3",
                     0, {"ew_count": 128, "out_bytes": 2048,
                         "out_type": "STATE"}, [3]),
        TraceCommand(5, OpCode.DMA_STORE, "S_out", "SRAM:T1:B2-3",
                     "DRAM:0x4000", 2048, {}, [4]),
    ]
    sim = Simulator(make_config())
    sim.load_trace(cmds)
    report = sim.run()
    obj = sim.mem_mgr.objects["S_out"]
    assert obj.obj_type == ObjType.STATE
    assert report["valid_simulation"] is True


def test_pim_nl_outputs_activation():
    """PIM_NL with out_type=ACTIVATION should not be mistyped as PSUM."""
    cmds = _base_weight_trace() + [
        TraceCommand(2, OpCode.SRAM_ALLOC, "X0", "-", "SRAM:T1:B0-1",
                     2048, {"type": "ACTIVATION"}, []),
        TraceCommand(3, OpCode.DMA_LOAD, "X0", "DRAM:0x1000",
                     "SRAM:T1:B0-1", 2048, {}, [2]),
        TraceCommand(4, OpCode.PIM_NL, "NL_out", "X0", "SRAM:T1:B2-3",
                     0, {"nl_count": 128, "out_bytes": 2048,
                         "out_type": "ACTIVATION"}, [3]),
        TraceCommand(5, OpCode.DMA_STORE, "NL_out", "SRAM:T1:B2-3",
                     "DRAM:0x5000", 2048, {}, [4]),
    ]
    sim = Simulator(make_config())
    sim.load_trace(cmds)
    report = sim.run()
    obj = sim.mem_mgr.objects["NL_out"]
    assert obj.obj_type == ObjType.ACTIVATION
    assert report["valid_simulation"] is True


def test_final_dirty_identifies_state_type():
    """Final dirty policy should correctly identify STATE-typed dirty objects."""
    cmds = _base_weight_trace() + [
        TraceCommand(2, OpCode.SRAM_ALLOC, "S0", "-", "SRAM:T1:B0-1",
                     2048, {"type": "ACTIVATION"}, []),
        TraceCommand(3, OpCode.DMA_LOAD, "S0", "DRAM:0x2000",
                     "SRAM:T1:B0-1", 2048, {}, [2]),
        TraceCommand(4, OpCode.PIM_EW_OP, "S_dirty", "S0", "SRAM:T1:B2-3",
                     0, {"ew_count": 128, "out_bytes": 2048,
                         "out_type": "STATE"}, [3]),
        # No DMA_STORE — S_dirty is left dirty
    ]
    sim = Simulator(make_config(final_dirty_policy="report"))
    sim.load_trace(cmds)
    report = sim.run()
    dirty_list = report["final_state"]["final_dirty_objects"]
    assert len(dirty_list) >= 1
    dirty_types = [d["type"] for d in dirty_list]
    assert "STATE" in dirty_types
