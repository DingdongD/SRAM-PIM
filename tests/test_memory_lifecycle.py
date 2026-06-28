"""P2-03: Memory lifecycle tests — alloc, load, compute, store, free."""

from src.simulator import Simulator
from src.config import SimConfig, SystemConfig, SRAMPIMConfig
from src.trace_ir import TraceCommand, OpCode
import pytest


def make_config(mode="strict"):
    return SimConfig(
        system=SystemConfig(frequency_hz=1_000_000_000, mode="cold_start",
                            correctness_mode=mode),
        sram_pim=SRAMPIMConfig(tiles=4, banks_per_tile=8,
                               bank_capacity_kb=8, total_capacity_kb=256),
    )


def test_full_lifecycle():
    """alloc -> load -> PIM -> store -> free: no errors."""
    cmds = [
        TraceCommand(0, OpCode.SRAM_ALLOC, "W0", "-", "SRAM:T0:B0-1",
                     1024, {"pinned": True, "type": "WEIGHT"}, []),
        TraceCommand(1, OpCode.DMA_LOAD, "W0", "DRAM:0x1000", "SRAM:T0:B0-1",
                     1024, {}, [0]),
        TraceCommand(2, OpCode.SRAM_ALLOC, "X0", "-", "SRAM:T0:B2-3",
                     512, {"type": "ACTIVATION"}, []),
        TraceCommand(3, OpCode.DMA_LOAD, "X0", "DRAM:0x2000", "SRAM:T0:B2-3",
                     512, {}, [2]),
        TraceCommand(4, OpCode.PIM_MAC, "Y0", "W0,X0", "SRAM:T1:B0-1",
                     0, {"mac_count": 256}, [1, 3]),
        TraceCommand(5, OpCode.DMA_STORE, "Y0", "SRAM:T1:B0-1", "DRAM:0x3000",
                     256, {}, [4]),
        TraceCommand(6, OpCode.SRAM_FREE, "Y0", "SRAM:T1", "-", 0, {}, [5]),
    ]
    sim = Simulator(make_config())
    sim.load_trace(cmds)
    report = sim.run()
    assert report["valid_simulation"] is True
    assert report["correctness"]["missing_input_object"] == 0


def test_alloc_without_load_then_pim_raises():
    """P0-01: alloc without DMA_LOAD -> PIM should raise in strict."""
    cmds = [
        TraceCommand(0, OpCode.SRAM_ALLOC, "X0", "-", "SRAM:T0:B0-1",
                     1024, {"type": "ACTIVATION"}, []),
        TraceCommand(1, OpCode.PIM_MAC, "Y0", "X0", "SRAM:T1:B0-1",
                     0, {"mac_count": 128}, [0]),
    ]
    sim = Simulator(make_config("strict"))
    sim.load_trace(cmds)
    with pytest.raises(RuntimeError, match="not valid in SRAM"):
        sim.run()


def test_power_gate_dirty_raises():
    """P1-07: Power-gating dirty object should raise in strict."""
    cmds = [
        TraceCommand(0, OpCode.SRAM_ALLOC, "W0", "-", "SRAM:T0:B0-1",
                     1024, {"pinned": True, "type": "WEIGHT", "preloaded": True}, []),
        TraceCommand(1, OpCode.SRAM_ALLOC, "X0", "-", "SRAM:T0:B2-3",
                     512, {"type": "ACTIVATION"}, []),
        TraceCommand(2, OpCode.DMA_LOAD, "X0", "DRAM:0x1000", "SRAM:T0:B2-3",
                     512, {}, [1]),
        TraceCommand(3, OpCode.PIM_MAC, "Y0", "W0,X0", "SRAM:T1:B0-1",
                     0, {"mac_count": 64}, [0, 2]),
        # Power-gate dirty Y0 without store
        TraceCommand(4, OpCode.POWER_SET, "Y0", "-", "-", 0,
                     {"state": "power_gated"}, [3]),
    ]
    sim = Simulator(make_config("strict"))
    sim.load_trace(cmds)
    with pytest.raises(RuntimeError, match="power-gate dirty"):
        sim.run()
