import pytest
from src.simulator import Simulator
from src.config import SimConfig, SRAMPIMConfig, SystemConfig
from src.trace_ir import TraceCommand, OpCode


def make_tiny_config(correctness_mode="strict", final_dirty_policy="ignore"):
    """Small SRAM to easily trigger capacity pressure."""
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


def test_pim_output_allocation_triggers_spill():
    """When SRAM is full, PIM output allocation must trigger spill/eviction."""
    cmds = [
        # Fill tile 0 with a clean object (victim)
        TraceCommand(0, OpCode.SRAM_ALLOC, "V0", "-", "SRAM:T0:B0-3",
                     8192, {"type": "ACTIVATION"}, []),
        TraceCommand(1, OpCode.DMA_LOAD, "V0", "DRAM:0x0",
                     "SRAM:T0:B0-3", 8192, {}, [0]),
        # Alloc weight on tile 1
        TraceCommand(2, OpCode.SRAM_ALLOC, "W0", "-", "SRAM:T1:B0-3",
                     4096, {"pinned": True, "type": "WEIGHT"}, []),
        TraceCommand(3, OpCode.DMA_LOAD, "W0", "DRAM:0x4000",
                     "SRAM:T1:B0-3", 4096, {}, [2]),
        # PIM output on tile 0 — must evict V0 to make room
        TraceCommand(4, OpCode.PIM_MAC, "Y0", "W0", "SRAM:T0:B0-1",
                     0, {"mac_count": 64, "out_bytes": 2048}, [1, 3]),
        TraceCommand(5, OpCode.DMA_STORE, "Y0", "SRAM:T0:B0-1",
                     "DRAM:0x8000", 2048, {}, [4]),
    ]
    sim = Simulator(make_tiny_config())
    sim.load_trace(cmds)
    report = sim.run()
    assert report["memory_lifecycle"]["eviction_count"] > 0
    assert report["valid_simulation"] is True


def test_pim_output_allocation_dirty_victim_writeback():
    """Dirty victim must be written back to DRAM during spill."""
    cmds = [
        # Weight on tile 1
        TraceCommand(0, OpCode.SRAM_ALLOC, "W0", "-", "SRAM:T1:B0-1",
                     2048, {"pinned": True, "type": "WEIGHT"}, []),
        TraceCommand(1, OpCode.DMA_LOAD, "W0", "DRAM:0x0",
                     "SRAM:T1:B0-1", 2048, {}, [0]),
        # Fill tile 0 completely with dirty Y0 (produced by PIM)
        TraceCommand(2, OpCode.SRAM_ALLOC, "X0", "-", "SRAM:T0:B2-3",
                     2048, {"type": "ACTIVATION"}, []),
        TraceCommand(3, OpCode.DMA_LOAD, "X0", "DRAM:0x2000",
                     "SRAM:T0:B2-3", 2048, {}, [2]),
        TraceCommand(4, OpCode.PIM_MAC, "Y0", "W0,X0", "SRAM:T0:B0-1",
                     0, {"mac_count": 64, "out_bytes": 4096}, [1, 3]),
        # Free X0 to make partial room
        TraceCommand(5, OpCode.SRAM_FREE, "X0", "-", "-", 0, {}, [4]),
        # New PIM output needs B0-1 on tile 0, Y0 is dirty on B0-1
        # Must evict dirty Y0 -> writeback
        TraceCommand(6, OpCode.PIM_MAC, "Y1", "W0", "SRAM:T0:B0-1",
                     0, {"mac_count": 64, "out_bytes": 2048}, [5]),
        TraceCommand(7, OpCode.DMA_STORE, "Y1", "SRAM:T0:B0-1",
                     "DRAM:0xA000", 2048, {}, [6]),
    ]
    sim = Simulator(make_tiny_config())
    sim.load_trace(cmds)
    report = sim.run()
    assert report["memory_lifecycle"]["writeback_count"] >= 1
    assert report["traffic"]["dram_write_bytes"] > 2048  # Y0 writeback + Y1 store


def test_output_allocation_no_victim_raises():
    """When all objects are pinned, output allocation must raise in strict mode."""
    cmds = [
        # Pin weight on tile 0
        TraceCommand(0, OpCode.SRAM_ALLOC, "W0", "-", "SRAM:T0:B0-3",
                     8192, {"pinned": True, "type": "WEIGHT"}, []),
        TraceCommand(1, OpCode.DMA_LOAD, "W0", "DRAM:0x0",
                     "SRAM:T0:B0-3", 8192, {}, [0]),
        # PIM output also on tile 0 — no unpinned victim available
        TraceCommand(2, OpCode.PIM_MAC, "Y0", "W0", "SRAM:T0:B0-1",
                     0, {"mac_count": 64, "out_bytes": 4096}, [1]),
    ]
    sim = Simulator(make_tiny_config())
    sim.load_trace(cmds)
    with pytest.raises(RuntimeError, match="eviction candidate|allocation failed"):
        sim.run()
