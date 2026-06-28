import pytest
from src.simulator import Simulator
from src.config import SimConfig, SRAMPIMConfig, SystemConfig
from src.trace_ir import TraceCommand, OpCode


def make_tiny_config(correctness_mode="strict", final_dirty_policy="ignore"):
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


def test_output_spill_does_not_evict_current_inputs():
    """PIM output spill must not evict the current command's src inputs."""
    # tile 0 capacity = 4 banks * 2KB = 8KB
    # Fill it completely: X0=4KB + V0=4KB = 8KB
    # PIM output Y0 needs 4KB on tile 0 → must evict V0, NOT X0 (input)
    cmds = [
        # Weight on tile 1 (pinned)
        TraceCommand(0, OpCode.SRAM_ALLOC, "W0", "-", "SRAM:T1:B0-1",
                     2048, {"pinned": True, "type": "WEIGHT"}, []),
        TraceCommand(1, OpCode.DMA_LOAD, "W0", "DRAM:0x0",
                     "SRAM:T1:B0-1", 2048, {}, [0]),
        # Input X0 on tile 0 banks 0-1 (4KB)
        TraceCommand(2, OpCode.SRAM_ALLOC, "X0", "-", "SRAM:T0:B0-1",
                     4096, {"type": "ACTIVATION"}, []),
        TraceCommand(3, OpCode.DMA_LOAD, "X0", "DRAM:0x1000",
                     "SRAM:T0:B0-1", 4096, {}, [2]),
        # Victim V0 on tile 0 banks 2-3 (4KB) — fills tile completely
        TraceCommand(4, OpCode.SRAM_ALLOC, "V0", "-", "SRAM:T0:B2-3",
                     4096, {"type": "ACTIVATION"}, []),
        TraceCommand(5, OpCode.DMA_LOAD, "V0", "DRAM:0x2000",
                     "SRAM:T0:B2-3", 4096, {}, [4]),
        # PIM uses W0 and X0 as inputs, output on tile 0
        # Must evict V0 (not X0!) to make room
        TraceCommand(6, OpCode.PIM_MAC, "Y0", "W0,X0", "SRAM:T0:B2-3",
                     0, {"mac_count": 64, "out_bytes": 4096}, [1, 3, 5]),
        TraceCommand(7, OpCode.DMA_STORE, "Y0", "SRAM:T0:B2-3",
                     "DRAM:0xA000", 4096, {}, [6]),
    ]
    sim = Simulator(make_tiny_config())
    sim.load_trace(cmds)
    report = sim.run()
    # X0 (input) must NOT have been evicted
    x0 = sim.mem_mgr.objects["X0"]
    assert x0.valid_in_sram
    # V0 should have been evicted
    assert report["memory_lifecycle"]["eviction_count"] >= 1
    assert report["valid_simulation"] is True


def test_protected_spill_only_protected_raises():
    """If only protected objects are available for eviction, strict must raise."""
    cmds = [
        # Input X0 fills tile 0 banks 0-3
        TraceCommand(0, OpCode.SRAM_ALLOC, "X0", "-", "SRAM:T0:B0-3",
                     8192, {"type": "ACTIVATION"}, []),
        TraceCommand(1, OpCode.DMA_LOAD, "X0", "DRAM:0x0",
                     "SRAM:T0:B0-3", 8192, {}, [0]),
        # PIM uses X0, needs output on tile 0 — but X0 is protected, nothing else to evict
        TraceCommand(2, OpCode.PIM_MAC, "Y0", "X0", "SRAM:T0:B0-1",
                     0, {"mac_count": 64, "out_bytes": 4096}, [1]),
    ]
    sim = Simulator(make_tiny_config())
    sim.load_trace(cmds)
    with pytest.raises(RuntimeError):
        sim.run()
