from src.simulator import Simulator
from src.config import SimConfig, SRAMPIMConfig, SystemConfig
from src.trace_ir import TraceCommand, OpCode


def make_config():
    return SimConfig(
        system=SystemConfig(
            frequency_hz=1_000_000_000,
            mode="cold_start",
            correctness_mode="strict",
            final_dirty_policy="ignore",
        ),
        sram_pim=SRAMPIMConfig(
            tiles=2, banks_per_tile=4,
            bank_capacity_kb=2, total_capacity_kb=16,
        ),
    )


def test_bank_overflow_spill_evicts_target_bank_object():
    """When target banks are full, eviction should prefer objects on those banks."""
    # tile 0 = 4 banks * 2KB = 8KB. Fill completely with A0(4KB) + B0(4KB)
    cmds = [
        # Object A on tile 0, banks 0-1 (4KB)
        TraceCommand(0, OpCode.SRAM_ALLOC, "A0", "-", "SRAM:T0:B0-1",
                     4096, {"type": "ACTIVATION"}, []),
        TraceCommand(1, OpCode.DMA_LOAD, "A0", "DRAM:0x0",
                     "SRAM:T0:B0-1", 4096, {}, [0]),
        # Object B on tile 0, banks 2-3 (4KB)
        TraceCommand(2, OpCode.SRAM_ALLOC, "B0", "-", "SRAM:T0:B2-3",
                     4096, {"type": "ACTIVATION"}, []),
        TraceCommand(3, OpCode.DMA_LOAD, "B0", "DRAM:0x1000",
                     "SRAM:T0:B2-3", 4096, {}, [2]),
        # Weight on tile 1
        TraceCommand(4, OpCode.SRAM_ALLOC, "W0", "-", "SRAM:T1:B0-1",
                     2048, {"pinned": True, "type": "WEIGHT"}, []),
        TraceCommand(5, OpCode.DMA_LOAD, "W0", "DRAM:0x2000",
                     "SRAM:T1:B0-1", 2048, {}, [4]),
        # PIM output needs banks 0-1 on tile 0 — should evict A0, not B0
        TraceCommand(6, OpCode.PIM_MAC, "Y0", "W0", "SRAM:T0:B0-1",
                     0, {"mac_count": 64, "out_bytes": 4096}, [1, 5]),
        TraceCommand(7, OpCode.DMA_STORE, "Y0", "SRAM:T0:B0-1",
                     "DRAM:0xA000", 4096, {}, [6]),
    ]
    sim = Simulator(make_config())
    sim.load_trace(cmds)
    report = sim.run()
    assert report["memory_lifecycle"]["eviction_count"] >= 1
    # A0 should have been evicted (it was on target banks 0-1)
    a0 = sim.mem_mgr.objects["A0"]
    assert not a0.valid_in_sram
    # B0 should still be valid (it was on banks 2-3, not target)
    b0 = sim.mem_mgr.objects["B0"]
    assert b0.valid_in_sram
