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


def test_multi_victim_eviction():
    """Eviction should be able to evict multiple objects to free enough space."""
    # tile 0 = 4 banks * 2KB = 8KB. Fill with 4 x 2KB objects.
    cmds = [
        TraceCommand(0, OpCode.SRAM_ALLOC, "A0", "-", "SRAM:T0:B0",
                     2048, {"type": "ACTIVATION"}, []),
        TraceCommand(1, OpCode.DMA_LOAD, "A0", "DRAM:0x0",
                     "SRAM:T0:B0", 2048, {}, [0]),
        TraceCommand(2, OpCode.SRAM_ALLOC, "A1", "-", "SRAM:T0:B1",
                     2048, {"type": "ACTIVATION"}, []),
        TraceCommand(3, OpCode.DMA_LOAD, "A1", "DRAM:0x1000",
                     "SRAM:T0:B1", 2048, {}, [2]),
        TraceCommand(4, OpCode.SRAM_ALLOC, "A2", "-", "SRAM:T0:B2",
                     2048, {"type": "ACTIVATION"}, []),
        TraceCommand(5, OpCode.DMA_LOAD, "A2", "DRAM:0x2000",
                     "SRAM:T0:B2", 2048, {}, [4]),
        TraceCommand(6, OpCode.SRAM_ALLOC, "A3", "-", "SRAM:T0:B3",
                     2048, {"type": "ACTIVATION"}, []),
        TraceCommand(7, OpCode.DMA_LOAD, "A3", "DRAM:0x3000",
                     "SRAM:T0:B3", 2048, {}, [6]),
        # Weight on tile 1
        TraceCommand(8, OpCode.SRAM_ALLOC, "W0", "-", "SRAM:T1:B0-1",
                     2048, {"pinned": True, "type": "WEIGHT"}, []),
        TraceCommand(9, OpCode.DMA_LOAD, "W0", "DRAM:0x4000",
                     "SRAM:T1:B0-1", 2048, {}, [8]),
        # PIM output needs 4KB on banks 0-1 → must evict A0 + A1
        TraceCommand(10, OpCode.PIM_MAC, "Y0", "W0", "SRAM:T0:B0-1",
                     0, {"mac_count": 64, "out_bytes": 4096}, [1, 3, 9]),
        TraceCommand(11, OpCode.DMA_STORE, "Y0", "SRAM:T0:B0-1",
                     "DRAM:0xA000", 4096, {}, [10]),
    ]
    sim = Simulator(make_config())
    sim.load_trace(cmds)
    report = sim.run()
    assert report["memory_lifecycle"]["eviction_count"] >= 2
    # A0 and A1 on target banks should be evicted
    assert not sim.mem_mgr.objects["A0"].valid_in_sram
    assert not sim.mem_mgr.objects["A1"].valid_in_sram
    # A2 and A3 on non-target banks should remain
    assert sim.mem_mgr.objects["A2"].valid_in_sram
    assert sim.mem_mgr.objects["A3"].valid_in_sram


def test_dirty_victim_writeback_during_bank_spill():
    """Dirty victims on target banks should be written back before eviction."""
    # Fill tile 0 completely, with one dirty object on target banks.
    # D0(dirty, banks 0-1) + B0(clean, banks 2-3) = 8KB.
    # PIM output Y0 needs banks 0-1 → must evict dirty D0 with writeback.
    cmds = [
        # Weight on tile 1
        TraceCommand(0, OpCode.SRAM_ALLOC, "W0", "-", "SRAM:T1:B0-1",
                     2048, {"pinned": True, "type": "WEIGHT"}, []),
        TraceCommand(1, OpCode.DMA_LOAD, "W0", "DRAM:0x0",
                     "SRAM:T1:B0-1", 2048, {}, [0]),
        # Input X0 on tile 0 banks 2-3 (used as PIM input)
        TraceCommand(2, OpCode.SRAM_ALLOC, "X0", "-", "SRAM:T0:B2-3",
                     4096, {"type": "ACTIVATION"}, []),
        TraceCommand(3, OpCode.DMA_LOAD, "X0", "DRAM:0x1000",
                     "SRAM:T0:B2-3", 4096, {}, [2]),
        # First PIM: produces D0 on banks 0-1 (becomes dirty output)
        TraceCommand(4, OpCode.PIM_MAC, "D0", "W0,X0", "SRAM:T0:B0-1",
                     0, {"mac_count": 64, "out_bytes": 4096,
                         "out_type": "OUTPUT"}, [1, 3]),
        # Now tile 0 has: D0(dirty,banks 0-1,4KB) + X0(clean,banks 2-3,4KB) = 8KB
        # PIM output Y0 needs banks 0-1 → must evict D0 (dirty → writeback)
        # X0 is a PIM input so it's protected
        TraceCommand(5, OpCode.PIM_MAC, "Y0", "W0,X0", "SRAM:T0:B0-1",
                     0, {"mac_count": 64, "out_bytes": 4096,
                         "out_type": "OUTPUT"}, [4]),
        TraceCommand(6, OpCode.DMA_STORE, "Y0", "SRAM:T0:B0-1",
                     "DRAM:0xA000", 4096, {}, [5]),
    ]
    sim = Simulator(make_config())
    sim.load_trace(cmds)
    report = sim.run()
    # D0 should have been evicted (it was dirty on target banks)
    assert not sim.mem_mgr.objects["D0"].valid_in_sram
    assert report["memory_lifecycle"]["eviction_count"] >= 1
    assert report["memory_lifecycle"]["writeback_count"] >= 1


def test_mixed_bank_object_eviction_priority():
    """Objects partially overlapping target banks should have intermediate priority."""
    # Use 2 banks per tile, 2KB each = 4KB tile capacity
    cfg = SimConfig(
        system=SystemConfig(
            frequency_hz=1_000_000_000,
            mode="cold_start",
            correctness_mode="strict",
            final_dirty_policy="ignore",
        ),
        sram_pim=SRAMPIMConfig(
            tiles=2, banks_per_tile=2,
            bank_capacity_kb=2, total_capacity_kb=8,
        ),
    )
    # tile 0 = 2 banks * 2KB = 4KB. Fill with 2 x 2KB.
    cmds = [
        # C0 on bank 0 (2KB)
        TraceCommand(0, OpCode.SRAM_ALLOC, "C0", "-", "SRAM:T0:B0",
                     2048, {"type": "ACTIVATION"}, []),
        TraceCommand(1, OpCode.DMA_LOAD, "C0", "DRAM:0x0",
                     "SRAM:T0:B0", 2048, {}, [0]),
        # C1 on bank 1 (2KB)
        TraceCommand(2, OpCode.SRAM_ALLOC, "C1", "-", "SRAM:T0:B1",
                     2048, {"type": "ACTIVATION"}, []),
        TraceCommand(3, OpCode.DMA_LOAD, "C1", "DRAM:0x1000",
                     "SRAM:T0:B1", 2048, {}, [2]),
        # Weight on tile 1
        TraceCommand(4, OpCode.SRAM_ALLOC, "W0", "-", "SRAM:T1:B0",
                     2048, {"pinned": True, "type": "WEIGHT"}, []),
        TraceCommand(5, OpCode.DMA_LOAD, "W0", "DRAM:0x2000",
                     "SRAM:T1:B0", 2048, {}, [4]),
        # PIM output needs bank 0 → should evict C0 (overlaps bank 0), not C1
        TraceCommand(6, OpCode.PIM_MAC, "Y0", "W0", "SRAM:T0:B0",
                     0, {"mac_count": 64, "out_bytes": 2048}, [1, 5]),
        TraceCommand(7, OpCode.DMA_STORE, "Y0", "SRAM:T0:B0",
                     "DRAM:0xA000", 2048, {}, [6]),
    ]
    sim = Simulator(cfg)
    sim.load_trace(cmds)
    report = sim.run()
    assert report["memory_lifecycle"]["eviction_count"] >= 1
    assert not sim.mem_mgr.objects["C0"].valid_in_sram
    assert sim.mem_mgr.objects["C1"].valid_in_sram
