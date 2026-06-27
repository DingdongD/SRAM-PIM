import pytest
from src.simulator import Simulator
from src.config import SimConfig, SRAMPIMConfig, SystemConfig
from src.trace_ir import TraceCommand, OpCode


def make_simple_config():
    return SimConfig(
        system=SystemConfig(frequency_hz=1_000_000_000, mode="cold_start"),
        sram_pim=SRAMPIMConfig(tiles=2, banks_per_tile=4, bank_capacity_kb=8, total_capacity_kb=64),
    )


def test_empty_trace():
    sim = Simulator(make_simple_config())
    sim.load_trace([])
    report = sim.run()
    assert report["latency"]["total_cycles"] == 0


def test_dma_load_then_pim_mac():
    """Basic: alloc -> load weight -> load activation -> PIM MAC -> store output"""
    cmds = [
        TraceCommand(0, OpCode.SRAM_ALLOC, "W0", "-", "SRAM:T0:B0-3", 8192, {"pinned": True, "type": "WEIGHT"}, []),
        TraceCommand(1, OpCode.DMA_LOAD, "W0", "DRAM:0x1000", "SRAM:T0:B0-3", 8192, {}, [0]),
        TraceCommand(2, OpCode.SRAM_ALLOC, "X0", "-", "SRAM:T0:B4-7", 4096, {"type": "ACTIVATION"}, []),
        TraceCommand(3, OpCode.DMA_LOAD, "X0", "DRAM:0x9000", "SRAM:T0:B4-7", 4096, {}, [2]),
        TraceCommand(4, OpCode.PIM_MAC, "Y0", "W0,X0", "SRAM:T1:B0-1", 0, {"mac_count": 1024}, [1, 3]),
        TraceCommand(5, OpCode.DMA_STORE, "Y0", "SRAM:T1:B0-1", "DRAM:0xA000", 2048, {}, [4]),
    ]
    sim = Simulator(make_simple_config())
    sim.load_trace(cmds)
    report = sim.run()
    assert report["latency"]["total_cycles"] > 0
    assert report["energy"]["total_pj"] > 0
    assert report["traffic"]["dram_read_bytes"] == 8192 + 4096
    assert report["traffic"]["dram_write_bytes"] == 2048


def test_dependency_enforcement():
    """PIM_MAC cannot run before its DMA_LOAD dependency completes"""
    cmds = [
        TraceCommand(0, OpCode.SRAM_ALLOC, "W0", "-", "SRAM:T0:B0-3", 8192, {"pinned": True, "type": "WEIGHT"}, []),
        TraceCommand(1, OpCode.DMA_LOAD, "W0", "DRAM:0x1000", "SRAM:T0:B0-3", 8192, {}, [0]),
        TraceCommand(2, OpCode.PIM_MAC, "Y0", "W0", "SRAM:T1:B0", 0, {"mac_count": 100}, [1]),
    ]
    sim = Simulator(make_simple_config())
    sim.load_trace(cmds)
    report = sim.run()
    # PIM_MAC starts after DMA_LOAD finishes
    assert report["latency"]["total_cycles"] > report["latency"]["dram_load_cycles"]


def test_unresident_read_detected():
    """Correctness: reading object not in SRAM should be flagged"""
    cmds = [
        # PIM_MAC without DMA_LOAD -- object not in SRAM
        TraceCommand(0, OpCode.PIM_MAC, "Y0", "W0_missing", "SRAM:T0:B0", 0, {"mac_count": 100}, []),
    ]
    sim = Simulator(make_simple_config())
    sim.load_trace(cmds)
    report = sim.run()
    assert report["correctness"]["illegal_read_unresident_object"] > 0


def test_warm_resident_mode():
    """In warm_resident mode, pre-loaded weights skip DMA cost"""
    cmds = [
        TraceCommand(0, OpCode.SRAM_ALLOC, "W0", "-", "SRAM:T0:B0-3", 8192, {"pinned": True, "type": "WEIGHT", "preloaded": True}, []),
        TraceCommand(1, OpCode.SRAM_ALLOC, "X0", "-", "SRAM:T0:B4-7", 4096, {"type": "ACTIVATION"}, []),
        TraceCommand(2, OpCode.DMA_LOAD, "X0", "DRAM:0x9000", "SRAM:T0:B4-7", 4096, {}, [1]),
        TraceCommand(3, OpCode.PIM_MAC, "Y0", "W0,X0", "SRAM:T1:B0-1", 0, {"mac_count": 1024}, [0, 2]),
    ]
    config = make_simple_config()
    config.system.mode = "warm_resident"
    sim = Simulator(config)
    sim.load_trace(cmds)
    report = sim.run()
    assert report["traffic"]["dram_read_bytes"] == 4096  # Only activation loaded
