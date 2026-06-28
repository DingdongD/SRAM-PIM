from src.simulator import Simulator
from src.config import (SimConfig, SystemConfig, SRAMPIMConfig,
                        PortModelConfig, PIMHardwareConfig)
from src.trace_ir import TraceCommand, OpCode


def make_config():
    return SimConfig(
        system=SystemConfig(frequency_hz=1_000_000_000, mode="warm_resident",
                            correctness_mode="strict"),
        sram_pim=SRAMPIMConfig(
            tiles=4, banks_per_tile=8, bank_capacity_kb=8, total_capacity_kb=256,
            port_model=PortModelConfig(
                pim_exclusive_with_read=True, pim_exclusive_with_write=True),
            pim=PIMHardwareConfig(
                lanes_per_bank=16, mac_latency_cycles=2,
                mac_issue_interval_cycles=1, max_parallel_banks=16),
        ),
    )


def test_bank_conflict_stall():
    """Two PIM_MACs on same bank should serialize."""
    config = make_config()
    # mac_count=256, lanes=16 -> ceil(256/16)=16 cycles per MAC
    cmds = [
        TraceCommand(0, OpCode.SRAM_ALLOC, "W0", "-", "SRAM:T0:B0-0", 1024,
                     {"pinned": True, "type": "WEIGHT", "preloaded": True}, []),
        TraceCommand(1, OpCode.SRAM_ALLOC, "X0", "-", "SRAM:T0:B1-1", 1024,
                     {"type": "ACTIVATION", "preloaded": True}, []),
        TraceCommand(2, OpCode.PIM_MAC, "Y0", "W0,X0", "SRAM:T0:B2-2", 0,
                     {"mac_count": 256, "banks": [2]}, [0, 1]),
        TraceCommand(3, OpCode.PIM_MAC, "Y1", "W0,X0", "SRAM:T0:B2-2", 0,
                     {"mac_count": 256, "banks": [2]}, [0, 1]),
    ]
    sim = Simulator(config)
    sim.load_trace(cmds)
    report = sim.run()
    # Two MACs serialized on same bank: total >= 2 * single MAC latency
    assert report["latency"]["total_cycles"] >= 32


def test_no_conflict_different_banks():
    """Two PIM_MACs on different banks should run in parallel."""
    config = make_config()
    cmds = [
        TraceCommand(0, OpCode.SRAM_ALLOC, "W0", "-", "SRAM:T0:B0-0", 1024,
                     {"pinned": True, "type": "WEIGHT", "preloaded": True}, []),
        TraceCommand(1, OpCode.SRAM_ALLOC, "X0", "-", "SRAM:T0:B1-1", 1024,
                     {"type": "ACTIVATION", "preloaded": True}, []),
        TraceCommand(2, OpCode.PIM_MAC, "Y0", "W0,X0", "SRAM:T0:B2-2", 0,
                     {"mac_count": 256, "banks": [2]}, [0, 1]),
        TraceCommand(3, OpCode.PIM_MAC, "Y1", "W0,X0", "SRAM:T0:B3-3", 0,
                     {"mac_count": 256, "banks": [3]}, [0, 1]),
    ]
    sim = Simulator(config)
    sim.load_trace(cmds)
    report = sim.run()
    # No conflict: both run in parallel, ~single MAC latency
    single_lat = 16  # ceil(256/16) = 16
    assert report["latency"]["total_cycles"] < single_lat * 2


def test_resource_stall_tracked():
    """Simulator should track bank_conflict_stall_cycles."""
    config = make_config()
    cmds = [
        TraceCommand(0, OpCode.SRAM_ALLOC, "W0", "-", "SRAM:T0:B0-0", 1024,
                     {"pinned": True, "type": "WEIGHT", "preloaded": True}, []),
        TraceCommand(1, OpCode.SRAM_ALLOC, "X0", "-", "SRAM:T0:B1-1", 1024,
                     {"type": "ACTIVATION", "preloaded": True}, []),
        TraceCommand(2, OpCode.PIM_MAC, "Y0", "W0,X0", "SRAM:T0:B2-2", 0,
                     {"mac_count": 256, "banks": [2]}, [0, 1]),
        TraceCommand(3, OpCode.PIM_MAC, "Y1", "W0,X0", "SRAM:T0:B2-2", 0,
                     {"mac_count": 256, "banks": [2]}, [0, 1]),
    ]
    sim = Simulator(config)
    sim.load_trace(cmds)
    report = sim.run()
    assert "bank_conflict_stall_cycles" in report["latency"]
    assert report["latency"]["bank_conflict_stall_cycles"] > 0
