from src.simulator import Simulator
from src.config import SimConfig, SystemConfig, SRAMPIMConfig, PortModelConfig
from src.trace_ir import TraceCommand, OpCode


def make_config():
    return SimConfig(
        system=SystemConfig(frequency_hz=1_000_000_000, mode="warm_resident"),
        sram_pim=SRAMPIMConfig(
            tiles=2, banks_per_tile=4, bank_capacity_kb=8, total_capacity_kb=64,
            port_model=PortModelConfig(pim_exclusive_with_read=True, pim_exclusive_with_write=True),
        ),
    )


def test_bank_conflict_stall():
    """Two PIM_MACs on same bank should serialize"""
    config = make_config()
    cmds = [
        TraceCommand(0, OpCode.SRAM_ALLOC, "W0", "-", "SRAM:T0:B0-0", 1024,
                     {"pinned": True, "type": "WEIGHT", "preloaded": True}, []),
        TraceCommand(1, OpCode.SRAM_ALLOC, "X0", "-", "SRAM:T0:B1-1", 1024,
                     {"type": "ACTIVATION", "preloaded": True}, []),
        TraceCommand(2, OpCode.PIM_MAC, "Y0", "W0,X0", "SRAM:T0:B0-0", 0,
                     {"mac_count": 100, "banks": [0]}, [0, 1]),
        TraceCommand(3, OpCode.PIM_MAC, "Y1", "W0,X0", "SRAM:T0:B0-0", 0,
                     {"mac_count": 100, "banks": [0]}, [0, 1]),
    ]
    sim = Simulator(config)
    sim.load_trace(cmds)
    report = sim.run()
    # Two MACs on same bank should take longer than one
    assert report["latency"]["total_cycles"] >= 200


def test_no_conflict_different_banks():
    """Two PIM_MACs on different banks should run in parallel (no extra stall)."""
    config = make_config()
    cmds = [
        TraceCommand(0, OpCode.SRAM_ALLOC, "W0", "-", "SRAM:T0:B0-0", 1024,
                     {"pinned": True, "type": "WEIGHT", "preloaded": True}, []),
        TraceCommand(1, OpCode.SRAM_ALLOC, "X0", "-", "SRAM:T0:B1-1", 1024,
                     {"type": "ACTIVATION", "preloaded": True}, []),
        TraceCommand(2, OpCode.PIM_MAC, "Y0", "W0,X0", "SRAM:T0:B0-0", 0,
                     {"mac_count": 100, "banks": [0]}, [0, 1]),
        TraceCommand(3, OpCode.PIM_MAC, "Y1", "W0,X0", "SRAM:T0:B2-2", 0,
                     {"mac_count": 100, "banks": [2]}, [0, 1]),
    ]
    sim = Simulator(config)
    sim.load_trace(cmds)
    report = sim.run()
    # With no conflict, both MACs run in parallel — total should be ~mac_latency
    # (they issue in the same cycle since both deps [0,1] are met simultaneously)
    single_mac_lat = config.sram_pim.pim.mac_issue_interval_cycles * 100
    assert report["latency"]["total_cycles"] < single_mac_lat * 2


def test_resource_stall_tracked():
    """Simulator should track bank_conflict_stall_cycles in latency breakdown."""
    config = make_config()
    cmds = [
        TraceCommand(0, OpCode.SRAM_ALLOC, "W0", "-", "SRAM:T0:B0-0", 1024,
                     {"pinned": True, "type": "WEIGHT", "preloaded": True}, []),
        TraceCommand(1, OpCode.SRAM_ALLOC, "X0", "-", "SRAM:T0:B1-1", 1024,
                     {"type": "ACTIVATION", "preloaded": True}, []),
        TraceCommand(2, OpCode.PIM_MAC, "Y0", "W0,X0", "SRAM:T0:B0-0", 0,
                     {"mac_count": 100, "banks": [0]}, [0, 1]),
        TraceCommand(3, OpCode.PIM_MAC, "Y1", "W0,X0", "SRAM:T0:B0-0", 0,
                     {"mac_count": 100, "banks": [0]}, [0, 1]),
    ]
    sim = Simulator(config)
    sim.load_trace(cmds)
    report = sim.run()
    assert "bank_conflict_stall_cycles" in report["latency"]
    assert report["latency"]["bank_conflict_stall_cycles"] > 0
