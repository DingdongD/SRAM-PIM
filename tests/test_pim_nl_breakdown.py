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
        sram_pim=SRAMPIMConfig(tiles=4, banks_per_tile=8,
                               bank_capacity_kb=8, total_capacity_kb=256),
    )


def test_pim_nl_cycles_tracked_separately():
    """PIM_NL compute cycles should appear in pim_nl_cycles, not pim_compute_cycles."""
    cmds = [
        TraceCommand(0, OpCode.SRAM_ALLOC, "X0", "-", "SRAM:T0:B0-3",
                     8192, {"type": "ACTIVATION"}, []),
        TraceCommand(1, OpCode.DMA_LOAD, "X0", "DRAM:0x0",
                     "SRAM:T0:B0-3", 8192, {}, [0]),
        TraceCommand(2, OpCode.PIM_NL, "Y0", "X0", "SRAM:T1:B0-1",
                     0, {"count": 128, "out_bytes": 4096,
                         "out_type": "ACTIVATION"}, [1]),
        TraceCommand(3, OpCode.DMA_STORE, "Y0", "SRAM:T1:B0-1",
                     "DRAM:0xA000", 4096, {}, [2]),
    ]
    sim = Simulator(make_config())
    sim.load_trace(cmds)
    report = sim.run()

    lat = report["latency"]
    assert lat["pim_nl_cycles"] > 0
    # pim_compute_cycles should not include NL cycles
    assert lat["pim_compute_cycles"] == 0
    assert report["valid_simulation"] is True


def test_pim_nl_and_mac_breakdown_independent():
    """When both MAC and NL ops are used, their cycles should be tracked independently."""
    cmds = [
        TraceCommand(0, OpCode.SRAM_ALLOC, "W0", "-", "SRAM:T0:B0-3",
                     8192, {"pinned": True, "type": "WEIGHT"}, []),
        TraceCommand(1, OpCode.DMA_LOAD, "W0", "DRAM:0x0",
                     "SRAM:T0:B0-3", 8192, {}, [0]),
        TraceCommand(2, OpCode.SRAM_ALLOC, "X0", "-", "SRAM:T0:B4-7",
                     8192, {"type": "ACTIVATION"}, []),
        TraceCommand(3, OpCode.DMA_LOAD, "X0", "DRAM:0x2000",
                     "SRAM:T0:B4-7", 8192, {}, [2]),
        # MAC op
        TraceCommand(4, OpCode.PIM_MAC, "P0", "W0,X0", "SRAM:T1:B0-3",
                     0, {"mac_count": 256, "out_bytes": 4096}, [1, 3]),
        # NL op on MAC output
        TraceCommand(5, OpCode.PIM_NL, "Y0", "P0", "SRAM:T2:B0-1",
                     0, {"count": 64, "out_bytes": 2048,
                         "out_type": "ACTIVATION"}, [4]),
        TraceCommand(6, OpCode.DMA_STORE, "Y0", "SRAM:T2:B0-1",
                     "DRAM:0xA000", 2048, {}, [5]),
    ]
    sim = Simulator(make_config())
    sim.load_trace(cmds)
    report = sim.run()

    lat = report["latency"]
    assert lat["pim_compute_cycles"] > 0
    assert lat["pim_nl_cycles"] > 0
    # They should be independent values
    assert lat["pim_compute_cycles"] != lat["pim_nl_cycles"]
