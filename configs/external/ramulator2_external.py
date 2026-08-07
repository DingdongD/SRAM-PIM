import ramulator

frontend = ramulator.frontend.External(clock_ratio=1)
dram = ramulator.dram.HBM2(
    org_preset="HBM2_4Gb",
    timing_preset="HBM2_2000Mbps",
)
controller = ramulator.controller.GenericDDR(
    dram=dram,
    scheduler=ramulator.scheduler.FRFCFS(),
    refresh_manager=ramulator.refresh_manager.AllBank(),
    row_policy=ramulator.row_policy.Open(),
    addr_mapper=ramulator.addr_mapper.RoBaRaCoCh(),
)
memory = ramulator.memory_system.GenericDRAM(
    clock_ratio=1,
    controllers=[controller],
    channel_mapper=ramulator.channel_mapper.CacheLineInterleave(),
)
sim = ramulator.Simulation(frontend, memory)
