from src.config import EnergyConfig, SRAMPIMConfig, DRAMConfig


class EnergyModel:
    def __init__(self, config: EnergyConfig, sram_config: SRAMPIMConfig,
                 dram_config: DRAMConfig, freq_hz: int):
        self.config = config
        self.sram_config = sram_config
        self.dram_config = dram_config
        self.freq_hz = freq_hz
        self.total_banks = sram_config.tiles * sram_config.banks_per_tile
        self._counters = self._zero_counters()

    def _zero_counters(self) -> dict:
        return {
            "dram_read_bytes": 0,
            "dram_write_bytes": 0,
            "sram_read_count": 0,
            "sram_write_count": 0,
            "pim_mac_count": 0,
            "pim_reduce_count": 0,
            "pim_nl_count": 0,
            "noc_bytes": 0,
            "leakage_cycles": 0,
            "command_count": 0,
        }

    def reset(self):
        self._counters = self._zero_counters()

    def add_dram_read(self, nbytes: int):
        self._counters["dram_read_bytes"] += nbytes

    def add_dram_write(self, nbytes: int):
        self._counters["dram_write_bytes"] += nbytes

    def add_sram_read(self, count: int):
        self._counters["sram_read_count"] += count

    def add_sram_write(self, count: int):
        self._counters["sram_write_count"] += count

    def add_pim_mac(self, count: int):
        self._counters["pim_mac_count"] += count

    def add_pim_reduce(self, count: int):
        self._counters["pim_reduce_count"] += count

    def add_pim_nl(self, count: int):
        self._counters["pim_nl_count"] += count

    def add_noc(self, nbytes: int):
        self._counters["noc_bytes"] += nbytes

    def add_leakage(self, cycles: int):
        self._counters["leakage_cycles"] += cycles

    def add_command(self):
        self._counters["command_count"] += 1

    def get_breakdown(self) -> dict:
        c = self._counters
        ec = self.config
        dc = self.dram_config

        dram_read = c["dram_read_bytes"] * dc.energy.read_pj_per_byte
        dram_write = c["dram_write_bytes"] * dc.energy.write_pj_per_byte
        io = (c["dram_read_bytes"] + c["dram_write_bytes"]) * dc.energy.io_pj_per_byte
        sram_read = c["sram_read_count"] * ec.sram.read_pj_per_access
        sram_write = c["sram_write_count"] * ec.sram.write_pj_per_access

        # Leakage: P_leak(W) * time(s) -> J -> pJ
        time_s = c["leakage_cycles"] / self.freq_hz
        leak_pj = self.total_banks * ec.sram.leakage_mw_per_bank * 1e-3 * time_s * 1e12

        pim_mac = c["pim_mac_count"] * ec.pim.mac_pj_per_op
        pim_reduce = c["pim_reduce_count"] * ec.pim.reduce_pj_per_op
        pim_nl = c["pim_nl_count"] * ec.pim.nonlinear_pj_per_elem
        noc = c["noc_bytes"] * ec.noc.pj_per_byte_per_hop * ec.noc.average_hops
        dma = (c["dram_read_bytes"] + c["dram_write_bytes"]) * ec.dma.pj_per_byte
        control = c["command_count"] * ec.pim.control_pj_per_command

        total = (dram_read + dram_write + io + sram_read + sram_write +
                 leak_pj + pim_mac + pim_reduce + pim_nl + noc + dma + control)

        return {
            "dram_read_pj": dram_read,
            "dram_write_pj": dram_write,
            "offchip_io_pj": io,
            "dma_pj": dma,
            "sram_read_pj": sram_read,
            "sram_write_pj": sram_write,
            "sram_leakage_pj": leak_pj,
            "pim_mac_pj": pim_mac,
            "pim_reduce_pj": pim_reduce,
            "pim_nl_pj": pim_nl,
            "noc_pj": noc,
            "control_pj": control,
            "total_pj": total,
            **{k: v for k, v in c.items()},
        }
