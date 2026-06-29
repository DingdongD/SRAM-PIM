from dataclasses import dataclass, field
from pathlib import Path
import yaml


@dataclass
class PrecisionConfig:
    activation: str = "int8"
    weight: str = "int8"
    psum: str = "int32"


@dataclass
class WorkloadConfig:
    model: str = "GEMM_test"
    batch_size: int = 1
    precision: PrecisionConfig = field(default_factory=PrecisionConfig)


@dataclass
class SystemConfig:
    frequency_hz: int = 1_000_000_000
    mode: str = "warm_resident"
    num_queries_for_amortization: int = 1000
    count_initial_preload: bool = False
    count_final_writeback: bool = True
    correctness_mode: str = "strict"  # strict | warn | auto_reload
    final_dirty_policy: str = "report"  # error | auto_writeback | report | ignore
    persistent_object_types: list = field(
        default_factory=lambda: ["OUTPUT", "STATE", "META", "DIST"]
    )
    spill_model: str = "blocking"  # blocking | event_level


@dataclass
class DRAMEnergyConfig:
    read_pj_per_byte: float = 15.0
    write_pj_per_byte: float = 18.0
    io_pj_per_byte: float = 8.0


@dataclass
class DRAMConfig:
    model: str = "analytical"  # analytical | trace | dramsim3
    type: str = "LPDDR4"
    channels: int = 1
    bus_width_bits: int = 64
    burst_bytes: int = 64
    effective_bandwidth_gbps: float = 25.6
    fixed_latency_ns: float = 80.0
    dramsim3_dir: str = "/home/NPU-PIM-co-simulator/DRAMsim3"
    dramsim3_config: str = ""
    energy: DRAMEnergyConfig = field(default_factory=DRAMEnergyConfig)


@dataclass
class PortModelConfig:
    read_ports_per_bank: int = 1
    write_ports_per_bank: int = 1
    pim_ports_per_bank: int = 1
    pim_exclusive_with_read: bool = True
    pim_exclusive_with_write: bool = True


@dataclass
class PIMHardwareConfig:
    mode: str = "digital_near_sram"
    lanes_per_bank: int = 128
    mac_latency_cycles: int = 2
    mac_issue_interval_cycles: int = 1
    reduce_latency_cycles: int = 4
    nonlinear_latency_cycles: int = 4
    max_parallel_banks: int = 16
    power_throttle: bool = True
    tile_power_budget_mw: float = 100.0
    nonlinear_units_per_tile: int = 1
    nonlinear_issue_interval_cycles: int = 1


@dataclass
class BufferConfig:
    input_buffer_kb_per_tile: int = 16
    psum_buffer_kb_per_tile: int = 32
    output_buffer_kb_per_tile: int = 16
    double_buffering: bool = True


@dataclass
class PowerStateConfig:
    allow_clock_gating: bool = True
    allow_power_gating: bool = False
    wakeup_latency_cycles: int = 20
    wakeup_energy_pj: float = 500.0


@dataclass
class SRAMPIMConfig:
    tiles: int = 16
    banks_per_tile: int = 32
    bank_capacity_kb: int = 8
    total_capacity_kb: int = 4096
    word_bytes: int = 16
    port_model: PortModelConfig = field(default_factory=PortModelConfig)
    pim: PIMHardwareConfig = field(default_factory=PIMHardwareConfig)
    buffers: BufferConfig = field(default_factory=BufferConfig)
    power_state: PowerStateConfig = field(default_factory=PowerStateConfig)


@dataclass
class SRAMEnergyConfig:
    source: str = "analytical"
    read_pj_per_access: float = 3.2
    write_pj_per_access: float = 3.8
    leakage_mw_per_bank: float = 0.39


@dataclass
class PIMEnergyConfig:
    mac_pj_per_op: float = 0.08
    ew_pj_per_op: float = 0.03
    reduce_pj_per_op: float = 0.04
    nonlinear_pj_per_elem: float = 0.12
    control_pj_per_command: float = 1.0


@dataclass
class NoCEnergyConfig:
    pj_per_byte_per_hop: float = 0.2
    average_hops: int = 2


@dataclass
class DMAEnergyConfig:
    pj_per_byte: float = 0.5


@dataclass
class EnergyConfig:
    sram: SRAMEnergyConfig = field(default_factory=SRAMEnergyConfig)
    pim: PIMEnergyConfig = field(default_factory=PIMEnergyConfig)
    noc: NoCEnergyConfig = field(default_factory=NoCEnergyConfig)
    dma: DMAEnergyConfig = field(default_factory=DMAEnergyConfig)


@dataclass
class SimConfig:
    system: SystemConfig = field(default_factory=SystemConfig)
    workload: WorkloadConfig = field(default_factory=WorkloadConfig)
    dram: DRAMConfig = field(default_factory=DRAMConfig)
    sram_pim: SRAMPIMConfig = field(default_factory=SRAMPIMConfig)
    energy: EnergyConfig = field(default_factory=EnergyConfig)


def _dict_to_dataclass(cls, data):
    if data is None:
        return cls()
    fieldtypes = {f.name: f.type for f in cls.__dataclass_fields__.values()}
    kwargs = {}
    for key, value in data.items():
        if key in fieldtypes:
            ft = fieldtypes[key]
            if isinstance(ft, str):
                ft = eval(ft)
            if hasattr(ft, '__dataclass_fields__') and isinstance(value, dict):
                kwargs[key] = _dict_to_dataclass(ft, value)
            else:
                kwargs[key] = value
    return cls(**kwargs)


def load_config(config_dir: str) -> SimConfig:
    config_path = Path(config_dir)
    raw = {}
    for name in ["system", "dram", "sram_pim", "energy"]:
        fpath = config_path / f"{name}.yaml"
        if fpath.exists():
            with open(fpath) as f:
                raw[name] = yaml.safe_load(f)

    system_raw = raw.get("system", {})
    system_cfg = _dict_to_dataclass(SystemConfig, system_raw.get("system", {}))
    workload_cfg = _dict_to_dataclass(WorkloadConfig, system_raw.get("workload", {}))
    dram_cfg = _dict_to_dataclass(DRAMConfig, raw.get("dram", {}).get("dram", {}))
    sram_pim_cfg = _dict_to_dataclass(SRAMPIMConfig, raw.get("sram_pim", {}).get("sram_pim", {}))
    energy_cfg = _dict_to_dataclass(EnergyConfig, raw.get("energy", {}).get("energy", {}))

    return SimConfig(
        system=system_cfg,
        workload=workload_cfg,
        dram=dram_cfg,
        sram_pim=sram_pim_cfg,
        energy=energy_cfg,
    )
