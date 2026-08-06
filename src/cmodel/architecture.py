"""Architecture specification for centralized and bank-local systolic NPUs."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .errors import ConfigurationError


class ComputePlacement(str, Enum):
    CENTRALIZED_LOGIC = "centralized_logic"
    BANK_LOCAL_LOGIC = "bank_local_logic"


@dataclass(frozen=True, slots=True)
class ArraySpec:
    count: int
    rows: int
    cols: int
    dataflow: str

    def __post_init__(self) -> None:
        if self.count <= 0 or self.rows <= 0 or self.cols <= 0:
            raise ConfigurationError("array count/rows/cols must be positive")
        if self.dataflow not in {"ws", "os", "is"}:
            raise ConfigurationError("array dataflow must be ws, os, or is")


@dataclass(frozen=True, slots=True)
class StackedSRAMSpec:
    tiers: int
    banks_per_tier: int
    bank_capacity_bytes: int
    line_bytes: int
    read_ports_per_bank: int
    write_ports_per_bank: int
    read_latency_cycles: int
    write_latency_cycles: int
    read_bytes_per_cycle: int
    write_bytes_per_cycle: int

    def __post_init__(self) -> None:
        positive = {
            "tiers": self.tiers,
            "banks_per_tier": self.banks_per_tier,
            "bank_capacity_bytes": self.bank_capacity_bytes,
            "line_bytes": self.line_bytes,
            "read_ports_per_bank": self.read_ports_per_bank,
            "write_ports_per_bank": self.write_ports_per_bank,
            "read_latency_cycles": self.read_latency_cycles,
            "write_latency_cycles": self.write_latency_cycles,
            "read_bytes_per_cycle": self.read_bytes_per_cycle,
            "write_bytes_per_cycle": self.write_bytes_per_cycle,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ConfigurationError(f"{name} must be positive, got {value}")
        if self.bank_capacity_bytes % self.line_bytes:
            raise ConfigurationError("bank capacity must be a multiple of line_bytes")

    @property
    def total_banks(self) -> int:
        return self.tiers * self.banks_per_tier

    @property
    def total_capacity_bytes(self) -> int:
        return self.total_banks * self.bank_capacity_bytes

    def bank_for_address(self, address: int) -> int:
        if address < 0:
            raise ConfigurationError("address cannot be negative")
        return (address // self.line_bytes) % self.total_banks


@dataclass(frozen=True, slots=True)
class LinkSpec:
    groups: int
    latency_cycles: int
    bytes_per_cycle: int

    def __post_init__(self) -> None:
        if self.groups <= 0 or self.latency_cycles <= 0 or self.bytes_per_cycle <= 0:
            raise ConfigurationError("link groups/latency/bandwidth must be positive")


@dataclass(frozen=True, slots=True)
class BufferSpec:
    capacity_bytes: int
    read_ports: int
    write_ports: int
    read_latency_cycles: int
    write_latency_cycles: int
    read_bytes_per_cycle: int
    write_bytes_per_cycle: int

    def __post_init__(self) -> None:
        values = (
            self.capacity_bytes,
            self.read_ports,
            self.write_ports,
            self.read_latency_cycles,
            self.write_latency_cycles,
            self.read_bytes_per_cycle,
            self.write_bytes_per_cycle,
        )
        if any(v <= 0 for v in values):
            raise ConfigurationError("buffer parameters must be positive")


@dataclass(frozen=True, slots=True)
class ReductionSpec:
    units: int
    lanes_per_unit: int
    latency_cycles: int

    def __post_init__(self) -> None:
        if self.units <= 0 or self.lanes_per_unit <= 0 or self.latency_cycles <= 0:
            raise ConfigurationError("reduction parameters must be positive")


@dataclass(frozen=True, slots=True)
class ArchitectureSpec:
    name: str
    frequency_hz: int
    placement: ComputePlacement
    arrays: ArraySpec
    sram: StackedSRAMSpec
    vertical_link: LinkSpec
    global_buffer: BufferSpec
    noc: LinkSpec
    reduction: ReductionSpec
    bank_groups: int

    def __post_init__(self) -> None:
        if not self.name:
            raise ConfigurationError("architecture name must be non-empty")
        if self.frequency_hz <= 0:
            raise ConfigurationError("frequency_hz must be positive")
        if self.bank_groups <= 0:
            raise ConfigurationError("bank_groups must be positive")
        if self.sram.total_banks % self.bank_groups:
            raise ConfigurationError(
                "total SRAM banks must be divisible by bank_groups"
            )
        if self.placement is ComputePlacement.BANK_LOCAL_LOGIC:
            if self.arrays.count != self.bank_groups:
                raise ConfigurationError(
                    "bank-local architecture requires one declared array per bank group"
                )

    @property
    def banks_per_group(self) -> int:
        return self.sram.total_banks // self.bank_groups

    def bank_group_for_bank(self, bank_id: int) -> int:
        if bank_id < 0 or bank_id >= self.sram.total_banks:
            raise ConfigurationError(f"invalid bank id {bank_id}")
        return bank_id // self.banks_per_group

    def local_bank_for_address(self, address: int, group_id: int) -> int:
        if group_id < 0 or group_id >= self.bank_groups:
            raise ConfigurationError(f"invalid bank group {group_id}")
        local = (address // self.sram.line_bytes) % self.banks_per_group
        return group_id * self.banks_per_group + local
